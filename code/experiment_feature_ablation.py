from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from transient_response_completion import ANCHOR_GRID, ScaledScalarRBFGP, interpolate_batch_to_anchors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "transition_batches_long.csv"
TEST_BATCH_PATH = PROJECT_ROOT / "evaluation" / "selected_test_batches.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "feature_ablation_experiment"
OBSERVED_ANCHOR_COUNT = 8
HISTORY_SIZE = 9
OUTPUTS = ["Cb", "T"]
CONFIGURATIONS = [
    "baseline", "without_anchor_index", "without_u1_frac_A",
    "without_u1_Tc_scaled", "without_delta_frac_A",
    "without_delta_Tc_scaled", "without_initial_output",
]
MODEL_RANDOM_STATE = 42


def feature_columns(output, configuration):
    columns = ["anchor_index", "u1_frac_A", "u1_Tc_scaled",
               "delta_frac_A", "delta_Tc_scaled", f"{output}_0"]
    removal = {
        "without_anchor_index": "anchor_index",
        "without_u1_frac_A": "u1_frac_A",
        "without_u1_Tc_scaled": "u1_Tc_scaled",
        "without_delta_frac_A": "delta_frac_A",
        "without_delta_Tc_scaled": "delta_Tc_scaled",
        "without_initial_output": f"{output}_0",
    }
    if configuration in removal:
        columns.remove(removal[configuration])
    return columns


def build_anchor_dataset(df, test_batches):
    needed = set(test_batches)
    for test_batch in test_batches:
        needed.update(range(test_batch - HISTORY_SIZE, test_batch))
    frames = []
    for batch_id in sorted(needed):
        frames.append(interpolate_batch_to_anchors(
            df[df["batch_id"] == batch_id], ANCHOR_GRID))
    anchor_df = pd.concat(frames, ignore_index=True)
    for output in OUTPUTS:
        initial = (anchor_df.sort_values("anchor_index")
                   .groupby("batch_id")[output].first().to_dict())
        anchor_df[f"{output}_0"] = anchor_df["batch_id"].map(initial)
    return anchor_df


def run_experiment(anchor_df, test_batches):
    results, models = [], []
    for test_number, test_batch in enumerate(test_batches, start=1):
        test_df = (anchor_df[anchor_df["batch_id"] == test_batch]
                   .sort_values("anchor_index").reset_index(drop=True))
        history = anchor_df[anchor_df["batch_id"].isin(
            range(test_batch - HISTORY_SIZE, test_batch))]
        observed = test_df.iloc[:OBSERVED_ANCHOR_COUNT]
        train_df = pd.concat([history, observed], ignore_index=True)
        hidden = np.arange(OBSERVED_ANCHOR_COUNT, len(test_df))
        for configuration in CONFIGURATIONS:
            for output in OUTPUTS:
                features = feature_columns(output, configuration)
                model = ScaledScalarRBFGP(random_state=MODEL_RANDOM_STATE).fit(
                    train_df[features].to_numpy(float),
                    train_df[output].to_numpy(float))
                mean, std = model.predict(test_df[features].to_numpy(float))
                true = test_df[output].to_numpy(float)
                hidden_error = mean[hidden] - true[hidden]
                final_error = abs(float(mean[-1]) - float(true[-1]))
                results.append({
                    "test_number": test_number, "test_batch": test_batch,
                    "output": output, "configuration": configuration,
                    "hidden_mae": float(np.mean(np.abs(hidden_error))),
                    "hidden_rmse": float(np.sqrt(np.mean(hidden_error ** 2))),
                    "final_abs_error": final_error,
                    "final_squared_error": final_error ** 2,
                    "final_posterior_std": float(std[-1]),
                })
                models.append({
                    "test_batch": test_batch, "output": output,
                    "configuration": configuration,
                    "features": ",".join(features),
                    "learned_kernel": str(model.gp.kernel_),
                })
        print(f"Completed test {test_number:02d}/{len(test_batches)}: batch {test_batch}", flush=True)
    return pd.DataFrame(results), pd.DataFrame(models)


def summarize(results):
    rows = []
    for (output, configuration), group in results.groupby(["output", "configuration"]):
        rows.append({
            "output": output, "configuration": configuration, "n_tests": len(group),
            "mean_final_mae": group["final_abs_error"].mean(),
            "ci95_final_mae": 1.96 * group["final_abs_error"].std(ddof=1) / np.sqrt(len(group)),
            "final_rmse": np.sqrt(group["final_squared_error"].mean()),
            "mean_hidden_mae": group["hidden_mae"].mean(),
            "ci95_hidden_mae": 1.96 * group["hidden_mae"].std(ddof=1) / np.sqrt(len(group)),
            "mean_hidden_rmse": group["hidden_rmse"].mean(),
            "mean_final_posterior_std": group["final_posterior_std"].mean(),
        })
    return pd.DataFrame(rows)


def paired_summary(results):
    rows = []
    for output in OUTPUTS:
        data = results[results["output"] == output]
        for configuration in CONFIGURATIONS[1:]:
            for metric in ["final_abs_error", "hidden_mae", "final_posterior_std"]:
                pivot = data.pivot(index="test_batch", columns="configuration", values=metric)
                diff = pivot[configuration] - pivot["baseline"]
                test = wilcoxon(pivot[configuration], pivot["baseline"], alternative="two-sided")
                rows.append({
                    "output": output, "configuration": configuration, "metric": metric,
                    "mean_baseline": pivot["baseline"].mean(),
                    "mean_ablated": pivot[configuration].mean(),
                    "percent_change": 100 * (pivot[configuration].mean() / pivot["baseline"].mean() - 1),
                    "ablated_better_count": int((diff < 0).sum()),
                    "baseline_better_count": int((diff > 0).sum()),
                    "ties": int((diff == 0).sum()),
                    "wilcoxon_statistic": test.statistic,
                    "wilcoxon_p_value": test.pvalue,
                })
    return pd.DataFrame(rows)


def plot_results(paired):
    labels = ["Anchor\nindex", "Initial\n$f_A$", "Initial\n$T_c$",
              "$\\Delta f_A$", "$\\Delta T_c$", "Initial\noutput"]
    configs = CONFIGURATIONS[1:]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True)
    for col, output in enumerate(OUTPUTS):
        for row, (metric, title) in enumerate([
            ("final_abs_error", "final-value MAE"),
            ("hidden_mae", "hidden-trajectory MAE")]):
            ax = axes[row, col]
            d = paired[(paired.output == output) & (paired.metric == metric)].set_index("configuration")
            vals = [d.loc[c, "percent_change"] for c in configs]
            colors = ["#c94c3b" if v > 0 else "#2c7fb8" for v in vals]
            ax.bar(np.arange(len(vals)), vals, color=colors, width=.72)
            ax.axhline(0, color="#333333", lw=.8)
            for i, c in enumerate(configs):
                if d.loc[c, "wilcoxon_p_value"] < .05:
                    ax.text(i, vals[i], "*", ha="center", va="bottom" if vals[i] >= 0 else "top", fontweight="bold")
            symbol = r"$T$" if output == "T" else r"$C_B$"
            ax.set_title(f"{symbol}: {title}", loc="left", fontweight="bold")
            ax.set_ylabel("Change from baseline (%)")
            ax.grid(axis="y", color="#dddddd", lw=.7)
            ax.set_axisbelow(True)
            ax.set_xticks(np.arange(len(labels)), labels, rotation=0)
            ax.text(-.12, 1.03, chr(ord('a') + row * 2 + col), transform=ax.transAxes, fontweight="bold")
    fig.suptitle("Leave-one-feature-out analysis", fontweight="bold", y=.995)
    fig.text(.99, .01, "* paired Wilcoxon p < 0.05", ha="right", fontsize=9)
    fig.tight_layout(rect=[0, .025, 1, .97])
    fig.savefig(OUTPUT_DIR / "feature_ablation_performance.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "feature_ablation_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    test_batches = pd.read_csv(TEST_BATCH_PATH)["test_batch"].astype(int).tolist()
    anchor_df = build_anchor_dataset(df, test_batches)
    results, models = run_experiment(anchor_df, test_batches)
    summary, paired = summarize(results), paired_summary(results)
    results.to_csv(OUTPUT_DIR / "feature_ablation_all_tests.csv", index=False)
    models.to_csv(OUTPUT_DIR / "feature_ablation_learned_kernels.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "feature_ablation_summary.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "feature_ablation_paired_summary.csv", index=False)
    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as handle:
        json.dump({"data_path": str(DATA_PATH), "test_batch_path": str(TEST_BATCH_PATH),
                   "configurations": CONFIGURATIONS, "observed_anchor_count": OBSERVED_ANCHOR_COUNT,
                   "temporal_feature": "anchor_index", "history_size": HISTORY_SIZE,
                   "history_selection": "nine immediately preceding consecutive batches",
                   "n_tests": len(test_batches), "outputs": OUTPUTS}, handle, indent=2)
    plot_results(paired)
    print("\nSummary:\n", summary.to_string(index=False))
    print("\nPaired comparisons:\n", paired.to_string(index=False))
    print(f"\nSaved outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
