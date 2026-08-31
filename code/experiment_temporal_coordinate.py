from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from transient_response_completion import (
    ANCHOR_GRID,
    ScaledScalarRBFGP,
    interpolate_batch_to_anchors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "transition_batches_long.csv"
TEST_BATCH_PATH = PROJECT_ROOT / "evaluation" / "selected_test_batches.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "temporal_coordinate_experiment"
OBSERVED_ANCHOR_COUNT = 8
HISTORY_SIZE = 9
OUTPUTS = ["Cb", "T"]
TIME_REPRESENTATIONS = ["anchor_index", "actual_time"]
MODEL_RANDOM_STATE = 42


def feature_columns(output, representation):
    temporal_feature = "anchor_index" if representation == "anchor_index" else "time"
    return [
        temporal_feature,
        "u1_frac_A",
        "u1_Tc_scaled",
        "delta_frac_A",
        "delta_Tc_scaled",
        f"{output}_0",
    ]


def build_anchor_dataset(df, test_batches):
    needed = set(test_batches)
    for test_batch in test_batches:
        needed.update(range(test_batch - HISTORY_SIZE, test_batch))

    frames = []
    for batch_id in sorted(needed):
        batch = df[df["batch_id"] == batch_id]
        frames.append(interpolate_batch_to_anchors(batch, ANCHOR_GRID))
    anchor_df = pd.concat(frames, ignore_index=True)

    for output in OUTPUTS:
        initial_values = (
            anchor_df.sort_values("anchor_index")
            .groupby("batch_id")[output]
            .first()
            .to_dict()
        )
        anchor_df[f"{output}_0"] = anchor_df["batch_id"].map(initial_values)
    return anchor_df


def run_experiment(anchor_df, test_batches):
    results = []
    models = []

    for test_number, test_batch in enumerate(test_batches, start=1):
        test_df = (
            anchor_df[anchor_df["batch_id"] == test_batch]
            .sort_values("anchor_index")
            .reset_index(drop=True)
        )
        history_batches = list(range(test_batch - HISTORY_SIZE, test_batch))
        history_df = anchor_df[anchor_df["batch_id"].isin(history_batches)].copy()
        observed_df = test_df.iloc[:OBSERVED_ANCHOR_COUNT].copy()
        hidden_indices = np.arange(OBSERVED_ANCHOR_COUNT, len(test_df))
        if len(hidden_indices) == 0:
            raise ValueError(f"Batch {test_batch} has no hidden anchors")
        train_df = pd.concat([history_df, observed_df], ignore_index=True)

        for representation in TIME_REPRESENTATIONS:
            for output in OUTPUTS:
                features = feature_columns(output, representation)
                model = ScaledScalarRBFGP(random_state=MODEL_RANDOM_STATE).fit(
                    train_df[features].to_numpy(float),
                    train_df[output].to_numpy(float),
                )
                pred_mean, pred_std = model.predict(test_df[features].to_numpy(float))
                true = test_df[output].to_numpy(float)
                hidden_errors = pred_mean[hidden_indices] - true[hidden_indices]
                final_error = abs(float(pred_mean[-1]) - float(true[-1]))

                results.append({
                    "test_number": test_number,
                    "test_batch": test_batch,
                    "output": output,
                    "time_representation": representation,
                    "observed_anchor_count": OBSERVED_ANCHOR_COUNT,
                    "last_observed_time_s": float(observed_df["time"].iloc[-1]),
                    "history_size": HISTORY_SIZE,
                    "history_batches": ",".join(map(str, history_batches)),
                    "n_train_points": len(train_df),
                    "n_hidden_anchors": len(hidden_indices),
                    "hidden_mae": float(np.mean(np.abs(hidden_errors))),
                    "hidden_rmse": float(np.sqrt(np.mean(hidden_errors ** 2))),
                    "final_abs_error": final_error,
                    "final_squared_error": final_error ** 2,
                    "final_posterior_std": float(pred_std[-1]),
                    "final_true": float(true[-1]),
                    "final_predicted": float(pred_mean[-1]),
                })
                models.append({
                    "test_batch": test_batch,
                    "output": output,
                    "time_representation": representation,
                    "features": ",".join(features),
                    "learned_kernel": str(model.gp.kernel_),
                })

        print(f"Completed test {test_number:02d}/{len(test_batches)}: batch {test_batch}", flush=True)

    return pd.DataFrame(results), pd.DataFrame(models)


def summarize(results):
    rows = []
    for (output, representation), group in results.groupby(["output", "time_representation"]):
        rows.append({
            "output": output,
            "time_representation": representation,
            "n_tests": len(group),
            "mean_final_mae": float(group["final_abs_error"].mean()),
            "median_final_abs_error": float(group["final_abs_error"].median()),
            "sd_final_abs_error": float(group["final_abs_error"].std(ddof=1)),
            "ci95_final_mae": float(1.96 * group["final_abs_error"].std(ddof=1) / np.sqrt(len(group))),
            "final_rmse": float(np.sqrt(group["final_squared_error"].mean())),
            "mean_hidden_mae": float(group["hidden_mae"].mean()),
            "sd_hidden_mae": float(group["hidden_mae"].std(ddof=1)),
            "ci95_hidden_mae": float(1.96 * group["hidden_mae"].std(ddof=1) / np.sqrt(len(group))),
            "mean_hidden_rmse": float(group["hidden_rmse"].mean()),
            "mean_final_posterior_std": float(group["final_posterior_std"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["output", "time_representation"])


def paired_summary(results):
    rows = []
    for output in OUTPUTS:
        data = results[results["output"] == output]
        for metric in ["final_abs_error", "hidden_mae", "final_posterior_std"]:
            pivot = data.pivot(index="test_batch", columns="time_representation", values=metric)
            difference = pivot["actual_time"] - pivot["anchor_index"]
            test = wilcoxon(
                pivot["actual_time"], pivot["anchor_index"],
                alternative="two-sided", zero_method="wilcox",
            )
            rows.append({
                "output": output,
                "metric": metric,
                "mean_anchor_index": float(pivot["anchor_index"].mean()),
                "mean_actual_time": float(pivot["actual_time"].mean()),
                "actual_time_percent_change": float(
                    100 * (pivot["actual_time"].mean() / pivot["anchor_index"].mean() - 1)
                ),
                "actual_time_better_count": int((difference < 0).sum()),
                "anchor_index_better_count": int((difference > 0).sum()),
                "ties": int((difference == 0).sum()),
                "wilcoxon_statistic": float(test.statistic),
                "wilcoxon_p_value": float(test.pvalue),
            })
    return pd.DataFrame(rows)


def plot_results(summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = {"anchor_index": "Anchor index", "actual_time": "Actual time"}
    colors = {"anchor_index": "#1764ab", "actual_time": "#e07a2d"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))

    for column, output in enumerate(OUTPUTS):
        data = summary[summary["output"] == output].set_index("time_representation")
        order = TIME_REPRESENTATIONS
        for row, metric_type in enumerate(["final", "hidden"]):
            ax = axes[row, column]
            if metric_type == "final":
                values = [data.loc[r, "mean_final_mae"] for r in order]
                errors = [data.loc[r, "ci95_final_mae"] for r in order]
                title_suffix = "final steady-state prediction"
            else:
                values = [data.loc[r, "mean_hidden_mae"] for r in order]
                errors = [data.loc[r, "ci95_hidden_mae"] for r in order]
                title_suffix = "hidden trajectory"

            x = np.arange(len(order))
            ax.bar(
                x, values, yerr=errors, capsize=4,
                color=[colors[r] for r in order], edgecolor="white", width=0.68,
            )
            ax.set_xticks(x, [labels[r] for r in order])
            ax.set_ylabel("Mean absolute error (95% CI)")
            ax.set_title(rf"${output if output == 'T' else 'C_B'}$: {title_suffix}", loc="left", fontweight="bold")
            ax.grid(axis="y", color="#dddddd", lw=0.7)
            ax.set_axisbelow(True)
            ax.text(-0.16, 1.03, chr(ord("a") + row * 2 + column), transform=ax.transAxes, fontweight="bold")

    fig.suptitle("Effect of temporal-coordinate representation on GP prediction", fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_DIR / "temporal_coordinate_performance.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "temporal_coordinate_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    test_batches = pd.read_csv(TEST_BATCH_PATH)["test_batch"].astype(int).tolist()
    anchor_df = build_anchor_dataset(df, test_batches)
    results, models = run_experiment(anchor_df, test_batches)
    summary = summarize(results)
    paired = paired_summary(results)

    results.to_csv(OUTPUT_DIR / "temporal_coordinate_all_tests.csv", index=False)
    models.to_csv(OUTPUT_DIR / "temporal_coordinate_learned_kernels.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "temporal_coordinate_summary.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "temporal_coordinate_paired_summary.csv", index=False)
    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as handle:
        json.dump({
            "data_path": str(DATA_PATH),
            "test_batch_path": str(TEST_BATCH_PATH),
            "time_representations": TIME_REPRESENTATIONS,
            "normalized_time": "not tested because normalizing by true settling time would leak unavailable future information",
            "observed_anchor_count": OBSERVED_ANCHOR_COUNT,
            "last_observed_time_s": 180,
            "history_size": HISTORY_SIZE,
            "history_selection": "nine immediately preceding consecutive batches",
            "n_tests": len(test_batches),
            "model_random_state": MODEL_RANDOM_STATE,
            "outputs": OUTPUTS,
        }, handle, indent=2)

    plot_results(summary)
    print("\nSummary:\n", summary.to_string(index=False))
    print("\nPaired comparison:\n", paired.to_string(index=False))
    print(f"\nSaved outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
