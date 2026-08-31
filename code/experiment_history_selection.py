from pathlib import Path
import json
from time import perf_counter

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
OUTPUT_DIR = PROJECT_ROOT / "results" / "history_selection_experiment"
HISTORY_SIZE = 30
OBSERVED_ANCHOR_COUNT = 8
OUTPUTS = ["Cb", "T"]
STRATEGIES = ["recent", "random", "nearest_initial", "nearest_target", "nearest_transition"]
RANDOM_SEED = 20260829
MODEL_RANDOM_STATE = 42


def model_features(output):
    return ["anchor_index", "u1_frac_A", "u1_Tc_scaled",
            "delta_frac_A", "delta_Tc_scaled", f"{output}_0"]


def batch_metadata(df):
    meta = (df.sort_values(["batch_id", "sample_id"]).groupby("batch_id").first()
            [["u0_frac_A", "u0_Tc_scaled", "u1_frac_A", "u1_Tc_scaled"]].copy())
    meta["delta_frac_A"] = meta.u1_frac_A - meta.u0_frac_A
    meta["delta_Tc_scaled"] = meta.u1_Tc_scaled - meta.u0_Tc_scaled
    return meta


def select_histories(test_batch, strategy, meta):
    candidates = meta.loc[meta.index < test_batch].copy()
    if len(candidates) < HISTORY_SIZE:
        raise ValueError(f"Batch {test_batch} has only {len(candidates)} prior transitions")
    if strategy == "recent":
        return candidates.index[-HISTORY_SIZE:].tolist()
    if strategy == "random":
        rng = np.random.default_rng(RANDOM_SEED + int(test_batch))
        return sorted(rng.choice(candidates.index.to_numpy(), HISTORY_SIZE, replace=False).tolist())
    columns = {
        "nearest_initial": ["u0_frac_A", "u0_Tc_scaled"],
        "nearest_target": ["u1_frac_A", "u1_Tc_scaled"],
        "nearest_transition": ["u0_frac_A", "u0_Tc_scaled", "delta_frac_A", "delta_Tc_scaled"],
    }[strategy]
    query = meta.loc[test_batch, columns].to_numpy(float)
    distance = np.linalg.norm(candidates[columns].to_numpy(float) - query, axis=1)
    return candidates.index[np.argsort(distance)[:HISTORY_SIZE]].tolist()


def build_anchor_dataset(df, max_batch):
    frames = [interpolate_batch_to_anchors(df[df.batch_id == b], ANCHOR_GRID)
              for b in range(max_batch + 1)]
    anchors = pd.concat(frames, ignore_index=True)
    for output in OUTPUTS:
        initial = (anchors.sort_values("anchor_index").groupby("batch_id")[output]
                   .first().to_dict())
        anchors[f"{output}_0"] = anchors.batch_id.map(initial)
    return anchors


def run(anchors, meta, tests):
    rows, selections, kernels = [], [], []
    for number, test_batch in enumerate(tests, 1):
        test = (anchors[anchors.batch_id == test_batch]
                .sort_values("anchor_index").reset_index(drop=True))
        observed = test.iloc[:OBSERVED_ANCHOR_COUNT]
        hidden = np.arange(OBSERVED_ANCHOR_COUNT, len(test))
        for strategy in STRATEGIES:
            start = perf_counter()
            history_ids = select_histories(test_batch, strategy, meta)
            selection_ms = 1000 * (perf_counter() - start)
            history = anchors[anchors.batch_id.isin(history_ids)]
            train = pd.concat([history, observed], ignore_index=True)
            selections.append({"test_batch": test_batch, "strategy": strategy,
                               "history_batches": ",".join(map(str, history_ids)),
                               "selection_time_ms": selection_ms})
            for output in OUTPUTS:
                cols = model_features(output)
                model = ScaledScalarRBFGP(random_state=MODEL_RANDOM_STATE).fit(
                    train[cols].to_numpy(float), train[output].to_numpy(float))
                mean, std = model.predict(test[cols].to_numpy(float))
                true = test[output].to_numpy(float)
                err = mean[hidden] - true[hidden]
                final = abs(float(mean[-1]) - float(true[-1]))
                rows.append({"test_number": number, "test_batch": test_batch,
                             "output": output, "strategy": strategy,
                             "hidden_mae": np.mean(np.abs(err)),
                             "hidden_rmse": np.sqrt(np.mean(err**2)),
                             "final_abs_error": final,
                             "final_squared_error": final**2,
                             "final_posterior_std": float(std[-1]),
                             "n_training_samples": len(train),
                             "selection_time_ms": selection_ms})
                kernels.append({"test_batch": test_batch, "output": output,
                                "strategy": strategy, "learned_kernel": str(model.gp.kernel_)})
        print(f"Completed test {number:02d}/{len(tests)}: batch {test_batch}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(selections), pd.DataFrame(kernels)


def summarize(results):
    rows = []
    for (output, strategy), g in results.groupby(["output", "strategy"]):
        rows.append({"output": output, "strategy": strategy, "n_tests": len(g),
                     "mean_final_mae": g.final_abs_error.mean(),
                     "ci95_final_mae": 1.96*g.final_abs_error.std(ddof=1)/np.sqrt(len(g)),
                     "final_rmse": np.sqrt(g.final_squared_error.mean()),
                     "mean_hidden_mae": g.hidden_mae.mean(),
                     "ci95_hidden_mae": 1.96*g.hidden_mae.std(ddof=1)/np.sqrt(len(g)),
                     "mean_hidden_rmse": g.hidden_rmse.mean(),
                     "mean_final_posterior_std": g.final_posterior_std.mean(),
                     "median_selection_time_ms": g.selection_time_ms.median()})
    return pd.DataFrame(rows)


def paired(results):
    rows = []
    for output in OUTPUTS:
        d = results[results.output == output]
        for strategy in STRATEGIES[1:]:
            for metric in ["final_abs_error", "hidden_mae", "final_posterior_std"]:
                p = d.pivot(index="test_batch", columns="strategy", values=metric)
                diff = p[strategy] - p["recent"]
                test = wilcoxon(p[strategy], p["recent"])
                rows.append({"output": output, "strategy": strategy, "metric": metric,
                             "mean_recent": p.recent.mean(), "mean_candidate": p[strategy].mean(),
                             "percent_change_vs_recent": 100*(p[strategy].mean()/p.recent.mean()-1),
                             "candidate_better_count": int((diff < 0).sum()),
                             "recent_better_count": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                             "wilcoxon_statistic": test.statistic, "wilcoxon_p_value": test.pvalue})
    return pd.DataFrame(rows)


def plot(summary):
    labels = ["Recent", "Random", "Nearest\ninitial", "Nearest\ntarget", "Nearest\ntransition"]
    colors = ["#1764ab", "#8c8c8c", "#42a5b3", "#e5902f", "#8e62aa"]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True)
    for col, output in enumerate(OUTPUTS):
        d = summary[summary.output == output].set_index("strategy")
        for row, (metric, ci, title) in enumerate([
            ("mean_final_mae", "ci95_final_mae", "final-value MAE"),
            ("mean_hidden_mae", "ci95_hidden_mae", "hidden-trajectory MAE")]):
            ax = axes[row, col]
            vals = d.loc[STRATEGIES, metric].to_numpy()
            errs = d.loc[STRATEGIES, ci].to_numpy()
            ax.bar(np.arange(5), vals, yerr=errs, capsize=3, color=colors, edgecolor="white")
            symbol = r"$T$" if output == "T" else r"$C_B$"
            ax.set_title(f"{symbol}: {title}", loc="left", fontweight="bold")
            ax.set_ylabel("Mean absolute error (95% CI)")
            ax.grid(axis="y", color="#dddddd", lw=.7); ax.set_axisbelow(True)
            ax.text(-.13, 1.03, chr(ord('a')+row*2+col), transform=ax.transAxes, fontweight="bold")
            ax.set_xticks(np.arange(5), labels)
    fig.suptitle("Effect of history-selection strategy", fontweight="bold", y=.995)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(OUTPUT_DIR/"history_selection_performance.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR/"history_selection_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    tests = pd.read_csv(TEST_BATCH_PATH).test_batch.astype(int).tolist()
    meta = batch_metadata(df)
    anchors = build_anchor_dataset(df, max(tests))
    results, selections, kernels = run(anchors, meta, tests)
    summary, comparisons = summarize(results), paired(results)
    results.to_csv(OUTPUT_DIR/"history_selection_all_tests.csv", index=False)
    selections.to_csv(OUTPUT_DIR/"selected_history_batches.csv", index=False)
    kernels.to_csv(OUTPUT_DIR/"history_selection_learned_kernels.csv", index=False)
    summary.to_csv(OUTPUT_DIR/"history_selection_summary.csv", index=False)
    comparisons.to_csv(OUTPUT_DIR/"history_selection_paired_vs_recent.csv", index=False)
    with open(OUTPUT_DIR/"experiment_config.json", "w", encoding="utf-8") as f:
        json.dump({"strategies": STRATEGIES, "history_size": HISTORY_SIZE,
                   "observed_anchor_count": OBSERVED_ANCHOR_COUNT,
                   "random_seed": RANDOM_SEED, "model_random_state": MODEL_RANDOM_STATE,
                   "distance_scaling": "manipulated variables already scaled",
                   "candidate_pool": "all transitions preceding each test batch",
                   "n_tests": len(tests)}, f, indent=2)
    plot(summary)
    print("\nSummary:\n", summary.to_string(index=False))
    print("\nPaired against recent:\n", comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
