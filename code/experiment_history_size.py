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
OUTPUT_DIR = PROJECT_ROOT / "results" / "history_size_experiment"
HISTORY_SIZES = [1, 3, 5, 7, 9, 12, 15, 20, 30]
BASELINE_HISTORY_SIZE = 9
OBSERVED_ANCHOR_COUNT = 8
OUTPUTS = ["Cb", "T"]
MODEL_RANDOM_STATE = 42
PREDICTION_TIMING_REPEATS = 50


def features(output):
    return ["anchor_index", "u1_frac_A", "u1_Tc_scaled",
            "delta_frac_A", "delta_Tc_scaled", f"{output}_0"]


def build_anchor_dataset(df, test_batches):
    needed = set(test_batches)
    for test_batch in test_batches:
        needed.update(range(test_batch - max(HISTORY_SIZES), test_batch))
    frames = [interpolate_batch_to_anchors(df[df.batch_id == b], ANCHOR_GRID)
              for b in sorted(needed)]
    anchor_df = pd.concat(frames, ignore_index=True)
    for output in OUTPUTS:
        initial = (anchor_df.sort_values("anchor_index")
                   .groupby("batch_id")[output].first().to_dict())
        anchor_df[f"{output}_0"] = anchor_df.batch_id.map(initial)
    return anchor_df


def run(anchor_df, test_batches):
    rows, kernels = [], []
    for number, test_batch in enumerate(test_batches, 1):
        test = (anchor_df[anchor_df.batch_id == test_batch]
                .sort_values("anchor_index").reset_index(drop=True))
        observed = test.iloc[:OBSERVED_ANCHOR_COUNT]
        hidden = np.arange(OBSERVED_ANCHOR_COUNT, len(test))
        for history_size in HISTORY_SIZES:
            history_ids = range(test_batch - history_size, test_batch)
            train = pd.concat([anchor_df[anchor_df.batch_id.isin(history_ids)], observed], ignore_index=True)
            for output in OUTPUTS:
                cols = features(output)
                x_train = train[cols].to_numpy(float)
                y_train = train[output].to_numpy(float)
                x_test = test[cols].to_numpy(float)
                fit_start = perf_counter()
                model = ScaledScalarRBFGP(random_state=MODEL_RANDOM_STATE).fit(x_train, y_train)
                fit_seconds = perf_counter() - fit_start
                mean, std = model.predict(x_test)  # warm-up and retained prediction
                predict_start = perf_counter()
                for _ in range(PREDICTION_TIMING_REPEATS):
                    model.predict(x_test)
                predict_seconds = (perf_counter() - predict_start) / PREDICTION_TIMING_REPEATS
                true = test[output].to_numpy(float)
                err = mean[hidden] - true[hidden]
                final = abs(float(mean[-1]) - float(true[-1]))
                rows.append({"test_number": number, "test_batch": test_batch,
                             "output": output, "history_size": history_size,
                             "hidden_mae": np.mean(np.abs(err)),
                             "hidden_rmse": np.sqrt(np.mean(err ** 2)),
                             "final_abs_error": final,
                             "final_squared_error": final ** 2,
                             "final_posterior_std": float(std[-1]),
                             "n_training_samples": len(train),
                             "fit_seconds": fit_seconds,
                             "predict_seconds": predict_seconds})
                kernels.append({"test_batch": test_batch, "output": output,
                                "history_size": history_size,
                                "learned_kernel": str(model.gp.kernel_)})
        print(f"Completed test {number:02d}/{len(test_batches)}: batch {test_batch}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(kernels)


def summarize(results):
    rows = []
    for (output, size), g in results.groupby(["output", "history_size"]):
        rows.append({"output": output, "history_size": size, "n_tests": len(g),
                     "mean_final_mae": g.final_abs_error.mean(),
                     "ci95_final_mae": 1.96*g.final_abs_error.std(ddof=1)/np.sqrt(len(g)),
                     "final_rmse": np.sqrt(g.final_squared_error.mean()),
                     "mean_hidden_mae": g.hidden_mae.mean(),
                     "ci95_hidden_mae": 1.96*g.hidden_mae.std(ddof=1)/np.sqrt(len(g)),
                     "mean_hidden_rmse": g.hidden_rmse.mean(),
                     "mean_final_posterior_std": g.final_posterior_std.mean(),
                     "mean_training_samples": g.n_training_samples.mean(),
                     "median_fit_seconds": g.fit_seconds.median(),
                     "mean_fit_seconds": g.fit_seconds.mean(),
                     "median_predict_ms": 1000*g.predict_seconds.median(),
                     "mean_predict_ms": 1000*g.predict_seconds.mean()})
    return pd.DataFrame(rows).sort_values(["output", "history_size"])


def paired(results):
    rows = []
    for output in OUTPUTS:
        d = results[results.output == output]
        for size in HISTORY_SIZES:
            if size == BASELINE_HISTORY_SIZE:
                continue
            for metric in ["final_abs_error", "hidden_mae", "final_posterior_std"]:
                p = d.pivot(index="test_batch", columns="history_size", values=metric)
                diff = p[size] - p[BASELINE_HISTORY_SIZE]
                test = wilcoxon(p[size], p[BASELINE_HISTORY_SIZE])
                rows.append({"output": output, "history_size": size, "metric": metric,
                             "mean_baseline_9": p[BASELINE_HISTORY_SIZE].mean(),
                             "mean_candidate": p[size].mean(),
                             "percent_change_vs_9": 100*(p[size].mean()/p[BASELINE_HISTORY_SIZE].mean()-1),
                             "candidate_better_count": int((diff < 0).sum()),
                             "baseline_better_count": int((diff > 0).sum()),
                             "ties": int((diff == 0).sum()),
                             "wilcoxon_statistic": test.statistic,
                             "wilcoxon_p_value": test.pvalue})
    return pd.DataFrame(rows)


def plot(summary):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True)
    for col, output in enumerate(OUTPUTS):
        d = summary[summary.output == output].set_index("history_size")
        for row, (metric, ci, title) in enumerate([
            ("mean_final_mae", "ci95_final_mae", "final-value MAE"),
            ("mean_hidden_mae", "ci95_hidden_mae", "hidden-trajectory MAE")]):
            ax = axes[row, col]
            y = d.loc[HISTORY_SIZES, metric].to_numpy()
            e = d.loc[HISTORY_SIZES, ci].to_numpy()
            ax.errorbar(HISTORY_SIZES, y, yerr=e, marker="o", color="#1764ab",
                        lw=1.8, capsize=3)
            ax.axvline(BASELINE_HISTORY_SIZE, color="#d95f02", ls="--", lw=1.2,
                       label="Baseline: 9" if row == 0 and col == 0 else None)
            symbol = r"$T$" if output == "T" else r"$C_B$"
            ax.set_title(f"{symbol}: {title}", loc="left", fontweight="bold")
            ax.set_ylabel("Mean absolute error (95% CI)")
            ax.grid(color="#dddddd", lw=.7)
            ax.set_axisbelow(True)
            ax.text(-.13, 1.03, chr(ord('a') + row*2 + col), transform=ax.transAxes, fontweight="bold")
    for ax in axes[-1]:
        ax.set_xlabel("Number of historical transitions")
        ax.set_xticks(HISTORY_SIZES)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Effect of history size on GP prediction", fontweight="bold", y=.995)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(OUTPUT_DIR/"history_size_performance.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR/"history_size_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_computation(summary):
    timing = summary.groupby("history_size", as_index=False).agg(
        median_fit_seconds=("median_fit_seconds", "mean"),
        median_predict_ms=("median_predict_ms", "mean"),
        training_samples=("mean_training_samples", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
    axes[0].plot(timing.history_size, timing.median_fit_seconds, marker="o", color="#1764ab")
    axes[0].set_ylabel("Median fitting time (s)")
    axes[0].set_title("a   Model fitting", loc="left", fontweight="bold")
    axes[1].plot(timing.history_size, timing.median_predict_ms, marker="o", color="#d95f02")
    axes[1].set_ylabel("Median prediction time (ms)")
    axes[1].set_title("b   Trajectory prediction", loc="left", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Number of historical transitions")
        ax.set_xticks(HISTORY_SIZES)
        ax.grid(color="#dddddd", lw=.7)
        ax.set_axisbelow(True)
    fig.suptitle("Computational cost of increasing history size", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"history_size_computation.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR/"history_size_computation.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    all_tests = pd.read_csv(TEST_BATCH_PATH).test_batch.astype(int).tolist()
    tests = [b for b in all_tests if b >= max(HISTORY_SIZES)]
    excluded = [b for b in all_tests if b < max(HISTORY_SIZES)]
    anchor_df = build_anchor_dataset(df, tests)
    results, kernels = run(anchor_df, tests)
    summary, comparisons = summarize(results), paired(results)
    results.to_csv(OUTPUT_DIR/"history_size_all_tests.csv", index=False)
    kernels.to_csv(OUTPUT_DIR/"history_size_learned_kernels.csv", index=False)
    summary.to_csv(OUTPUT_DIR/"history_size_summary.csv", index=False)
    comparisons.to_csv(OUTPUT_DIR/"history_size_paired_vs_9.csv", index=False)
    pd.DataFrame({"test_batch": tests}).to_csv(OUTPUT_DIR/"selected_test_batches.csv", index=False)
    with open(OUTPUT_DIR/"experiment_config.json", "w", encoding="utf-8") as f:
        json.dump({"history_sizes": HISTORY_SIZES, "baseline_history_size": 9,
                   "observed_anchor_count": 8, "temporal_feature": "anchor_index",
                   "history_selection": "immediately preceding consecutive batches",
                   "n_tests": len(tests), "excluded_test_batches": excluded,
                   "exclusion_reason": "insufficient preceding batches for history size 30"}, f, indent=2)
    plot(summary)
    plot_computation(summary)
    print("\nSummary:\n", summary.to_string(index=False))
    print("\nPaired against history size 9:\n", comparisons.to_string(index=False))
    print("\nExcluded test batches:", excluded)


if __name__ == "__main__":
    main()
