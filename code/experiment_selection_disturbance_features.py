from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import experiment_history_selection as base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "selection_disturbance_feature_experiment"
STRATEGIES = ["full", "without_delta_frac_A", "without_delta_Tc", "initial_only", "disturbance_only"]
DESCRIPTORS = {
    "full": ["u0_frac_A", "u0_Tc_scaled", "delta_frac_A", "delta_Tc_scaled"],
    "without_delta_frac_A": ["u0_frac_A", "u0_Tc_scaled", "delta_Tc_scaled"],
    "without_delta_Tc": ["u0_frac_A", "u0_Tc_scaled", "delta_frac_A"],
    "initial_only": ["u0_frac_A", "u0_Tc_scaled"],
    "disturbance_only": ["delta_frac_A", "delta_Tc_scaled"],
}


def select_histories(test_batch, strategy, meta):
    candidates = meta.loc[meta.index < test_batch]
    columns = DESCRIPTORS[strategy]
    query = meta.loc[test_batch, columns].to_numpy(float)
    distance = np.linalg.norm(candidates[columns].to_numpy(float) - query, axis=1)
    return candidates.index[np.argsort(distance)[:base.HISTORY_SIZE]].tolist()


def plot(summary):
    labels = ["Full", "Without\n$\\Delta f_A$", "Without\n$\\Delta T_c$",
              "Initial\nonly", "Disturbance\nonly"]
    colors = ["#1764ab", "#55a868", "#c44e52", "#8172b3", "#dd8452"]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True)
    for col, output in enumerate(base.OUTPUTS):
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
    fig.suptitle("Disturbance features used for history selection", fontweight="bold", y=.995)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(OUTPUT_DIR/"selection_disturbance_features.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR/"selection_disturbance_features.pdf", bbox_inches="tight")
    plt.close(fig)


def paired(results):
    rows = []
    for output in base.OUTPUTS:
        data = results[results.output == output]
        for strategy in STRATEGIES[1:]:
            for metric in ["final_abs_error", "hidden_mae", "final_posterior_std"]:
                p = data.pivot(index="test_batch", columns="strategy", values=metric)
                diff = p[strategy] - p["full"]
                test = wilcoxon(p[strategy], p["full"])
                rows.append({"output": output, "strategy": strategy, "metric": metric,
                             "mean_full": p.full.mean(), "mean_candidate": p[strategy].mean(),
                             "percent_change_vs_full": 100*(p[strategy].mean()/p.full.mean()-1),
                             "candidate_better_count": int((diff < 0).sum()),
                             "full_better_count": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                             "wilcoxon_statistic": test.statistic, "wilcoxon_p_value": test.pvalue})
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(base.DATA_PATH)
    tests = pd.read_csv(base.TEST_BATCH_PATH).test_batch.astype(int).tolist()
    meta = base.batch_metadata(df)
    anchors = base.build_anchor_dataset(df, max(tests))
    base.STRATEGIES = STRATEGIES
    base.select_histories = select_histories
    results, selections, kernels = base.run(anchors, meta, tests)
    summary, comparisons = base.summarize(results), paired(results)
    results.to_csv(OUTPUT_DIR/"selection_disturbance_all_tests.csv", index=False)
    selections.to_csv(OUTPUT_DIR/"selected_history_batches.csv", index=False)
    kernels.to_csv(OUTPUT_DIR/"selection_disturbance_learned_kernels.csv", index=False)
    summary.to_csv(OUTPUT_DIR/"selection_disturbance_summary.csv", index=False)
    comparisons.to_csv(OUTPUT_DIR/"selection_disturbance_paired_vs_full.csv", index=False)
    with open(OUTPUT_DIR/"experiment_config.json", "w", encoding="utf-8") as f:
        json.dump({"strategies": STRATEGIES, "descriptors": DESCRIPTORS,
                   "history_size": base.HISTORY_SIZE,
                   "candidate_pool": "all transitions preceding each test batch",
                   "distance": "Euclidean distance in scaled coordinates",
                   "n_tests": len(tests)}, f, indent=2)
    plot(summary)
    print("\nSummary:\n", summary.to_string(index=False))
    print("\nPaired against full descriptor:\n", comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
