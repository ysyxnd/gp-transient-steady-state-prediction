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
OUTPUT_DIR = PROJECT_ROOT / "results" / "transition_direction_experiment"
STRATEGIES = ["unrestricted", "same_delta_frac_A", "same_delta_Tc", "same_both"]
DESCRIPTOR = ["u0_frac_A", "u0_Tc_scaled", "delta_frac_A", "delta_Tc_scaled"]


def direction_mask(candidates, query, strategy):
    mask = np.ones(len(candidates), dtype=bool)
    if strategy in ["same_delta_frac_A", "same_both"]:
        mask &= np.sign(candidates.delta_frac_A.to_numpy()) == np.sign(query.delta_frac_A)
    if strategy in ["same_delta_Tc", "same_both"]:
        mask &= np.sign(candidates.delta_Tc_scaled.to_numpy()) == np.sign(query.delta_Tc_scaled)
    return mask


def candidate_counts(test_batch, meta):
    candidates = meta.loc[meta.index < test_batch]
    query = meta.loc[test_batch]
    return {s: int(direction_mask(candidates, query, s).sum()) for s in STRATEGIES}


def select_histories(test_batch, strategy, meta):
    candidates = meta.loc[meta.index < test_batch]
    query = meta.loc[test_batch]
    candidates = candidates.loc[direction_mask(candidates, query, strategy)]
    distance = np.linalg.norm(candidates[DESCRIPTOR].to_numpy(float) -
                              query[DESCRIPTOR].to_numpy(float), axis=1)
    return candidates.index[np.argsort(distance)[:base.HISTORY_SIZE]].tolist()


def paired(results):
    rows = []
    for output in base.OUTPUTS:
        data = results[results.output == output]
        for strategy in STRATEGIES[1:]:
            for metric in ["final_abs_error", "hidden_mae", "final_posterior_std"]:
                p = data.pivot(index="test_batch", columns="strategy", values=metric)
                diff = p[strategy] - p.unrestricted
                test = wilcoxon(p[strategy], p.unrestricted)
                rows.append({"output": output, "strategy": strategy, "metric": metric,
                             "mean_unrestricted": p.unrestricted.mean(),
                             "mean_candidate": p[strategy].mean(),
                             "percent_change_vs_unrestricted": 100*(p[strategy].mean()/p.unrestricted.mean()-1),
                             "candidate_better_count": int((diff < 0).sum()),
                             "unrestricted_better_count": int((diff > 0).sum()),
                             "ties": int((diff == 0).sum()),
                             "wilcoxon_statistic": test.statistic,
                             "wilcoxon_p_value": test.pvalue})
    return pd.DataFrame(rows)


def plot(summary):
    labels = ["Unrestricted", "Same $\\Delta f_A$\nsign",
              "Same $\\Delta T_c$\nsign", "Same signs\nof both"]
    colors = ["#1764ab", "#55a868", "#dd8452", "#8172b3"]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.3), sharex=True)
    for col, output in enumerate(base.OUTPUTS):
        d = summary[summary.output == output].set_index("strategy")
        for row, (metric, ci, title) in enumerate([
            ("mean_final_mae", "ci95_final_mae", "final-value MAE"),
            ("mean_hidden_mae", "ci95_hidden_mae", "hidden-trajectory MAE")]):
            ax = axes[row, col]
            vals = d.loc[STRATEGIES, metric].to_numpy()
            errs = d.loc[STRATEGIES, ci].to_numpy()
            ax.bar(np.arange(4), vals, yerr=errs, capsize=3, color=colors, edgecolor="white")
            symbol = r"$T$" if output == "T" else r"$C_B$"
            ax.set_title(f"{symbol}: {title}", loc="left", fontweight="bold")
            ax.set_ylabel("Mean absolute error (95% CI)")
            ax.grid(axis="y", color="#dddddd", lw=.7); ax.set_axisbelow(True)
            ax.text(-.13, 1.03, chr(ord('a')+row*2+col), transform=ax.transAxes, fontweight="bold")
            ax.set_xticks(np.arange(4), labels, fontsize=9)
    fig.suptitle("Effect of transition-direction classification", fontweight="bold", y=.995)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(OUTPUT_DIR/"transition_direction_performance.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR/"transition_direction_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(base.DATA_PATH)
    all_tests = pd.read_csv(base.TEST_BATCH_PATH).test_batch.astype(int).tolist()
    meta = base.batch_metadata(df)
    counts = pd.DataFrame([{"test_batch": b, **candidate_counts(b, meta)} for b in all_tests])
    tests = counts.loc[counts[STRATEGIES].min(axis=1) >= base.HISTORY_SIZE, "test_batch"].tolist()
    excluded = [b for b in all_tests if b not in tests]
    anchors = base.build_anchor_dataset(df, max(tests))
    base.STRATEGIES = STRATEGIES
    base.select_histories = select_histories
    results, selections, kernels = base.run(anchors, meta, tests)
    summary, comparisons = base.summarize(results), paired(results)
    results.to_csv(OUTPUT_DIR/"transition_direction_all_tests.csv", index=False)
    selections.to_csv(OUTPUT_DIR/"selected_history_batches.csv", index=False)
    kernels.to_csv(OUTPUT_DIR/"transition_direction_learned_kernels.csv", index=False)
    summary.to_csv(OUTPUT_DIR/"transition_direction_summary.csv", index=False)
    comparisons.to_csv(OUTPUT_DIR/"transition_direction_paired_vs_unrestricted.csv", index=False)
    counts.to_csv(OUTPUT_DIR/"direction_candidate_counts.csv", index=False)
    pd.DataFrame({"test_batch": tests}).to_csv(OUTPUT_DIR/"selected_test_batches.csv", index=False)
    with open(OUTPUT_DIR/"experiment_config.json", "w", encoding="utf-8") as f:
        json.dump({"strategies": STRATEGIES, "history_size": base.HISTORY_SIZE,
                   "similarity_descriptor": DESCRIPTOR,
                   "direction_definition": "sign, with zero treated as a separate class",
                   "n_tests": len(tests), "excluded_test_batches": excluded,
                   "exclusion_reason": "fewer than 30 candidates for at least one direction strategy"}, f, indent=2)
    plot(summary)
    print("\nSummary:\n", summary.to_string(index=False))
    print("\nPaired against unrestricted:\n", comparisons.to_string(index=False))
    print("\nExcluded:", excluded)


if __name__ == "__main__":
    main()
