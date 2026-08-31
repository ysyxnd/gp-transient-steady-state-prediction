from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transient_response_completion import (
    ANCHOR_GRID,
    ScaledScalarRBFGP,
    feature_cols_for,
    interpolate_batch_to_anchors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "transition_batches_long.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "anchor_count_experiment"
ANCHOR_COUNTS = list(range(3, 13))
HISTORY_SIZE = 9
N_TESTS = 30
TEST_SELECTION_SEED = 123
MODEL_RANDOM_STATE = 42
OUTPUTS = ["Cb", "T"]


def select_test_batches(df):
    all_batches = set(map(int, df["batch_id"].unique()))
    final_times = df.groupby("batch_id")["time"].max()
    max_observed = max(ANCHOR_COUNTS)
    candidates = []

    for batch_id, final_time in final_times.items():
        batch_id = int(batch_id)
        anchors = [a for a in ANCHOR_GRID if a <= float(final_time) + 1e-12]
        if len(anchors) == 0 or not np.isclose(anchors[-1], float(final_time)):
            anchors.append(float(final_time))
        anchors = sorted(set(np.round(anchors, 12)))
        previous = list(range(batch_id - HISTORY_SIZE, batch_id))
        if len(anchors) > max_observed and all(p in all_batches for p in previous):
            candidates.append(batch_id)

    if len(candidates) < N_TESTS:
        raise ValueError(f"Only {len(candidates)} eligible test transitions were available")

    rng = np.random.default_rng(TEST_SELECTION_SEED)
    return sorted(map(int, rng.choice(candidates, size=N_TESTS, replace=False)))


def build_anchor_dataset(df, test_batches):
    needed = set(test_batches)
    for batch_id in test_batches:
        needed.update(range(batch_id - HISTORY_SIZE, batch_id))

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
    records = []
    model_records = []

    for test_number, test_batch in enumerate(test_batches, start=1):
        test_df = (
            anchor_df[anchor_df["batch_id"] == test_batch]
            .sort_values("anchor_index")
            .reset_index(drop=True)
        )
        history_batches = list(range(test_batch - HISTORY_SIZE, test_batch))
        history_df = anchor_df[anchor_df["batch_id"].isin(history_batches)].copy()

        for observed_count in ANCHOR_COUNTS:
            observed_df = test_df.iloc[:observed_count].copy()
            hidden_df = test_df.iloc[observed_count:].copy()
            if hidden_df.empty:
                raise ValueError(f"Batch {test_batch} has no hidden anchors for {observed_count}")

            train_df = pd.concat([history_df, observed_df], ignore_index=True)
            last_observed_time = float(observed_df["time"].iloc[-1])
            final_time = float(test_df["time"].iloc[-1])

            for output in OUTPUTS:
                feature_cols = feature_cols_for(output)
                model = ScaledScalarRBFGP(random_state=MODEL_RANDOM_STATE).fit(
                    train_df[feature_cols].to_numpy(float),
                    train_df[output].to_numpy(float),
                )
                pred_mean, pred_std = model.predict(test_df[feature_cols].to_numpy(float))
                hidden_indices = np.arange(observed_count, len(test_df))
                hidden_true = test_df.iloc[hidden_indices][output].to_numpy(float)
                hidden_pred = pred_mean[hidden_indices]
                hidden_abs_errors = np.abs(hidden_pred - hidden_true)
                final_error = abs(float(pred_mean[-1]) - float(test_df[output].iloc[-1]))

                records.append({
                    "test_number": test_number,
                    "test_batch": test_batch,
                    "output": output,
                    "observed_anchor_count": observed_count,
                    "last_observed_time_s": last_observed_time,
                    "final_time_s": final_time,
                    "observed_fraction_of_settling_time": last_observed_time / final_time,
                    "waiting_time_reduction": 1.0 - last_observed_time / final_time,
                    "history_size": HISTORY_SIZE,
                    "history_batches": ",".join(map(str, history_batches)),
                    "n_train_points": len(train_df),
                    "n_hidden_anchors": len(hidden_indices),
                    "hidden_mae": float(np.mean(hidden_abs_errors)),
                    "hidden_rmse": float(np.sqrt(np.mean((hidden_pred - hidden_true) ** 2))),
                    "final_abs_error": final_error,
                    "final_squared_error": final_error ** 2,
                    "final_posterior_std": float(pred_std[-1]),
                    "final_true": float(test_df[output].iloc[-1]),
                    "final_predicted": float(pred_mean[-1]),
                })
                model_records.append({
                    "test_batch": test_batch,
                    "output": output,
                    "observed_anchor_count": observed_count,
                    "learned_kernel": str(model.gp.kernel_),
                })

        print(f"Completed test {test_number:02d}/{len(test_batches)}: batch {test_batch}", flush=True)

    return pd.DataFrame(records), pd.DataFrame(model_records)


def summarize(results):
    rows = []
    for (output, count), group in results.groupby(["output", "observed_anchor_count"]):
        rows.append({
            "output": output,
            "observed_anchor_count": int(count),
            "last_observed_time_s": float(group["last_observed_time_s"].iloc[0]),
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
            "mean_waiting_time_reduction": float(group["waiting_time_reduction"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["output", "observed_anchor_count"])


def plot_results(summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharex=True)
    colors = {"Cb": "#1764ab", "T": "#c44536"}
    titles = {
        ("Cb", "final"): r"$C_B$: final steady-state prediction",
        ("T", "final"): r"$T$: final steady-state prediction",
        ("Cb", "hidden"): r"$C_B$: hidden trajectory",
        ("T", "hidden"): r"$T$: hidden trajectory",
    }

    for column, output in enumerate(OUTPUTS):
        data = summary[summary["output"] == output]
        x = data["observed_anchor_count"].to_numpy()
        for row, kind in enumerate(["final", "hidden"]):
            ax = axes[row, column]
            if kind == "final":
                y = data["mean_final_mae"].to_numpy()
                error = data["ci95_final_mae"].to_numpy()
            else:
                y = data["mean_hidden_mae"].to_numpy()
                error = data["ci95_hidden_mae"].to_numpy()
            ax.errorbar(
                x, y, yerr=error, marker="o", ms=4.5, lw=1.7, capsize=2.5,
                color=colors[output], ecolor="#8b8b8b",
            )
            ax.set_title(titles[(output, kind)], loc="left", fontweight="bold")
            ax.set_ylabel("Mean absolute error (95% CI)")
            ax.grid(color="#dddddd", lw=0.7)
            ax.set_xticks(ANCHOR_COUNTS)
            ax.text(-0.16, 1.03, chr(ord("a") + row * 2 + column), transform=ax.transAxes, fontweight="bold")

    axes[1, 0].set_xlabel("Number of observed anchors")
    axes[1, 1].set_xlabel("Number of observed anchors")
    fig.suptitle("Effect of observed trajectory length on GP prediction", fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_DIR / "anchor_count_performance.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "anchor_count_performance.pdf", bbox_inches="tight")
    plt.close(fig)

    cb = summary[summary["output"] == "Cb"].copy()
    fig, ax1 = plt.subplots(figsize=(7.2, 4.5))
    ax1.plot(cb["observed_anchor_count"], 100 * cb["mean_waiting_time_reduction"], marker="o", color="#2a9d8f", lw=2)
    ax1.set_xlabel("Number of observed anchors")
    ax1.set_ylabel("Mean remaining settling time (%)", color="#1f776d")
    ax1.tick_params(axis="y", colors="#1f776d")
    ax1.set_xticks(ANCHOR_COUNTS)
    ax1.grid(color="#dddddd", lw=0.7)
    ax2 = ax1.twinx()
    ax2.plot(cb["observed_anchor_count"], cb["last_observed_time_s"], marker="s", color="#4d4d4d", lw=1.7)
    ax2.set_ylabel("Last observed time (s)", color="#4d4d4d")
    ax2.tick_params(axis="y", colors="#4d4d4d")
    ax1.set_title("Observation length and potential waiting-time reduction", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "anchor_count_waiting_time.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "anchor_count_waiting_time.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    test_batches = select_test_batches(df)
    anchor_df = build_anchor_dataset(df, test_batches)
    results, models = run_experiment(anchor_df, test_batches)
    summary = summarize(results)

    results.to_csv(OUTPUT_DIR / "anchor_count_all_tests.csv", index=False)
    models.to_csv(OUTPUT_DIR / "anchor_count_learned_kernels.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "anchor_count_summary.csv", index=False)
    pd.DataFrame({"test_batch": test_batches}).to_csv(OUTPUT_DIR / "selected_test_batches.csv", index=False)
    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as handle:
        json.dump({
            "data_path": str(DATA_PATH),
            "anchor_counts": ANCHOR_COUNTS,
            "anchor_grid_s": ANCHOR_GRID.tolist(),
            "history_size": HISTORY_SIZE,
            "history_selection": "nine immediately preceding consecutive batches",
            "n_tests": N_TESTS,
            "test_selection_seed": TEST_SELECTION_SEED,
            "model_random_state": MODEL_RANDOM_STATE,
            "outputs": OUTPUTS,
        }, handle, indent=2)

    plot_results(summary)
    print("\nSummary:\n", summary.to_string(index=False))
    print(f"\nSaved outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
