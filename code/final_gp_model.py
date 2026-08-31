"""Final GP transient-completion model used in the dissertation.

Configuration
-------------
* first 8 anchors of the new transition are observed;
* transient progress is represented by anchor index;
* 30 preceding historical transitions are selected by nearest-neighbour
  distance in (f_A,0, T_c,0, delta f_A, delta T_c);
* separate scalar RBF GPs predict C_B and T;
* the GP input is (anchor index, f_A,1, T_c,1, delta f_A, delta T_c, y_0).

The script evaluates the configuration on the fixed test batches used by the
comparative experiments and writes predictions, selected histories, timing,
kernel and objective accuracy summaries to ``final_model_outputs``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from transient_response_completion import (
    ANCHOR_GRID,
    ScaledScalarRBFGP,
    interpolate_batch_to_anchors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "transition_batches_long.csv"
TEST_BATCH_PATH = PROJECT_ROOT / "evaluation" / "selected_test_batches.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_model_outputs"

OBSERVED_ANCHOR_COUNT = 8
HISTORY_SIZE = 30
OUTPUTS = ("Cb", "T")
MODEL_RANDOM_STATE = 42

SELECTION_FEATURES = (
    "u0_frac_A",
    "u0_Tc_scaled",
    "delta_frac_A",
    "delta_Tc_scaled",
)


def model_features(output: str) -> list[str]:
    return [
        "anchor_index",
        "u1_frac_A",
        "u1_Tc_scaled",
        "delta_frac_A",
        "delta_Tc_scaled",
        f"{output}_0",
    ]


def batch_metadata(df: pd.DataFrame) -> pd.DataFrame:
    order_col = "sample_id" if "sample_id" in df.columns else "time"
    meta = (
        df.sort_values(["batch_id", order_col])
        .groupby("batch_id")
        .first()[["u0_frac_A", "u0_Tc_scaled", "u1_frac_A", "u1_Tc_scaled"]]
        .copy()
    )
    meta["delta_frac_A"] = meta["u1_frac_A"] - meta["u0_frac_A"]
    meta["delta_Tc_scaled"] = meta["u1_Tc_scaled"] - meta["u0_Tc_scaled"]
    return meta


def select_nearest_histories(test_batch: int, meta: pd.DataFrame) -> pd.DataFrame:
    """Select 30 operationally nearest transitions available before test_batch."""
    candidates = meta.loc[meta.index < test_batch].copy()
    if len(candidates) < HISTORY_SIZE:
        raise ValueError(
            f"Batch {test_batch} has only {len(candidates)} preceding transitions; "
            f"{HISTORY_SIZE} are required."
        )

    # These four manipulated-variable descriptors are already scaled in the
    # generated dataset; use the same Euclidean rule as the final experiment.
    candidate_x = candidates[list(SELECTION_FEATURES)].to_numpy(float)
    query_x = meta.loc[test_batch, list(SELECTION_FEATURES)].to_numpy(float)
    distances = np.linalg.norm(candidate_x - query_x, axis=1)

    selected = candidates.iloc[np.argsort(distances)[:HISTORY_SIZE]].copy()
    selected["selection_distance"] = distances[np.argsort(distances)[:HISTORY_SIZE]]
    selected["selection_rank"] = np.arange(1, HISTORY_SIZE + 1)
    return selected


def build_anchor_dataset(df: pd.DataFrame, needed_batches: set[int]) -> pd.DataFrame:
    frames = []
    for batch in sorted(needed_batches):
        batch_df = df[df["batch_id"] == batch]
        if batch_df.empty:
            raise ValueError(f"Batch {batch} is missing from the transient dataset.")
        frames.append(interpolate_batch_to_anchors(batch_df, ANCHOR_GRID))

    anchors = pd.concat(frames, ignore_index=True)
    for output in OUTPUTS:
        initial = (
            anchors.sort_values("anchor_index")
            .groupby("batch_id")[output]
            .first()
            .to_dict()
        )
        anchors[f"{output}_0"] = anchors["batch_id"].map(initial)
    return anchors


def evaluate_one_test(
    test_number: int,
    test_batch: int,
    anchors: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    selected = select_nearest_histories(test_batch, meta)
    history_ids = selected.index.astype(int).tolist()

    test = (
        anchors[anchors["batch_id"] == test_batch]
        .sort_values("anchor_index")
        .reset_index(drop=True)
    )
    if len(test) <= OBSERVED_ANCHOR_COUNT:
        raise ValueError(f"Batch {test_batch} has no hidden anchors after anchor 8.")

    observed = test.iloc[:OBSERVED_ANCHOR_COUNT]
    hidden_mask = np.arange(len(test)) >= OBSERVED_ANCHOR_COUNT
    history = anchors[anchors["batch_id"].isin(history_ids)]
    train = pd.concat([history, observed], ignore_index=True)

    selection_rows = []
    for history_batch, row in selected.iterrows():
        selection_rows.append({
            "test_number": test_number,
            "test_batch": test_batch,
            "history_batch": int(history_batch),
            "selection_rank": int(row["selection_rank"]),
            "selection_distance": float(row["selection_distance"]),
        })

    prediction_rows, test_rows, kernel_rows = [], [], []
    for output in OUTPUTS:
        features = model_features(output)
        fit_start = perf_counter()
        model = ScaledScalarRBFGP(random_state=MODEL_RANDOM_STATE).fit(
            train[features].to_numpy(float), train[output].to_numpy(float)
        )
        fit_seconds = perf_counter() - fit_start

        prediction_start = perf_counter()
        mean, std = model.predict(test[features].to_numpy(float))
        prediction_seconds = perf_counter() - prediction_start
        true = test[output].to_numpy(float)
        error = mean - true

        for i, row in test.iterrows():
            prediction_rows.append({
                "test_number": test_number,
                "test_batch": test_batch,
                "output": output,
                "anchor_index": int(row["anchor_index"]),
                "time": float(row["time"]),
                "observed": bool(not hidden_mask[i]),
                "true": float(true[i]),
                "predicted": float(mean[i]),
                "posterior_std": float(std[i]),
                "error": float(error[i]),
                "abs_error": float(abs(error[i])),
            })

        hidden_error = error[hidden_mask]
        hidden_true = true[hidden_mask]
        hidden_mean = mean[hidden_mask]
        hidden_std = std[hidden_mask]
        test_rows.append({
            "test_number": test_number,
            "test_batch": test_batch,
            "output": output,
            "n_histories": HISTORY_SIZE,
            "n_training_samples": len(train),
            "last_observed_time": float(observed["time"].iloc[-1]),
            "settling_time": float(test["time"].iloc[-1]),
            "observation_fraction": float(observed["time"].iloc[-1] / test["time"].iloc[-1]),
            "final_true": float(true[-1]),
            "final_predicted": float(mean[-1]),
            "final_error": float(error[-1]),
            "final_abs_error": float(abs(error[-1])),
            "final_posterior_std": float(std[-1]),
            "hidden_mae": float(np.mean(np.abs(hidden_error))),
            "hidden_rmse": float(np.sqrt(np.mean(hidden_error**2))),
            "hidden_95_coverage": float(np.mean(np.abs(hidden_true-hidden_mean) <= 1.96*hidden_std)),
            "fit_seconds": float(fit_seconds),
            "prediction_seconds": float(prediction_seconds),
        })
        kernel_rows.append({
            "test_batch": test_batch,
            "output": output,
            "features": ", ".join(features),
            "learned_kernel": str(model.gp.kernel_),
        })

    return prediction_rows, test_rows, selection_rows, kernel_rows


def objective_summary(per_test: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Return scale-aware final and hidden-trajectory performance metrics."""
    rows = []
    for output in OUTPUTS:
        tests = per_test[per_test["output"] == output].copy()
        hidden = predictions[(predictions["output"] == output) & (~predictions["observed"])].copy()

        final_range = float(tests["final_true"].max() - tests["final_true"].min())
        hidden_range = float(hidden["true"].max() - hidden["true"].min())
        final_errors = tests["final_abs_error"].to_numpy(float)
        safe_final_range = final_range if final_range > 0 else np.nan
        safe_hidden_range = hidden_range if hidden_range > 0 else np.nan
        final_r2 = (float(r2_score(tests["final_true"], tests["final_predicted"]))
                    if len(tests) >= 2 and final_range > 0 else np.nan)
        within = lambda fraction: (float(100*np.mean(final_errors <= fraction*final_range))
                                   if final_range > 0 else np.nan)

        rows.append({
            "output": output,
            "n_tests": len(tests),
            "final_mae": float(np.mean(final_errors)),
            "final_median_ae": float(np.median(final_errors)),
            "final_rmse": float(np.sqrt(np.mean(tests["final_error"]**2))),
            "final_p90_ae": float(np.quantile(final_errors, 0.90)),
            "final_max_ae": float(np.max(final_errors)),
            "final_test_range": final_range,
            "final_nmae_percent_of_range": float(100*np.mean(final_errors)/safe_final_range),
            "final_r2": final_r2,
            "final_within_1pct_range_percent": within(.01),
            "final_within_2pct_range_percent": within(.02),
            "final_within_5pct_range_percent": within(.05),
            # Each transition receives equal weight for trajectory metrics.
            "mean_hidden_mae": float(tests["hidden_mae"].mean()),
            "mean_hidden_rmse": float(tests["hidden_rmse"].mean()),
            "hidden_test_range": hidden_range,
            "hidden_nmae_percent_of_range": float(100*tests["hidden_mae"].mean()/safe_hidden_range),
            "mean_hidden_95_coverage_percent": float(100*tests["hidden_95_coverage"].mean()),
            "mean_observation_fraction_percent": float(100*tests["observation_fraction"].mean()),
            "mean_waiting_time_reduction_percent": float(100*(1-tests["observation_fraction"].mean())),
            "median_fit_seconds": float(tests["fit_seconds"].median()),
            "median_complete_prediction_ms": float(1000*tests["prediction_seconds"].median()),
        })
    return pd.DataFrame(rows)


def plot_final_parity(per_test: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    for ax, output in zip(axes, OUTPUTS):
        data = per_test[per_test["output"] == output]
        lo = min(data["final_true"].min(), data["final_predicted"].min())
        hi = max(data["final_true"].max(), data["final_predicted"].max())
        pad = .05*(hi-lo) if hi > lo else 1.0
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color="black", lw=1, ls="--")
        ax.scatter(data["final_true"], data["final_predicted"], s=25,
                   facecolor="white", edgecolor="black", linewidth=.9)
        symbol = r"$T$ (K)" if output == "T" else r"$C_B$"
        ax.set_xlabel(f"True final {symbol}")
        ax.set_ylabel(f"Predicted final {symbol}")
        ax.grid(color="#dddddd", lw=.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir/"final_model_parity.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir/"final_model_parity.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--tests", type=Path, default=TEST_BATCH_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-tests", type=int, default=None,
                        help="Optional small smoke-test limit; omit for the full evaluation.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    meta = batch_metadata(df)
    tests = pd.read_csv(args.tests)["test_batch"].astype(int).tolist()
    if args.max_tests is not None:
        tests = tests[:args.max_tests]

    selected_by_test = {b: select_nearest_histories(b, meta).index.astype(int).tolist()
                        for b in tests}
    needed_batches = set(tests)
    for ids in selected_by_test.values():
        needed_batches.update(ids)
    anchors = build_anchor_dataset(df, needed_batches)

    prediction_rows, test_rows, selection_rows, kernel_rows = [], [], [], []
    for number, test_batch in enumerate(tests, 1):
        pred, test, selected, kernels = evaluate_one_test(number, test_batch, anchors, meta)
        prediction_rows.extend(pred); test_rows.extend(test)
        selection_rows.extend(selected); kernel_rows.extend(kernels)
        print(f"Completed {number:02d}/{len(tests)}: batch {test_batch}", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    per_test = pd.DataFrame(test_rows)
    selections = pd.DataFrame(selection_rows)
    kernels = pd.DataFrame(kernel_rows)
    summary = objective_summary(per_test, predictions)

    predictions.to_csv(args.output_dir/"final_model_predictions.csv", index=False)
    per_test.to_csv(args.output_dir/"final_model_per_test_metrics.csv", index=False)
    selections.to_csv(args.output_dir/"final_model_selected_histories.csv", index=False)
    kernels.to_csv(args.output_dir/"final_model_learned_kernels.csv", index=False)
    summary.to_csv(args.output_dir/"final_model_objective_summary.csv", index=False)
    plot_final_parity(per_test, args.output_dir)

    config = {
        "observed_anchor_count": OBSERVED_ANCHOR_COUNT,
        "history_size": HISTORY_SIZE,
        "temporal_coordinate": "anchor_index",
        "selection_features": list(SELECTION_FEATURES),
        "candidate_pool": "transitions preceding each test batch",
        "model_features": {output: model_features(output) for output in OUTPUTS},
        "categorical_direction_encoding": False,
        "transition_distance_model_feature": False,
        "model_random_state": MODEL_RANDOM_STATE,
        "n_tests": len(tests),
    }
    with open(args.output_dir/"final_model_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print("\nFinal model objective performance\n")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
