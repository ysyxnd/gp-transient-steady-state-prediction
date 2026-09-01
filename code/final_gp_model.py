"""Final GP transient-completion model used in the dissertation.

Configuration
-------------
* first 8 anchors of the new transition are observed;
* transient progress is represented by anchor index;
* 30 preceding historical transitions are selected by nearest-neighbour
  distance in (f_A,0, T_c,0, delta f_A, delta T_c);
* separate scalar RBF GPs predict C_B and T;
* the GP input is (anchor index, f_A,1, T_c,1, delta f_A, delta T_c, y_0).
* the true settling endpoint is withheld during prediction; a common endpoint
  is detected from convergence of the two predicted output trajectories over
  a fixed future anchor range.

By default, the script evaluates the configuration on the fixed test batches
used by the comparative experiments.  The additional holdout evaluation can
be reproduced with ``--additional-holdout-size 50``.  Outputs are written to
the directory supplied through ``--output-dir``.
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "reproduced_final_model_outputs"

OBSERVED_ANCHOR_COUNT = 8
HISTORY_SIZE = 30
OUTPUTS = ("Cb", "T")
MODEL_RANDOM_STATE = 42
ADDITIONAL_TEST_SEED = 20260901

# These deployment settings are fixed for every test transition.  Prediction
# is restricted to the predeclared nominal anchor grid; an occasionally
# appended, trajectory-specific settling anchor is deliberately excluded.
# The 2% normalised-change threshold is aligned with the settling tolerance
# used to generate the dataset and must not be retuned on the test outcomes.
MAX_PREDICTION_INDEX = len(ANCHOR_GRID) - 1
CONVERGENCE_THRESHOLD_FRACTION = 0.02
CONSECUTIVE_STABLE_INTERVALS = 2

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


def select_nearest_histories(
    test_batch: int,
    meta: pd.DataFrame,
    excluded_batches: set[int] | None = None,
) -> pd.DataFrame:
    """Select 30 operationally nearest transitions available before test_batch."""
    candidates = meta.loc[meta.index < test_batch].copy()
    if excluded_batches:
        candidates = candidates.loc[~candidates.index.isin(excluded_batches)]
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


def select_additional_holdout_tests(
    df: pd.DataFrame,
    original_tests: list[int],
    n_tests: int,
    seed: int,
    additional_exclusions: set[int] | None = None,
) -> list[int]:
    """Select a fixed holdout without examining prediction performance.

    Eligibility uses only batch identity and whether at least eight anchors can
    be revealed while leaving a hidden continuation.  The minimum batch ID
    guarantees that 30 preceding non-holdout histories remain available even
    if every other holdout happens to occur earlier in the sequence.
    """
    if n_tests <= 0:
        raise ValueError("The additional holdout size must be positive.")

    final_times = df.groupby("batch_id")["time"].max()
    last_observed_time = float(ANCHOR_GRID[OBSERVED_ANCHOR_COUNT - 1])
    minimum_batch = HISTORY_SIZE + n_tests
    original_set = set(map(int, original_tests))
    exclusion_set = original_set | set(additional_exclusions or set())
    eligible = np.array([
        int(batch)
        for batch, final_time in final_times.items()
        if int(batch) >= minimum_batch
        and int(batch) not in exclusion_set
        and float(final_time) > last_observed_time
    ], dtype=int)

    if len(eligible) < n_tests:
        raise ValueError(
            f"Only {len(eligible)} eligible additional holdout tests are available; "
            f"{n_tests} were requested."
        )

    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(eligible, size=n_tests, replace=False).tolist())
    holdout_set = set(selected)
    for test_batch in selected:
        available_prior = [
            batch for batch in final_times.index
            if int(batch) < test_batch and int(batch) not in holdout_set
        ]
        if len(available_prior) < HISTORY_SIZE:
            raise RuntimeError(
                f"Holdout batch {test_batch} has only {len(available_prior)} "
                "preceding non-holdout histories."
            )
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


def build_fixed_prediction_frame(observed: pd.DataFrame) -> pd.DataFrame:
    """Create deployment inputs without using the hidden test trajectory.

    All non-temporal model inputs are known at the operating change or from
    the first observed sample.  The prediction horizon is identical for every
    test transition and therefore contains no information about its true
    settling endpoint.
    """
    template = observed.iloc[0]
    indices = np.arange(0, MAX_PREDICTION_INDEX + 1, dtype=int)
    frame = pd.DataFrame({"anchor_index": indices})
    for column in (
        "u1_frac_A", "u1_Tc_scaled", "delta_frac_A", "delta_Tc_scaled",
        "Cb_0", "T_0",
    ):
        frame[column] = float(template[column])
    return frame


def detect_predicted_endpoint(
    predicted_means: dict[str, np.ndarray],
    observed: pd.DataFrame,
    output_scales: dict[str, float],
) -> tuple[int, bool, pd.DataFrame]:
    """Detect a joint endpoint from predicted Cb and T convergence.

    Changes are measured from the last observed value to the first future
    prediction and then between successive future predictions.  Both outputs
    must remain below the fixed normalised threshold for the requested number
    of consecutive intervals.  No true future value or true endpoint is used.
    """
    future_indices = np.arange(
        OBSERVED_ANCHOR_COUNT, MAX_PREDICTION_INDEX + 1, dtype=int
    )
    change_columns: dict[str, np.ndarray] = {}
    stable = np.ones(len(future_indices), dtype=bool)

    for output in OUTPUTS:
        previous_and_future = np.concatenate((
            [float(observed[output].iloc[-1])],
            predicted_means[output][future_indices],
        ))
        scale = max(float(output_scales[output]), np.finfo(float).eps)
        normalised_change = np.abs(np.diff(previous_and_future)) / scale
        change_columns[f"{output}_normalised_change"] = normalised_change
        stable &= normalised_change < CONVERGENCE_THRESHOLD_FRACTION

    endpoint = None
    for end_position in range(CONSECUTIVE_STABLE_INTERVALS - 1, len(stable)):
        start_position = end_position - CONSECUTIVE_STABLE_INTERVALS + 1
        if stable[start_position:end_position + 1].all():
            endpoint = int(future_indices[end_position])
            break

    identified = endpoint is not None
    if endpoint is None:
        endpoint = MAX_PREDICTION_INDEX

    diagnostics = pd.DataFrame({
        "anchor_index": future_indices,
        **change_columns,
        "joint_stable_interval": stable,
    })
    diagnostics["predicted_endpoint"] = diagnostics["anchor_index"] == endpoint
    return endpoint, identified, diagnostics


def evaluate_one_test(
    test_number: int,
    test_batch: int,
    anchors: pd.DataFrame,
    meta: pd.DataFrame,
    excluded_history_batches: set[int] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    selected = select_nearest_histories(
        test_batch, meta, excluded_batches=excluded_history_batches
    )
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

    # Only these revealed rows are supplied when constructing test inputs.
    # The remaining rows in `test` are retained solely for retrospective
    # scoring after endpoint detection has finished.
    prediction_frame = build_fixed_prediction_frame(observed)
    output_scales = {
        output: float(train[output].max() - train[output].min())
        for output in OUTPUTS
    }

    prediction_rows, test_rows, kernel_rows = [], [], []
    predicted_means: dict[str, np.ndarray] = {}
    predicted_stds: dict[str, np.ndarray] = {}
    fit_times: dict[str, float] = {}
    prediction_times: dict[str, float] = {}

    for output in OUTPUTS:
        features = model_features(output)
        fit_start = perf_counter()
        model = ScaledScalarRBFGP(random_state=MODEL_RANDOM_STATE).fit(
            train[features].to_numpy(float), train[output].to_numpy(float)
        )
        fit_seconds = perf_counter() - fit_start

        prediction_start = perf_counter()
        mean, std = model.predict(prediction_frame[features].to_numpy(float))
        prediction_times[output] = perf_counter() - prediction_start
        predicted_means[output] = mean
        predicted_stds[output] = std
        fit_times[output] = fit_seconds

        kernel_rows.append({
            "test_batch": test_batch,
            "output": output,
            "features": ", ".join(features),
            "learned_kernel": str(model.gp.kernel_),
        })

    predicted_endpoint, endpoint_identified, diagnostics = detect_predicted_endpoint(
        predicted_means, observed, output_scales
    )
    diagnostic_lookup = diagnostics.set_index("anchor_index")
    true_endpoint = int(test["anchor_index"].iloc[-1])
    true_lookup = test.set_index("anchor_index")

    for output in OUTPUTS:
        mean = predicted_means[output]
        std = predicted_stds[output]
        true = test[output].to_numpy(float)

        # Store the fixed prediction range.  True values are attached only
        # afterwards for retrospective scoring and are left missing beyond the
        # simulated trajectory's actual endpoint.
        for _, row in prediction_frame.iterrows():
            anchor_index = int(row["anchor_index"])
            has_true = anchor_index in true_lookup.index
            true_value = (float(true_lookup.loc[anchor_index, output])
                          if has_true else np.nan)
            predicted_value = float(mean[anchor_index])
            error = predicted_value - true_value if has_true else np.nan
            is_observed = anchor_index < OBSERVED_ANCHOR_COUNT
            prediction_rows.append({
                "test_number": test_number,
                "test_batch": test_batch,
                "output": output,
                "anchor_index": anchor_index,
                "observed": bool(is_observed),
                "within_true_trajectory": bool(has_true),
                "true": true_value,
                "predicted": predicted_value,
                "posterior_std": float(std[anchor_index]),
                "error": float(error) if has_true else np.nan,
                "abs_error": float(abs(error)) if has_true else np.nan,
                "normalised_change": (
                    float(diagnostic_lookup.loc[anchor_index, f"{output}_normalised_change"])
                    if anchor_index >= OBSERVED_ANCHOR_COUNT else np.nan
                ),
                "joint_stable_interval": (
                    bool(diagnostic_lookup.loc[anchor_index, "joint_stable_interval"])
                    if anchor_index >= OBSERVED_ANCHOR_COUNT else False
                ),
                "predicted_endpoint": bool(anchor_index == predicted_endpoint),
                "endpoint_identified": bool(endpoint_identified),
            })

        comparable = test[
            (test["anchor_index"] >= OBSERVED_ANCHOR_COUNT)
            & (test["anchor_index"] <= MAX_PREDICTION_INDEX)
        ]
        comparable_indices = comparable["anchor_index"].to_numpy(int)
        hidden_true = comparable[output].to_numpy(float)
        hidden_mean = mean[comparable_indices]
        hidden_std = std[comparable_indices]
        hidden_error = hidden_mean - hidden_true

        final_true = float(true[-1])
        final_predicted = float(mean[predicted_endpoint])
        final_error = final_predicted - final_true
        test_rows.append({
            "test_number": test_number,
            "test_batch": test_batch,
            "output": output,
            "n_histories": HISTORY_SIZE,
            "n_training_samples": len(train),
            "last_observed_time": float(observed["time"].iloc[-1]),
            "settling_time": float(test["time"].iloc[-1]),
            "observation_fraction": float(observed["time"].iloc[-1] / test["time"].iloc[-1]),
            "true_endpoint_index": true_endpoint,
            "predicted_endpoint_index": predicted_endpoint,
            "endpoint_identified": bool(endpoint_identified),
            "endpoint_index_error": int(predicted_endpoint - true_endpoint),
            "endpoint_abs_index_error": int(abs(predicted_endpoint - true_endpoint)),
            "final_true": final_true,
            "final_predicted": final_predicted,
            "final_error": float(final_error),
            "final_abs_error": float(abs(final_error)),
            "final_posterior_std": float(std[predicted_endpoint]),
            "hidden_mae": float(np.mean(np.abs(hidden_error))),
            "hidden_rmse": float(np.sqrt(np.mean(hidden_error**2))),
            "hidden_95_coverage": float(np.mean(np.abs(hidden_true-hidden_mean) <= 1.96*hidden_std)),
            "fit_seconds": float(fit_times[output]),
            "prediction_seconds": float(prediction_times[output]),
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
            "endpoint_identification_rate_percent": float(100*tests["endpoint_identified"].mean()),
            "endpoint_index_mae": float(tests["endpoint_abs_index_error"].mean()),
            "endpoint_index_median_ae": float(tests["endpoint_abs_index_error"].median()),
            "unresolved_tests": int((~tests["endpoint_identified"]).sum()),
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
    parser.add_argument(
        "--additional-holdout-size", type=int, default=None,
        help=("Select this many new fixed holdout transitions, excluding the test "
              "batches listed by --tests and excluding all selected holdouts from "
              "one another's historical candidate pools."),
    )
    parser.add_argument(
        "--additional-test-seed", type=int, default=ADDITIONAL_TEST_SEED,
        help="Random seed used only when --additional-holdout-size is supplied.",
    )
    parser.add_argument(
        "--exclude-batches", type=int, nargs="*", default=[],
        help=("Additional batches excluded from holdout selection, "
              "for example cases already used during implementation smoke tests."),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    meta = batch_metadata(df)
    original_tests = pd.read_csv(args.tests)["test_batch"].astype(int).tolist()
    additional_holdout = args.additional_holdout_size is not None
    if additional_holdout:
        tests = select_additional_holdout_tests(
            df, original_tests, args.additional_holdout_size,
            args.additional_test_seed, set(args.exclude_batches),
        )
        excluded_history_batches = set(tests)
        pd.DataFrame({"test_batch": tests}).to_csv(
            args.output_dir / "selected_additional_holdout_batches.csv", index=False
        )
    else:
        tests = original_tests
        excluded_history_batches = set()
    if args.max_tests is not None:
        tests = tests[:args.max_tests]

    selected_by_test = {b: select_nearest_histories(
                            b, meta, excluded_batches=excluded_history_batches
                        ).index.astype(int).tolist()
                        for b in tests}
    needed_batches = set(tests)
    for ids in selected_by_test.values():
        needed_batches.update(ids)
    anchors = build_anchor_dataset(df, needed_batches)

    prediction_rows, test_rows, selection_rows, kernel_rows = [], [], [], []
    for number, test_batch in enumerate(tests, 1):
        pred, test, selected, kernels = evaluate_one_test(
            number, test_batch, anchors, meta,
            excluded_history_batches=excluded_history_batches,
        )
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
        "true_endpoint_used_for_prediction": False,
        "max_prediction_index": MAX_PREDICTION_INDEX,
        "convergence_threshold_fraction": CONVERGENCE_THRESHOLD_FRACTION,
        "consecutive_stable_intervals": CONSECUTIVE_STABLE_INTERVALS,
        "endpoint_rule": "joint normalised convergence of predicted Cb and T",
        "unresolved_rule": "use prediction at fixed maximum index and flag unresolved",
        "evaluation_protocol": (
            "additional holdout" if additional_holdout else "configured test batches"
        ),
        "additional_holdout_size": (
            args.additional_holdout_size if additional_holdout else None
        ),
        "additional_test_seed": (
            args.additional_test_seed if additional_holdout else None
        ),
        "holdouts_excluded_from_history_pool": bool(additional_holdout),
        "original_configuration_tests_excluded_from_holdout_selection": bool(additional_holdout),
        "additional_holdout_selection_exclusions": (
            sorted(set(args.exclude_batches)) if additional_holdout else []
        ),
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
