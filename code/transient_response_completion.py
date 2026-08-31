import os
import glob
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# User settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "transition_batches_long.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "gp_completion_outputs_multi_tests"

OBSERVED_ANCHOR_COUNT = 8
OUTPUTS = ["Cb", "T"]

# Number of different test-batch combinations to run
N_TEST_COMBINATIONS = 6

# Random seed only affects which test batches are selected
RANDOM_SEED = 123

# Same logic as your original script:
# compare 3, 6, and 9 previous runs
HISTORY_SIZES = [3, 6, 9]

# Options:
# "previous_consecutive" -> for test batch k, use k-9, ..., k-1
# "random_history"       -> for each test batch, randomly select 9 other batches
BATCH_SELECTION_MODE = "previous_consecutive"

# Optional manual test batches.
# If None, the script randomly selects test batches.
# Example:
# MANUAL_TEST_BATCHES = [19, 120, 250, 400, 650, 900]
MANUAL_TEST_BATCHES = None

# Optional fully manual experiments.
# This overrides MANUAL_TEST_BATCHES and random selection.
# Example:
# MANUAL_EXPERIMENTS = [
#     {"test_batch": 19, "history_batches": [10, 11, 12, 13, 14, 15, 16, 17, 18]},
#     {"test_batch": 55, "history_batches": [46, 47, 48, 49, 50, 51, 52, 53, 54]},
# ]
MANUAL_EXPERIMENTS = None

ANCHOR_GRID = np.array(
    [0, 10, 20, 30, 50, 80, 120, 180, 260, 360, 500, 700, 900, 1100],
    dtype=float,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Helper for finding CSV
# ============================================================

def resolve_data_path(data_path):
    if os.path.exists(data_path):
        return data_path

    candidates = sorted(glob.glob("transition_batches_long*.csv"))
    if len(candidates) > 0:
        print(f"DATA_PATH='{data_path}' not found.")
        print(f"Using detected CSV instead: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find '{data_path}' or any file matching "
        "'transition_batches_long*.csv' in the current folder."
    )


# ============================================================
# Original GP/interpolation logic
# ============================================================

def feature_cols_for(output_name):
    return [
        "anchor_index",
        "u1_frac_A",
        "u1_Tc_scaled",
        "delta_frac_A",
        "delta_Tc_scaled",
        f"{output_name}_0",
    ]


def interpolate_batch_to_anchors(batch_df, anchor_grid):
    batch_df = batch_df.sort_values("time").copy()
    t_raw = batch_df["time"].to_numpy(dtype=float)
    t_final = float(t_raw[-1])

    anchors = [float(a) for a in anchor_grid if a <= t_final + 1e-12]
    if len(anchors) == 0 or not np.isclose(anchors[-1], t_final):
        anchors.append(t_final)
    anchors = np.array(sorted(set(np.round(anchors, 12))), dtype=float)

    row0 = batch_df.iloc[0]
    records = []

    for anchor_index, t in enumerate(anchors):
        rec = {
            "batch_id": int(row0["batch_id"]),
            "anchor_index": float(anchor_index),
            "time": float(t),
            "u0_frac_A": float(row0["u0_frac_A"]),
            "u0_Tc_scaled": float(row0["u0_Tc_scaled"]),
            "u1_frac_A": float(row0["u1_frac_A"]),
            "u1_Tc_scaled": float(row0["u1_Tc_scaled"]),
            "delta_frac_A": float(row0["u1_frac_A"] - row0["u0_frac_A"]),
            "delta_Tc_scaled": float(row0["u1_Tc_scaled"] - row0["u0_Tc_scaled"]),
            "is_final_anchor": bool(np.isclose(t, t_final)),
        }

        for col in ["Ca", "Cb", "T", "Cc", "Cd"]:
            rec[col] = float(np.interp(t, t_raw, batch_df[col].to_numpy(dtype=float)))

        records.append(rec)

    return pd.DataFrame(records)


def bounded_lbfgs_optimizer(obj_func, initial_theta, bounds):
    result = minimize(
        obj_func,
        initial_theta,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 80, "ftol": 1e-6, "gtol": 1e-5},
    )
    return result.x, result.fun


class ScaledScalarRBFGP:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.gp = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        Xs = self.x_scaler.fit_transform(X)
        ys = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        n_features = Xs.shape[1]

        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(length_scale=np.ones(n_features), length_scale_bounds=(0.2, 20.0))
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2))
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=False,
            optimizer=bounded_lbfgs_optimizer,
            n_restarts_optimizer=0,
            random_state=self.random_state,
        )

        self.gp.fit(Xs, ys)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = self.x_scaler.transform(X)
        mean_scaled, std_scaled = self.gp.predict(Xs, return_std=True)
        mean = self.y_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        std = std_scaled * float(self.y_scaler.scale_[0])
        return mean, std


# ============================================================
# Multi-combination logic
# ============================================================

def anchor_count_from_final_time(t_final, anchor_grid):
    anchors = [float(a) for a in anchor_grid if a <= float(t_final) + 1e-12]
    if len(anchors) == 0 or not np.isclose(anchors[-1], float(t_final)):
        anchors.append(float(t_final))
    anchors = np.array(sorted(set(np.round(anchors, 12))), dtype=float)
    return len(anchors)


def make_history_variants(history_pool, history_sizes):
    """
    Preserves the current nested logic.

    Example:
    history_pool = [10, 11, 12, 13, 14, 15, 16, 17, 18]

    Then:
    3 previous runs -> [10, 11, 12]
    6 previous runs -> [10, 11, 12, 13, 14, 15]
    9 previous runs -> [10, 11, 12, 13, 14, 15, 16, 17, 18]
    """
    history_pool = list(map(int, history_pool))

    history_variants = {}
    for n in history_sizes:
        if len(history_pool) < n:
            raise ValueError(f"Need at least {n} history batches, got {len(history_pool)}.")

        history_variants[f"{n} previous runs"] = history_pool[:n]

    return history_variants


def choose_experiment_configs(df, anchor_grid):
    rng = np.random.default_rng(RANDOM_SEED)

    max_history_size = int(max(HISTORY_SIZES))
    all_batches = sorted(map(int, df["batch_id"].unique()))
    all_batch_set = set(all_batches)

    final_time_by_batch = df.groupby("batch_id")["time"].max().to_dict()

    valid_test_batches = []
    for b in all_batches:
        n_anchors = anchor_count_from_final_time(final_time_by_batch[b], anchor_grid)
        if n_anchors > OBSERVED_ANCHOR_COUNT:
            valid_test_batches.append(b)

    valid_test_set = set(valid_test_batches)

    if MANUAL_EXPERIMENTS is not None:
        experiments = []

        for run_index, item in enumerate(MANUAL_EXPERIMENTS, start=1):
            test_batch = int(item["test_batch"])
            history_pool = list(map(int, item["history_batches"]))

            if test_batch not in valid_test_set:
                raise ValueError(
                    f"Manual test batch {test_batch} does not have enough anchors "
                    f"for OBSERVED_ANCHOR_COUNT={OBSERVED_ANCHOR_COUNT}."
                )

            experiments.append({
                "run_index": run_index,
                "test_batch": test_batch,
                "history_variants": make_history_variants(history_pool, HISTORY_SIZES),
            })

        return experiments

    if MANUAL_TEST_BATCHES is not None:
        chosen_test_batches = list(map(int, MANUAL_TEST_BATCHES))
    else:
        if BATCH_SELECTION_MODE == "previous_consecutive":
            candidate_test_batches = []

            for b in valid_test_batches:
                previous_batches = list(range(b - max_history_size, b))
                if all(p in all_batch_set for p in previous_batches):
                    candidate_test_batches.append(b)

        elif BATCH_SELECTION_MODE == "random_history":
            candidate_test_batches = valid_test_batches

        else:
            raise ValueError(
                "BATCH_SELECTION_MODE must be either "
                "'previous_consecutive' or 'random_history'."
            )

        if len(candidate_test_batches) < N_TEST_COMBINATIONS:
            raise ValueError(
                f"Requested {N_TEST_COMBINATIONS} test combinations, "
                f"but only {len(candidate_test_batches)} are available."
            )

        chosen_test_batches = list(
            rng.choice(candidate_test_batches, size=N_TEST_COMBINATIONS, replace=False)
        )
        chosen_test_batches = list(map(int, chosen_test_batches))

    experiments = []

    for run_index, test_batch in enumerate(chosen_test_batches, start=1):
        if test_batch not in valid_test_set:
            raise ValueError(
                f"Test batch {test_batch} does not have enough anchors "
                f"for OBSERVED_ANCHOR_COUNT={OBSERVED_ANCHOR_COUNT}."
            )

        if BATCH_SELECTION_MODE == "previous_consecutive":
            history_pool = list(range(test_batch - max(HISTORY_SIZES), test_batch))

            missing = [b for b in history_pool if b not in all_batch_set]
            if missing:
                raise ValueError(
                    f"Test batch {test_batch} cannot use previous "
                    f"{max(HISTORY_SIZES)} consecutive runs. Missing: {missing}"
                )

        elif BATCH_SELECTION_MODE == "random_history":
            possible_history_batches = [b for b in all_batches if b != test_batch]

            history_pool = list(
                rng.choice(
                    possible_history_batches,
                    size=max(HISTORY_SIZES),
                    replace=False,
                )
            )

            history_pool = sorted(map(int, history_pool))

        experiments.append({
            "run_index": run_index,
            "test_batch": int(test_batch),
            "history_variants": make_history_variants(history_pool, HISTORY_SIZES),
        })

    return experiments


def run_one_completion_experiment(anchor_df, run_index, test_batch, history_variants):
    test_df_all = (
        anchor_df[anchor_df["batch_id"] == test_batch]
        .sort_values("anchor_index")
        .copy()
        .reset_index(drop=True)
    )

    if OBSERVED_ANCHOR_COUNT >= len(test_df_all):
        raise ValueError(
            f"Test batch {test_batch} has only {len(test_df_all)} anchors; "
            f"OBSERVED_ANCHOR_COUNT={OBSERVED_ANCHOR_COUNT} leaves no future anchors."
        )

    observed_anchor_indices = test_df_all["anchor_index"].to_numpy()[:OBSERVED_ANCHOR_COUNT]
    observed_times = test_df_all["time"].to_numpy()[:OBSERVED_ANCHOR_COUNT]

    last_observed_time = float(observed_times[-1])
    final_time = float(test_df_all["time"].iloc[-1])
    final_anchor_index = float(test_df_all["anchor_index"].iloc[-1])
    observed_anchor_index_set = set(observed_anchor_indices)

    pred_records = []
    final_records = []
    model_records = []

    for history_label, history_batches in history_variants.items():
        history_df = anchor_df[anchor_df["batch_id"].isin(history_batches)].copy()
        current_observed_df = test_df_all.iloc[:OBSERVED_ANCHOR_COUNT].copy()

        for ycol in OUTPUTS:
            feat_cols = feature_cols_for(ycol)

            train_df = pd.concat([history_df, current_observed_df], ignore_index=True)

            model = ScaledScalarRBFGP(random_state=42).fit(
                train_df[feat_cols].to_numpy(dtype=float),
                train_df[ycol].to_numpy(dtype=float),
            )

            y_mean, y_std = model.predict(test_df_all[feat_cols].to_numpy(dtype=float))

            model_records.append({
                "run_index": run_index,
                "test_batch": test_batch,
                "history_label": history_label,
                "history_size": len(history_batches),
                "history_batches": ", ".join(map(str, history_batches)),
                "output": ycol,
                "n_train": len(train_df),
                "feature_cols": ", ".join(feat_cols),
                "learned_kernel": str(model.gp.kernel_),
            })

            for i, row in test_df_all.iterrows():
                anchor_index = float(row["anchor_index"])
                t = float(row["time"])
                true_val = float(row[ycol])
                pred_val = float(y_mean[i])
                std_val = float(y_std[i])
                observed = anchor_index in observed_anchor_index_set

                pred_records.append({
                    "run_index": run_index,
                    "test_batch": test_batch,
                    "history_label": history_label,
                    "history_size": len(history_batches),
                    "history_batches": ", ".join(map(str, history_batches)),
                    "output": ycol,
                    "anchor_index": anchor_index,
                    "time": t,
                    "observed_in_current_batch": observed,
                    "true": true_val,
                    "predicted": pred_val,
                    "std": std_val,
                    "abs_error": abs(pred_val - true_val),
                })

            final_true = float(test_df_all.iloc[-1][ycol])
            final_pred = float(y_mean[-1])
            final_std = float(y_std[-1])

            final_records.append({
                "run_index": run_index,
                "test_batch": test_batch,
                "history_label": history_label,
                "history_size": len(history_batches),
                "history_batches": ", ".join(map(str, history_batches)),
                "output": ycol,
                "observed_anchor_count": OBSERVED_ANCHOR_COUNT,
                "observed_anchor_indices": ", ".join(f"{a:g}" for a in observed_anchor_indices),
                "observed_times_s": ", ".join(f"{t:g}" for t in observed_times),
                "last_observed_time_s": last_observed_time,
                "final_anchor_index": final_anchor_index,
                "final_time_s": final_time,
                "predicted_final": final_pred,
                "real_final": final_true,
                "posterior_std_final": final_std,
                "abs_error_final": abs(final_pred - final_true),
            })

    pred_df = pd.DataFrame(pred_records)
    final_df = pd.DataFrame(final_records)
    model_df = pd.DataFrame(model_records)

    history_order = list(history_variants.keys())

    fig, axes = plt.subplots(
        nrows=len(history_order),
        ncols=len(OUTPUTS),
        figsize=(7 * len(OUTPUTS), 3.4 * len(history_order)),
        sharex=False,
    )

    axes = np.asarray(axes).reshape(len(history_order), len(OUTPUTS))

    fig.suptitle(
        f"Run {run_index}: batch {test_batch} completion with "
        f"{OBSERVED_ANCHOR_COUNT} observed anchors\n"
        "RBF scalar GPs; GP input uses anchor index; plot uses actual time",
        fontsize=14,
    )

    for r, history_label in enumerate(history_order):
        for c, ycol in enumerate(OUTPUTS):
            ax = axes[r, c]

            sub = pred_df[
                (pred_df["history_label"] == history_label)
                & (pred_df["output"] == ycol)
            ].sort_values("anchor_index")

            obs = sub[sub["observed_in_current_batch"]]
            fut = sub[~sub["observed_in_current_batch"]]

            ax.plot(
                sub["time"],
                sub["true"],
                marker="o",
                label=f"real batch-{test_batch} anchors",
            )

            ax.plot(
                fut["time"],
                fut["predicted"],
                marker="x",
                linestyle="--",
                label="GP inferred future anchors",
            )

            ax.scatter(
                obs["time"],
                obs["true"],
                marker="s",
                s=45,
                label=f"observed batch-{test_batch} anchors",
            )

            ax.axvline(last_observed_time, linestyle=":", linewidth=1)

            final_err = final_df[
                (final_df["history_label"] == history_label)
                & (final_df["output"] == ycol)
            ]["abs_error_final"].iloc[0]

            hist_size = final_df[
                (final_df["history_label"] == history_label)
                & (final_df["output"] == ycol)
            ]["history_size"].iloc[0]

            err_str = f"{final_err:.2f}" if ycol == "T" else f"{final_err:.4f}"

            ax.set_title(f"{ycol}; {hist_size} previous runs; final error = {err_str}")
            ax.set_xlabel("time / s")
            ax.set_ylabel(ycol)
            ax.grid(True, alpha=0.3)

            if r == 0 and c == 0:
                ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    run_tag = f"run_{run_index:03d}_testbatch{test_batch}_{OBSERVED_ANCHOR_COUNT}anchors"

    plot_path = os.path.join(
        OUTPUT_DIR,
        f"gp_rbf_anchorindex_plot_time_{run_tag}_cb_t.png",
    )

    pred_csv = os.path.join(
        OUTPUT_DIR,
        f"gp_rbf_anchorindex_plot_time_{run_tag}_predictions_cb_t.csv",
    )

    final_csv = os.path.join(
        OUTPUT_DIR,
        f"gp_rbf_anchorindex_plot_time_{run_tag}_final_table_cb_t.csv",
    )

    model_csv = os.path.join(
        OUTPUT_DIR,
        f"gp_rbf_anchorindex_plot_time_{run_tag}_model_table_cb_t.csv",
    )

    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()

    pred_df.to_csv(pred_csv, index=False)
    final_df.to_csv(final_csv, index=False)
    model_df.to_csv(model_csv, index=False)

    return pred_df, final_df, model_df, plot_path


# ============================================================
# Main script
# ============================================================

def main():
    data_path = resolve_data_path(DATA_PATH)

    df = pd.read_csv(data_path)

    required_cols = [
        "batch_id",
        "time",
        "u0_frac_A",
        "u0_Tc_scaled",
        "u1_frac_A",
        "u1_Tc_scaled",
        "Ca",
        "Cb",
        "T",
        "Cc",
        "Cd",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"The data file is missing these columns: {missing_cols}")

    print("Loaded data:")
    print(f"- path: {data_path}")
    print(f"- rows: {len(df)}")
    print(f"- batches: {df['batch_id'].nunique()}")
    print(f"- batch ID range: {df['batch_id'].min()} to {df['batch_id'].max()}")

    experiments = choose_experiment_configs(df, ANCHOR_GRID)

    print("\nSelected experiments:")
    for exp in experiments:
        print(f"\nRun {exp['run_index']}: test batch {exp['test_batch']}")
        for label, batches in exp["history_variants"].items():
            print(f"  {label}: {batches}")

    needed_batches = set()

    for exp in experiments:
        needed_batches.add(exp["test_batch"])
        for batches in exp["history_variants"].values():
            needed_batches.update(batches)

    needed_batches = sorted(needed_batches)

    df_needed = df[df["batch_id"].isin(needed_batches)].copy()

    anchor_df = pd.concat(
        [
            interpolate_batch_to_anchors(g, ANCHOR_GRID)
            for _, g in df_needed.groupby("batch_id")
        ],
        ignore_index=True,
    )

    for ycol in OUTPUTS:
        y0_map = (
            anchor_df
            .sort_values("anchor_index")
            .groupby("batch_id")[ycol]
            .first()
            .to_dict()
        )
        anchor_df[f"{ycol}_0"] = anchor_df["batch_id"].map(y0_map)

    all_pred_dfs = []
    all_final_dfs = []
    all_model_dfs = []
    plot_records = []

    for exp in experiments:
        print(
            f"\nRunning experiment {exp['run_index']} "
            f"with test batch {exp['test_batch']}..."
        )

        pred_df, final_df, model_df, plot_path = run_one_completion_experiment(
            anchor_df=anchor_df,
            run_index=exp["run_index"],
            test_batch=exp["test_batch"],
            history_variants=exp["history_variants"],
        )

        all_pred_dfs.append(pred_df)
        all_final_dfs.append(final_df)
        all_model_dfs.append(model_df)

        plot_records.append({
            "run_index": exp["run_index"],
            "test_batch": exp["test_batch"],
            "plot_path": plot_path,
        })

    all_pred_df = pd.concat(all_pred_dfs, ignore_index=True)
    all_final_df = pd.concat(all_final_dfs, ignore_index=True)
    all_model_df = pd.concat(all_model_dfs, ignore_index=True)
    plot_df = pd.DataFrame(plot_records)

    summary_rows = []

    for (history_label, history_size, output), g in all_final_df.groupby(
        ["history_label", "history_size", "output"]
    ):
        errors = g["abs_error_final"].to_numpy(dtype=float)
        posterior_stds = g["posterior_std_final"].to_numpy(dtype=float)

        summary_rows.append({
            "history_label": history_label,
            "history_size": int(history_size),
            "output": output,
            "n_tests": int(len(g)),
            "mean_abs_error_final": float(np.mean(errors)),
            "median_abs_error_final": float(np.median(errors)),
            "std_abs_error_final": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
            "rmse_final": float(np.sqrt(np.mean(errors ** 2))),
            "min_abs_error_final": float(np.min(errors)),
            "max_abs_error_final": float(np.max(errors)),
            "mean_posterior_std_final": float(np.mean(posterior_stds)),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["output", "history_size"])

    comparison_pivot_df = all_final_df.pivot_table(
        index=["run_index", "test_batch"],
        columns=["output", "history_size"],
        values="abs_error_final",
        aggfunc="first",
    )

    comparison_pivot_df.columns = [
        f"{output}_abs_err_hist{history_size}"
        for output, history_size in comparison_pivot_df.columns
    ]

    comparison_pivot_df = comparison_pivot_df.reset_index()

    all_pred_csv = os.path.join(
        OUTPUT_DIR,
        "all_test_combinations_predictions_cb_t.csv",
    )

    all_final_csv = os.path.join(
        OUTPUT_DIR,
        "all_test_combinations_final_table_cb_t.csv",
    )

    all_model_csv = os.path.join(
        OUTPUT_DIR,
        "all_test_combinations_model_table_cb_t.csv",
    )

    summary_csv = os.path.join(
        OUTPUT_DIR,
        "comparison_summary_cb_t.csv",
    )

    pivot_csv = os.path.join(
        OUTPUT_DIR,
        "comparison_pivot_abs_errors_cb_t.csv",
    )

    plot_csv = os.path.join(
        OUTPUT_DIR,
        "plot_paths_cb_t.csv",
    )

    all_pred_df.to_csv(all_pred_csv, index=False)
    all_final_df.to_csv(all_final_csv, index=False)
    all_model_df.to_csv(all_model_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    comparison_pivot_df.to_csv(pivot_csv, index=False)
    plot_df.to_csv(plot_csv, index=False)

    print("\n============================================================")
    print("Final-anchor results for every run")
    print("============================================================")
    print(all_final_df)

    print("\n============================================================")
    print("Comparison summary across test combinations")
    print("============================================================")
    print(summary_df)

    print("\n============================================================")
    print("Per-run absolute-error comparison table")
    print("============================================================")
    print(comparison_pivot_df)

    print("\n============================================================")
    print("Saved outputs")
    print("============================================================")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"- {all_pred_csv}")
    print(f"- {all_final_csv}")
    print(f"- {all_model_csv}")
    print(f"- {summary_csv}")
    print(f"- {pivot_csv}")
    print(f"- {plot_csv}")

    print("\nSaved plot files:")
    for p in plot_df["plot_path"]:
        print(f"- {p}")


if __name__ == "__main__":
    main()
