from pathlib import Path
import argparse, sys, json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import odeint

STATE_LABELS = ["Ca", "Cb", "T", "Cc", "Cd"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

def sampled_times(t_end, sample_interval):
    """Return sample times clipped to the transition end, including t_end."""
    if t_end <= 0:
        return np.array([0.0])

    ts = np.arange(0.0, t_end, sample_interval)
    if len(ts) == 0 or not np.isclose(ts[-1], t_end):
        ts = np.append(ts, t_end)
    return ts

def build_grid(frac_bounds, tc_bounds, n=25):
    fa = np.linspace(*frac_bounds, n)
    tc = np.linspace(*tc_bounds, n)
    pts = []
    for i, a in enumerate(fa):
        for j, t in enumerate(tc):
            pts.append({"id": i*n+j, "frac_A": float(a), "Tc_scaled": float(t), "ij": (i, j)})
    return pts

def pick_path(grid, n_batches, seed=1, local_prob=0.5):
    rng = np.random.default_rng(seed)
    visits = np.zeros(len(grid))
    path = [grid[rng.integers(len(grid))]]
    visits[path[0]["id"]] += 1

    for _ in range(n_batches):
        cur = path[-1]
        u = np.array([cur["frac_A"], cur["Tc_scaled"]])
        coords = np.array([[p["frac_A"], p["Tc_scaled"]] for p in grid])
        d = np.linalg.norm(coords - u, axis=1)
        d[cur["id"]] = np.inf

        finite = d[np.isfinite(d)]
        if rng.random() < local_prob:
            cand = np.where(np.isfinite(d) & (d <= np.quantile(finite, 0.20)))[0]
        else:
            cand = np.where(np.isfinite(d) & (d >= np.quantile(finite, 0.70)))[0]

        w = 1.0 / (1.0 + visits[cand])
        w = w / w.sum()
        nxt = grid[int(rng.choice(cand, p=w))]
        visits[nxt["id"]] += 1
        path.append(nxt)

    return path

def state_cols(prefix, x):
    return {f"{prefix}_{name}": float(val) for name, val in zip(STATE_LABELS, x)}

def input_cols(prefix, p):
    return {
        f"{prefix}_frac_A": p["frac_A"],
        f"{prefix}_Tc_scaled": p["Tc_scaled"],
    }

def settled_index(states, x_final, tol):
    scale = np.maximum(np.abs(x_final), 1.0)
    err = np.max(np.abs((states - x_final) / scale), axis=1)
    for k in range(len(err)):
        if np.all(err[k:] <= tol):
            return k
    return len(err) - 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=CODE_DIR)
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--preset", default="trust", choices=["trust", "ifac_grid"])
    ap.add_argument("--n-batches", type=int, default=1000)
    ap.add_argument("--grid-size", type=int, default=25)
    ap.add_argument("--frac-min", type=float, default=0.02)
    ap.add_argument("--frac-max", type=float, default=0.98)
    ap.add_argument("--tc-min", type=float, default=0.05)
    ap.add_argument("--tc-max", type=float, default=0.95)
    ap.add_argument("--sample-interval", type=float, default=10.0)
    ap.add_argument("--transition-t-end", type=float, default=3000.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    sys.path.insert(0, str(args.model_dir.resolve()))
    import cstr_dynamic_model as model

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    grid = build_grid((args.frac_min, args.frac_max), (args.tc_min, args.tc_max), args.grid_size)
    path = pick_path(grid, args.n_batches, seed=args.seed)

    cfg = model.get_config(args.preset)
    prices = model.DEFAULT_PRICES
    steady_cache = {}

    def steady(p):
        if p["id"] not in steady_cache:
            r = model.simulate_cstr(
                frac_A=p["frac_A"],
                Tc_scaled=p["Tc_scaled"],
                preset=args.preset,
                t_end=args.transition_t_end,
                n_points=900,
                settling_tolerance=args.tol,
            )
            steady_cache[p["id"]] = np.asarray(r.steady_state, dtype=float)
        return steady_cache[p["id"]]

    long_rows, boundary_rows, target_rows, summary_rows = [], [], [], []
    transient_rows_raw = []
    max_samples = 0

    for b in range(args.n_batches):
        p0, p1 = path[b], path[b+1]
        x0, x1 = steady(p0), steady(p1)

        t = np.arange(0, args.transition_t_end + args.dt, args.dt)
        states = odeint(
            model.cstr_rhs,
            x0,
            t,
            args=(p1["frac_A"], p1["Tc_scaled"], cfg, prices, model.DEFAULT_T_RANGE),
        )

        k_end = settled_index(states, x1, args.tol)
        t_end = t[k_end]
        ts = sampled_times(t_end, args.sample_interval)
        xs = np.column_stack([np.interp(ts, t, states[:, i]) for i in range(5)])

        max_samples = max(max_samples, len(ts))
        transient_rows_raw.append((b, ts, xs))

        boundary_rows.append({
            "batch_id": b,
            "start_point_id": p0["id"],
            "end_point_id": p1["id"],
            **input_cols("u0", p0),
            **input_cols("u1", p1),
            **state_cols("x0", x0),
            **state_cols("x1", x1),
        })

        target_rows.append({"batch_id": b, **state_cols("y", x1)})

        summary_rows.append({
            "batch_id": b,
            "settling_time": float(t_end),
            "n_samples": len(ts),
            "input_distance": float(np.linalg.norm(
                np.array([p1["frac_A"], p1["Tc_scaled"]]) -
                np.array([p0["frac_A"], p0["Tc_scaled"]])
            )),
            "state_distance": float(np.linalg.norm(x1 - x0)),
        })

        for s, time in enumerate(ts):
            row = {
                "batch_id": b,
                "sample_id": s,
                "time": float(time),
                **input_cols("u0", p0),
                **input_cols("u1", p1),
            }
            row.update({name: float(xs[s, i]) for i, name in enumerate(STATE_LABELS)})
            long_rows.append(row)

    transient_rows = []
    for b, ts, xs in transient_rows_raw:
        row = {"batch_id": b, "n_samples": len(ts)}
        for s in range(max_samples):
            row[f"tr_s{s:03d}_time"] = float(ts[s]) if s < len(ts) else np.nan
            for i, name in enumerate(STATE_LABELS):
                row[f"tr_s{s:03d}_{name}"] = float(xs[s, i]) if s < len(ts) else np.nan
        transient_rows.append(row)

    pd.DataFrame(long_rows).to_csv(out / "transition_batches_long.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(out / "boundary_features.csv", index=False)
    pd.DataFrame(transient_rows).to_csv(out / "transient_state_features.csv", index=False)
    pd.DataFrame(target_rows).to_csv(out / "steady_state_targets.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "transition_summary.csv", index=False)
    pd.DataFrame(path).to_csv(out / "visited_operating_path.csv", index=False)
    pd.DataFrame(grid).to_csv(out / "input_grid_25x25.csv", index=False)

    u = np.array([[p["frac_A"], p["Tc_scaled"]] for p in path])
    plt.figure(figsize=(7, 6))
    plt.plot(u[:, 0], u[:, 1], lw=1, alpha=0.7)
    plt.scatter(u[:, 0], u[:, 1], c=np.arange(len(u)), cmap="viridis", s=25)
    plt.xlabel("frac_A")
    plt.ylabel("Tc_scaled")
    plt.title("Visited input trajectory")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "diagnostic_input_trajectory.png", dpi=220)
    plt.close()

    summary = pd.DataFrame(summary_rows)
    plt.figure(figsize=(7, 5))
    plt.hist(summary["input_distance"], bins=25, edgecolor="white")
    plt.xlabel("input transition distance")
    plt.ylabel("count")
    plt.title("Close and far transition coverage")
    plt.tight_layout()
    plt.savefig(out / "diagnostic_transition_distances.png", dpi=220)
    plt.close()

    meta = {
        "u0_*": "initial input before transition",
        "u1_*": "final input after transition",
        "x0_*": "initial steady-state states",
        "x1_*": "final steady-state states",
        "tr_sNNN_*": "sampled transient state at sample index NNN",
        "y_*": "target final steady-state states",
        "state_order": STATE_LABELS,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved dataset in: {out.resolve()}")
    print(f"Batches: {args.n_batches}")
    print(f"Max samples per batch: {max_samples}")

if __name__ == "__main__":
    main()
