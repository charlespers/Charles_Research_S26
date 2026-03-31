"""
Fabrication-parameter sweep for the Iris reservoir benchmark.

Sweeps a grid of physical fab knobs (coupling motif, coupling strength,
pump-current bounds, laser spacing, etc.) over one or more Iris workload
scenarios, collecting the classification accuracy at each grid point.
Results are written to CSV for downstream analysis by ``fab_meta_model``.

Two sampling strategies are supported:

factorial
    Full Cartesian product of the discrete knob lists.  Use for small grids
    or exhaustive search.

lhc (Latin Hypercube)
    Space-filling quasi-random samples over the continuous knob ranges, with
    categorical knobs (motif, noise) assigned by round-robin.  Use for
    larger continuous search spaces.

Typical usage
-------------
Call ``run_fab_sweep_long`` directly, or drive it from the pipeline entry
point ``pipeline_iris/run_pipeline.py``.
"""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from benchmark_iris import run_iris_model7_benchmark
from fab_design import assign_categoricals_lhc_rows, latin_hypercube_fab_samples
from output_io import ensure_run_dir, new_run_id, save_json


# ============================================================
# Workload descriptors
# ============================================================

def _iris_bench_sizes(test_size: float, n_total: int = 150) -> tuple[int, int]:
    """Return (n_train, n_test) for a given test fraction of the Iris dataset."""
    n_test  = int(round(n_total * test_size))
    n_train = n_total - n_test
    return n_train, n_test


def scenario_bench_descriptor(sc: dict[str, Any]) -> dict[str, Any]:
    """Extract numeric workload features used as inputs to the meta-model.

    Returns a dict with keys:
        task_encoding        (0 = iris)
        bench_n_train
        bench_n_test
        bench_washout_macros (0 for Iris)
        bench_hold_ns        (simulation window length in ns)
    """
    ts     = float(sc.get("test_size", 0.25))
    n_tr, n_te = _iris_bench_sizes(ts)
    t_span = sc.get("t_span_ns", (0.0, 20.0))
    return {
        "task_encoding":        0,
        "bench_n_train":        n_tr,
        "bench_n_test":         n_te,
        "bench_washout_macros": 0,
        "bench_hold_ns":        float(t_span[1] - t_span[0]),
    }


def estimate_latency_euler_steps(sc: dict[str, Any], fab: dict[str, Any]) -> int:
    """Estimate simulation cost as the number of Euler steps for one train split."""
    dt_ns  = float(fab.get("dt_ns", 5e-4)) or 5e-4
    t_span = sc.get("t_span_ns", (0.0, 20.0))
    steps_one = max(1, int(round((float(t_span[1]) - float(t_span[0])) / dt_ns)))
    n_tr, _ = _iris_bench_sizes(float(sc.get("test_size", 0.25)))
    return steps_one * n_tr


def log10_sim_budget(sc: dict[str, Any], fab: dict[str, Any]) -> float:
    return float(math.log10(max(1, estimate_latency_euler_steps(sc, fab))))


# ============================================================
# Benchmark dispatcher
# ============================================================

def _run_benchmark(
    sc: dict[str, Any],
    fab: dict[str, Any],
    run_seed: int,
    use_gpu: bool = False,
) -> tuple[float, bool, str]:
    """Run one Iris benchmark for a (scenario, fab) pair.

    Returns
    -------
    metric        : float   classification accuracy on test split
    maximize      : bool    True (accuracy is maximised)
    metric_name   : str     "accuracy_test"
    """
    r = run_iris_model7_benchmark(
        motif       = fab["motif"],
        default_k   = float(fab["default_k"]),
        i_min       = float(fab["i_min"]),
        i_max       = float(fab["i_max"]),
        spacing_m   = float(fab["spacing_m"]),
        noise_on    = bool(fab["noise_on"]),
        I_th_A      = float(fab.get("I_th_A", 17.35e-3)),
        lambda0_m   = float(fab.get("lambda0_m", 850e-9)),
        dt_ns       = float(fab.get("dt_ns", 5e-4)),
        store_every = int(fab.get("store_every", 400)),
        base_seed   = sc.get("base_seed", run_seed),
        test_size   = float(sc.get("test_size", 0.25)),
        random_state_split = int(sc.get("random_state_split", 0)),
        t_span_ns   = tuple(sc.get("t_span_ns", (0.0, 20.0))),
        washout_ns  = float(sc.get("washout_ns", 10.0)),
        save_outputs = False,
        use_gpu     = use_gpu,
    )
    return float(r["accuracy_test"]), True, "accuracy_test"


# ============================================================
# Fab grid iterators
# ============================================================

def _iter_fab_factorial(fab_grid: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield every combination in the Cartesian product of the fab grid lists."""
    defaults = {
        "I_th_As":      [17.35e-3],
        "lambda0_ms":   [850e-9],
        "dt_ns_s":      [5e-4],
        "store_everys": [400],
        "augment_inputs": [True],
    }
    keys = (
        "motifs", "default_ks", "i_mins", "i_maxs", "spacing_ms",
        "noise_flags", "I_th_As", "lambda0_ms", "dt_ns_s",
        "store_everys", "augment_inputs",
    )
    lists = [fab_grid.get(k, defaults[k]) if k in defaults else fab_grid[k] for k in keys]
    for tup in product(*lists):
        yield {
            "motif":        tup[0],
            "default_k":    tup[1],
            "i_min":        tup[2],
            "i_max":        tup[3],
            "spacing_m":    tup[4],
            "noise_on":     tup[5],
            "I_th_A":       tup[6],
            "lambda0_m":    tup[7],
            "dt_ns":        tup[8],
            "store_every":  tup[9],
            "augment_input": tup[10],
        }


def _bounds_from_list_or(
    fab_grid: dict[str, Any],
    list_key: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    if fab_grid.get(list_key):
        v = fab_grid[list_key]
        return (float(min(v)), float(max(v)))
    return default


def _iter_fab_lhc(
    n_samples: int,
    fab_grid: dict[str, Any],
    seed: int,
) -> Iterator[dict[str, Any]]:
    """Yield ``n_samples`` Latin-Hypercube fab configurations."""
    motifs      = fab_grid.get("motifs",      ["auxiliary"])
    noise_flags = fab_grid.get("noise_flags", [True])
    store_everys = fab_grid.get("store_everys", [400])

    dk  = fab_grid.get("default_k_bounds") or _bounds_from_list_or(fab_grid, "default_ks",  (10.0, 20.0))
    imn = fab_grid.get("i_min_bounds")     or _bounds_from_list_or(fab_grid, "i_mins",       (0.45, 0.58))
    imx = fab_grid.get("i_max_bounds")     or _bounds_from_list_or(fab_grid, "i_maxs",       (1.35, 1.58))
    sp  = fab_grid.get("spacing_m_bounds") or _bounds_from_list_or(fab_grid, "spacing_ms",   (45e-6, 55e-6))
    Ith = fab_grid.get("I_th_A_bounds")    or (_bounds_from_list_or(fab_grid, "I_th_As", (16e-3, 18e-3))
                                                if fab_grid.get("I_th_As") else None)
    lam = fab_grid.get("lambda0_m_bounds") or (_bounds_from_list_or(fab_grid, "lambda0_ms", (840e-9, 860e-9))
                                                if fab_grid.get("lambda0_ms") else None)
    dtn = fab_grid.get("dt_ns_bounds")     or (_bounds_from_list_or(fab_grid, "dt_ns_s", (4e-4, 6e-4))
                                                if fab_grid.get("dt_ns_s") else None)

    base_rows = latin_hypercube_fab_samples(
        n_samples,
        default_k_bounds=dk, i_min_bounds=imn, i_max_bounds=imx,
        spacing_m_bounds=sp, I_th_A_bounds=Ith, lambda0_m_bounds=lam,
        dt_ns_bounds=dtn, seed=seed,
    )
    rows = assign_categoricals_lhc_rows(
        base_rows, motifs=motifs, noise_flags=noise_flags,
        store_every_values=store_everys,
    )
    for r in rows:
        r["augment_input"] = fab_grid.get("augment_input_default", True)
        yield r


# ============================================================
# Aggregation and Pareto helpers
# ============================================================

def aggregate_sweep_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse replicate rows to one row per (scenario, fab) with mean/std metric."""
    group_cols = [
        "scenario_tag", "task", "task_encoding",
        "bench_n_train", "bench_n_test", "bench_washout_macros", "bench_hold_ns",
        "log10_sim_budget", "motif", "default_k", "i_min", "i_max",
        "spacing_m", "spacing_um", "noise_on", "I_th_A", "lambda0_m",
        "dt_ns", "store_every", "augment_input", "metric_name", "maximize_metric",
        "latency_euler_steps",
    ]
    id_cols = [c for c in group_cols if c in runs_df.columns]
    agg = (
        runs_df.groupby(id_cols, dropna=False, as_index=False)
        .agg(metric_mean=("metric", "mean"), metric_std=("metric", "std"),
             n_replicates=("metric", "count"))
    )
    agg["metric"]     = agg["metric_mean"]
    agg["metric_std"] = agg["metric_std"].fillna(0.0)
    return agg


def pareto_mask_maximize_minimize(
    metrics: np.ndarray,
    latencies: np.ndarray,
    maximize_metric: bool,
) -> np.ndarray:
    """Return a boolean mask of Pareto-non-dominated points.

    Optimises jointly for quality (maximised or minimised) and simulation
    latency (always minimised).
    """
    q   = metrics if maximize_metric else -metrics
    lat = latencies
    n   = len(metrics)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if q[j] >= q[i] and lat[j] <= lat[i] and (q[j] > q[i] or lat[j] < lat[i]):
                keep[i] = False
                break
    return keep


# ============================================================
# Main sweep runner
# ============================================================

def run_fab_sweep_long(
    scenarios: list[dict[str, Any]],
    *,
    fab_grid: dict[str, Any] | None = None,
    design: str = "factorial",
    n_lhc_samples: int = 24,
    n_replicates: int = 1,
    seed: int = 0,
    outputs_base: Path | None = None,
    use_gpu: bool = False,
) -> tuple[Path, Path]:
    """Run a fab sweep over all (scenario, fab-config) pairs and save results.

    Parameters
    ----------
    scenarios : list of scenario dicts (see ``scenarios_iris_*``).
    fab_grid  : dict of fab knob lists/ranges.  Defaults to quick grid.
    design    : "factorial" or "lhc".
    n_lhc_samples : number of LHC samples (only used when design="lhc").
    n_replicates  : number of independent replicates per (scenario, fab) pair.
    seed      : base random seed.
    outputs_base  : directory for sweep output files.
    use_gpu   : route Iris simulation through the JAX GPU backend.

    Returns
    -------
    (sweep_runs_csv, sweep_long_csv) : paths to the raw and aggregated CSVs.
    """
    base    = outputs_base or Path(__file__).resolve().parent
    run_id  = new_run_id("fab_long")
    out_dir = ensure_run_dir(base, "sweeps", "fab_long", run_id)
    fab_grid = fab_grid or default_fab_grid_quick()

    rows_raw: list[dict[str, Any]] = []
    rep_offset = 0

    for sc in scenarios:
        desc = scenario_bench_descriptor(sc)
        tag  = sc["scenario_tag"]

        if design == "factorial":
            fab_iter = _iter_fab_factorial(fab_grid)
        elif design == "lhc":
            fab_iter = _iter_fab_lhc(n_lhc_samples, fab_grid, seed + rep_offset)
        else:
            raise ValueError("design must be 'factorial' or 'lhc'")

        for fab in fab_iter:
            lat = estimate_latency_euler_steps(sc, fab)
            lsb = log10_sim_budget(sc, fab)
            for rep in range(n_replicates):
                run_seed = int(seed + rep_offset + rep * 9973)
                metric, maximize, mname = _run_benchmark(sc, fab, run_seed, use_gpu=use_gpu)
                rows_raw.append({
                    "scenario_tag":       tag,
                    "task":               "iris",
                    **desc,
                    "log10_sim_budget":   lsb,
                    "latency_euler_steps": lat,
                    "motif":              fab["motif"],
                    "default_k":          fab["default_k"],
                    "i_min":              fab["i_min"],
                    "i_max":              fab["i_max"],
                    "spacing_m":          fab["spacing_m"],
                    "spacing_um":         fab["spacing_m"] * 1e6,
                    "noise_on":           fab["noise_on"],
                    "I_th_A":             fab["I_th_A"],
                    "lambda0_m":          fab["lambda0_m"],
                    "dt_ns":              fab["dt_ns"],
                    "store_every":        fab["store_every"],
                    "augment_input":      fab.get("augment_input", True),
                    "metric":             metric,
                    "metric_name":        mname,
                    "maximize_metric":    maximize,
                    "replicate_id":       rep,
                    "run_seed":           run_seed,
                })
            rep_offset += 17

    df_runs    = pd.DataFrame(rows_raw)
    runs_path  = out_dir / "sweep_runs.csv"
    df_runs.to_csv(runs_path, index=False)

    if n_replicates > 1:
        df_long = aggregate_sweep_runs(df_runs)
    else:
        df_long = df_runs.drop(columns=["replicate_id", "run_seed"], errors="ignore")
        df_long["metric_mean"]  = df_long["metric"]
        df_long["metric_std"]   = 0.0
        df_long["n_replicates"] = 1

    long_path = out_dir / "sweep_long.csv"
    df_long.to_csv(long_path, index=False)

    save_json(out_dir / "sweep_config.json", {
        "run_id":        run_id,
        "design":        design,
        "n_lhc_samples": n_lhc_samples,
        "n_replicates":  n_replicates,
        "n_run_rows":    len(df_runs),
        "n_long_rows":   len(df_long),
        "scenarios":     scenarios,
        "fab_grid":      fab_grid,
        "seed":          seed,
    })
    print(f"Wrote {runs_path} ({len(df_runs)} runs), {long_path} ({len(df_long)} rows)")
    return runs_path, long_path


# ============================================================
# Scenario and grid presets
# ============================================================

def scenarios_iris_quick() -> list[dict[str, Any]]:
    """Single default Iris scenario (25 % test split)."""
    return [{"scenario_tag": "iris_default", "task": "iris", "test_size": 0.25, "base_seed": 30}]


def scenarios_iris_stretch() -> list[dict[str, Any]]:
    """Three Iris scenarios covering different train/test splits."""
    return [
        {"scenario_tag": "iris_small_test",  "task": "iris", "test_size": 0.15, "base_seed": 30},
        {"scenario_tag": "iris_default",     "task": "iris", "test_size": 0.25, "base_seed": 30},
        {"scenario_tag": "iris_large_test",  "task": "iris", "test_size": 0.40, "base_seed": 30},
    ]


def scenarios_for_task(*, stretch: bool = False) -> list[dict[str, Any]]:
    """Return the appropriate Iris scenario list.

    Parameters
    ----------
    stretch : bool
        If True, use three workload shapes; otherwise just the default split.
    """
    return scenarios_iris_stretch() if stretch else scenarios_iris_quick()


def default_fab_grid_quick() -> dict[str, Any]:
    """Small fab grid: 2 motifs × 2 coupling strengths = 4 configs."""
    return {
        "motifs":       ["auxiliary", "chain"],
        "default_ks":   [12.0, 18.0],
        "i_mins":       [0.5],
        "i_maxs":       [1.5],
        "spacing_ms":   [50e-6],
        "noise_flags":  [True],
        "I_th_As":      [17.35e-3],
        "lambda0_ms":   [850e-9],
        "dt_ns_s":      [5e-4],
        "store_everys": [400],
        "augment_inputs": [True],
    }


def default_fab_grid_standard() -> dict[str, Any]:
    """Full fab grid for thorough sweeps (~7 000 configs, slow)."""
    return {
        "motifs":       ["auxiliary", "chain", "relay"],
        "default_ks":   [10.0, 15.0, 20.0],
        "i_mins":       [0.45, 0.55],
        "i_maxs":       [1.35, 1.55],
        "spacing_ms":   [45e-6, 50e-6, 55e-6],
        "noise_flags":  [True, False],
        "I_th_As":      [16.5e-3, 17.35e-3, 18.2e-3],
        "lambda0_ms":   [840e-9, 850e-9, 860e-9],
        "dt_ns_s":      [4e-4, 5e-4],
        "store_everys": [320, 400],
        "augment_inputs": [True, False],
    }
