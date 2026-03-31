"""
Shared CLI entry point for the Iris fab-sweep pipeline.

Flow
----
1. Parse command-line arguments.
2. Build a fab grid (quick / standard / LHC).
3. Call ``run_fab_sweep_long`` to evaluate every (scenario, fab) combination
   and write sweep_runs.csv + sweep_long.csv.
4. Optionally aggregate sweep results and train a fab recommender model.

This module is imported by ``pipeline_iris/run_pipeline.py`` which passes
the pipeline root directory so that outputs land under
``pipeline_iris/outputs/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fab_meta_model import aggregate_optimal, train_recommender
from fab_sweep import (
    default_fab_grid_quick,
    default_fab_grid_standard,
    run_fab_sweep_long,
    scenarios_for_task,
)
from output_io import ensure_run_dir, new_run_id


def run_cli(pipe_root: Path) -> None:
    """Parse CLI args and run the Iris pipeline.

    Parameters
    ----------
    pipe_root : Path
        Root directory of the calling pipeline folder.  Sweep outputs and
        recommender artefacts are written under ``pipe_root/outputs/``.
    """
    ap = argparse.ArgumentParser(
        description="Iris fab sweep + recommender training "
                    f"— outputs under {pipe_root}/outputs/"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Use a 4-config fab grid and a single Iris scenario (fast).",
    )
    ap.add_argument(
        "--stretch-scenarios",
        action="store_true",
        help="Use three Iris scenarios (15%%, 25%%, 40%% test splits) instead of one.",
    )
    ap.add_argument(
        "--standard-grid",
        action="store_true",
        help="Use the full fab grid (~7 000 configs) instead of the quick grid.",
    )
    ap.add_argument(
        "--lhc",
        action="store_true",
        help="Use Latin-Hypercube sampling instead of a Cartesian-product grid.",
    )
    ap.add_argument("--n-lhc",       type=int,   default=24,  dest="n_lhc",
                    help="Number of LHC samples (only used with --lhc).")
    ap.add_argument("--n-replicates", type=int,   default=1,
                    help="Independent replicates per (scenario, fab) pair.")
    ap.add_argument("--seed",         type=int,   default=0,
                    help="Base random seed.")
    ap.add_argument(
        "--sweep-only",
        action="store_true",
        help="Stop after the sweep CSV; skip aggregate + recommender training.",
    )
    ap.add_argument("--robust-lambda", type=float, default=0.0, dest="robust_lambda",
                    help="Penalty for metric variance across replicates when "
                         "selecting the optimal fab config.")
    ap.add_argument("--surrogate",  action="store_true",
                    help="Use surrogate-model optimum instead of empirical best.")
    ap.add_argument("--pareto",     action="store_true",
                    help="Multi-objective Pareto optimisation (quality vs latency).")
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="Route Iris simulation through the JAX GPU backend (requires JAX).",
    )
    args = ap.parse_args()

    # ── Build fab grid ────────────────────────────────────────────────────────
    if args.quick:
        grid = {
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
    else:
        grid = default_fab_grid_standard() if args.standard_grid else default_fab_grid_quick()

    scenarios = scenarios_for_task(stretch=args.stretch_scenarios)

    # ── Run sweep ─────────────────────────────────────────────────────────────
    _, long_csv = run_fab_sweep_long(
        scenarios,
        fab_grid=grid,
        design="lhc" if args.lhc else "factorial",
        n_lhc_samples=args.n_lhc,
        n_replicates=args.n_replicates,
        seed=args.seed,
        outputs_base=pipe_root.resolve(),
        use_gpu=args.gpu,
    )
    print(f"Sweep CSV: {long_csv}")

    if args.sweep_only:
        return

    # ── Aggregate + train recommender ─────────────────────────────────────────
    opt = aggregate_optimal(
        long_csv,
        None,
        robust_penalty_lambda=args.robust_lambda,
        use_surrogate_optimum=args.surrogate,
        pareto_multi_objective=args.pareto,
    )
    train_dir = ensure_run_dir(
        pipe_root.resolve(), "analysis", "fab_recommender", new_run_id("train")
    )
    train_recommender(opt, train_dir)
    print(f"Recommender artefacts: {train_dir}")
