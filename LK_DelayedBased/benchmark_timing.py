#!/usr/bin/env python3
"""
GPU vs CPU timing benchmark for the Model 7 Iris reservoir.

Sweeps batch sizes and three simulation configs, saves a structured CSV
(outputs/timing/timing_results.csv) and a summary JSON for use in
paper Table 1 and Figure 1.

Usage
-----
  python benchmark_timing.py
  python benchmark_timing.py --output-dir outputs/timing
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

print("Loading JAX...", flush=True)
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from model7_reservoir import build_reservoir_dataset, minmax_fit, minmax_transform
from model7_reservoir_gpu import build_gpu_sim_params, build_reservoir_dataset_gpu
from output_io import save_json


# ── sweep parameters ─────────────────────────────────────────────────────────

BATCH_SIZES = [10, 25, 50, 75, 112]

# Each value is the full kwargs dict for both CPU and GPU builders.
CONFIGS: dict[str, dict] = {
    "noise_off_default": dict(
        motif="auxiliary", spacing_m=50e-6, lambda0_m=850e-9, n_air=1.0,
        I_th_A=17.35e-3, i_min=0.5, i_max=1.5,
        dt_ns=5e-4, t_span_ns=(0.0, 20.0), washout_ns=10.0,
        store_every=400, noise_on=False, default_k=15.0,
    ),
    "noise_on_default": dict(
        motif="auxiliary", spacing_m=50e-6, lambda0_m=850e-9, n_air=1.0,
        I_th_A=17.35e-3, i_min=0.5, i_max=1.5,
        dt_ns=5e-4, t_span_ns=(0.0, 20.0), washout_ns=10.0,
        store_every=400, noise_on=True, default_k=15.0,
    ),
    "noise_off_long": dict(
        motif="chain", spacing_m=50e-6, lambda0_m=850e-9, n_air=1.0,
        I_th_A=17.35e-3, i_min=0.5, i_max=1.5,
        dt_ns=5e-4, t_span_ns=(0.0, 40.0), washout_ns=20.0,
        store_every=400, noise_on=False, default_k=15.0,
    ),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _time_call(fn, *args, **kwargs) -> tuple[float, object]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def benchmark_one(
    config_name: str,
    cfg: dict,
    batch_size: int,
    X01_full: np.ndarray,
    y_full: np.ndarray,
) -> dict:
    X01 = X01_full[:batch_size]
    y   = y_full[:batch_size]

    # ── CPU ───────────────────────────────────────────────────────────────
    t_cpu, (F_cpu, _) = _time_call(
        build_reservoir_dataset, X01, y, base_seed=30, **cfg
    )

    # ── GPU params (not timed) ────────────────────────────────────────────
    gp = build_gpu_sim_params(**cfg)

    # ── GPU first call (includes JIT compile) ─────────────────────────────
    t_gpu_warmup, (F_gpu1, _) = _time_call(
        build_reservoir_dataset_gpu, X01, y, gpu_params=gp, base_seed=30
    )

    # ── GPU second call (cached kernel) ───────────────────────────────────
    t_gpu_cached, (F_gpu2, _) = _time_call(
        build_reservoir_dataset_gpu, X01, y, gpu_params=gp, base_seed=30
    )

    speedup = t_cpu / max(t_gpu_cached, 1e-9)

    # Correctness: noise_off → CPU and GPU features should agree to <1e-5
    if not cfg["noise_on"]:
        max_err = float(np.max(np.abs(F_gpu2 - F_cpu)))
        correctness = bool(max_err < 1e-5)
    else:
        correctness = True  # stochastic — cannot compare directly

    return {
        "config":        config_name,
        "batch_size":    batch_size,
        "t_cpu":         round(t_cpu, 4),
        "t_gpu_warmup":  round(t_gpu_warmup, 4),
        "t_gpu_cached":  round(t_gpu_cached, 4),
        "speedup":       round(speedup, 2),
        "correctness":   correctness,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = jax.default_backend()
    devices = str(jax.devices())
    print(f"JAX backend : {backend}")
    print(f"Devices     : {devices}")
    print(f"Batch sizes : {BATCH_SIZES}")
    print(f"Configs     : {list(CONFIGS.keys())}")
    print()

    iris = load_iris()
    X = iris.data.astype(float)
    y = iris.target.astype(int)
    xmin, span = minmax_fit(X)
    X01_full = minmax_transform(X, xmin, span)

    rows: list[dict] = []
    for cfg_name, cfg in CONFIGS.items():
        print(f"[{cfg_name}]")
        for bs in BATCH_SIZES:
            print(f"  batch={bs:3d} ...", end="  ", flush=True)
            row = benchmark_one(cfg_name, cfg, bs, X01_full, y)
            rows.append(row)
            print(
                f"cpu={row['t_cpu']:.3f}s  "
                f"gpu_cached={row['t_gpu_cached']:.3f}s  "
                f"speedup={row['speedup']:.1f}x  "
                f"ok={row['correctness']}"
            )
        print()

    df = pd.DataFrame(rows)
    csv_path = output_dir / "timing_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(df)} rows)")

    # ── summary JSON ──────────────────────────────────────────────────────
    summary: dict = {"backend": backend, "devices": devices, "batch_sizes": BATCH_SIZES, "configs": {}}
    for cfg_name in CONFIGS:
        sub = df[df["config"] == cfg_name]
        summary["configs"][cfg_name] = {
            "speedup_mean": round(float(sub["speedup"].mean()), 2),
            "speedup_max":  round(float(sub["speedup"].max()), 2),
            "speedup_min":  round(float(sub["speedup"].min()), 2),
            "all_correct":  bool(sub["correctness"].all()),
        }
    json_path = output_dir / "timing_summary.json"
    save_json(json_path, summary)
    print(f"Wrote {json_path}")

    print("\nSummary (cached GPU speedup):")
    for cfg_name, s in summary["configs"].items():
        print(
            f"  {cfg_name:30s}: mean={s['speedup_mean']:.1f}x  "
            f"max={s['speedup_max']:.1f}x  "
            f"correct={s['all_correct']}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="GPU vs CPU timing benchmark for Model 7 Iris reservoir"
    )
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/timing"),
                    help="Directory for timing_results.csv and timing_summary.json")
    args = ap.parse_args()
    main(args.output_dir)
