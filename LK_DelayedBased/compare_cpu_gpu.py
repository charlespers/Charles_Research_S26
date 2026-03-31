"""
Direct CPU vs GPU timing comparison for the Model 7 Iris reservoir.
Run on a GPU node:
  python compare_cpu_gpu.py
"""
import time
import numpy as np

print("Loading JAX...", flush=True)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
print(f"JAX backend : {jax.default_backend()}")
print(f"Devices     : {jax.devices()}", flush=True)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from model7_reservoir import (
    build_reservoir_dataset, minmax_fit, minmax_transform,
)
from model7_reservoir_gpu import build_gpu_sim_params, build_reservoir_dataset_gpu

# ── shared data ───────────────────────────────────────────────────────────────
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data.astype(float), iris.target.astype(int),
    test_size=0.25, random_state=0, stratify=iris.target,
)
xmin, span = minmax_fit(X_train)
X_train01 = minmax_transform(X_train, xmin, span)
X_test01  = minmax_transform(X_test,  xmin, span)

shared_kwargs = dict(
    motif="auxiliary", spacing_m=50e-6, lambda0_m=850e-9, n_air=1.0,
    I_th_A=17.35e-3, i_min=0.5, i_max=1.5,
    dt_ns=5e-4, t_span_ns=(0.0, 20.0), washout_ns=10.0,
    store_every=400, noise_on=False, default_k=15.0,
)

# ── CPU run ───────────────────────────────────────────────────────────────────
print("\n[CPU] Running...", flush=True)
t0 = time.perf_counter()
F_tr_cpu, _ = build_reservoir_dataset(X_train01, y_train, base_seed=30, **shared_kwargs)
F_te_cpu, _ = build_reservoir_dataset(X_test01,  y_test,  base_seed=10030, **shared_kwargs)
t_cpu = time.perf_counter() - t0
print(f"[CPU] Done in {t_cpu:.1f}s", flush=True)

# ── GPU setup (one-time, CPU-side) ────────────────────────────────────────────
print("\n[GPU] Building params...", flush=True)
gp = build_gpu_sim_params(**shared_kwargs)

# ── GPU first call (includes XLA compilation) ─────────────────────────────────
print("[GPU] First call (includes JIT compile)...", flush=True)
t0 = time.perf_counter()
F_tr_gpu1, _ = build_reservoir_dataset_gpu(X_train01, y_train, gpu_params=gp, base_seed=30)
F_te_gpu1, _ = build_reservoir_dataset_gpu(X_test01,  y_test,  gpu_params=gp, base_seed=10030)
t_gpu1 = time.perf_counter() - t0
print(f"[GPU] First call done in {t_gpu1:.1f}s", flush=True)

# ── GPU second call (cached kernel) ──────────────────────────────────────────
print("[GPU] Second call (cached)...", flush=True)
t0 = time.perf_counter()
F_tr_gpu2, _ = build_reservoir_dataset_gpu(X_train01, y_train, gpu_params=gp, base_seed=30)
F_te_gpu2, _ = build_reservoir_dataset_gpu(X_test01,  y_test,  gpu_params=gp, base_seed=10030)
t_gpu2 = time.perf_counter() - t0
print(f"[GPU] Second call done in {t_gpu2:.1f}s", flush=True)

# ── Correctness check (noise_off => should be identical) ─────────────────────
max_err_tr = float(np.max(np.abs(F_tr_gpu2 - F_tr_cpu)))
max_err_te = float(np.max(np.abs(F_te_gpu2 - F_te_cpu)))
match = max_err_tr < 1e-6 and max_err_te < 1e-6

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print(f"  CPU time              : {t_cpu:.2f}s")
print(f"  GPU time (1st call)  : {t_gpu1:.2f}s  (JIT compile included)")
print(f"  GPU time (2nd call)  : {t_gpu2:.2f}s  (cached)")
print(f"  Speedup (cached)     : {t_cpu/t_gpu2:.1f}x")
print(f"  Feature max abs diff : train={max_err_tr:.2e}  test={max_err_te:.2e}")
print(f"  Correctness          : {'PASS (< 1e-6)' if match else 'FAIL'}")
print("="*50)
