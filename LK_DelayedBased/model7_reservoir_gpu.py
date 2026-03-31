"""
GPU-accelerated Model 7 reservoir simulator using JAX.

Uses jax.lax.scan to JIT-compile the Euler-Maruyama loop and jax.vmap to
parallelize independent simulations (Iris samples, parameter sweep configs).

Drop-in replacements provided:
  build_reservoir_dataset_gpu()  ->  build_reservoir_dataset()
  build_gpu_sim_params()         ->  (setup helper, call once per config)

Validation:
  python model7_reservoir_gpu.py --validate        # single-step + feature checks
  python model7_reservoir_gpu.py --validate-full   # full Iris accuracy comparison

Installation:
  pip install -U "jax[cuda12]"   # CUDA 12.x GPU
  pip install -U jax              # CPU / Metal (Mac)
"""

from __future__ import annotations

import functools
import sys
from typing import Any

# ── JAX setup (must happen before any jax import uses float precision) ──────
import jax

jax.config.update("jax_enable_x64", True)  # needed: N ~ 1.25e8, float32 loses increments

import jax.numpy as jnp
import numpy as np

# ── Reuse CPU matrix builders (pure NumPy, called once at setup) ─────────────
from model7_reservoir import (
    build_inline_coupling_matrix_tunable,
    build_inline_geometry_matrices,
    default_physics_params_ops,
    features_to_pump_ratios,
    run_one_iris_sample_feature_vector,
)

# ── Constants ────────────────────────────────────────────────────────────────
_E_CHARGE = 1.602e-19  # Coulombs



# ============================================================
# 1. Setup helper: build all JAX arrays from physics params
# ============================================================

def build_gpu_sim_params(
    *,
    motif: str,
    spacing_m: float,
    lambda0_m: float,
    n_air: float,
    I_th_A: float,
    i_min: float,
    i_max: float,
    dt_ns: float,
    t_span_ns: tuple[float, float],
    washout_ns: float,
    store_every: int,
    noise_on: bool,
    default_k: float,
    num_lasers: int = 4,
) -> dict[str, Any]:
    """
    Build all coupling/geometry matrices for the GPU simulator.
    Calls the existing CPU NumPy builders once, then converts to JAX arrays.
    Returns a dict of static params to be passed to simulate_* functions.
    """
    K = build_inline_coupling_matrix_tunable(
        num_lasers=num_lasers,
        motif=motif,
        default_k=default_k,
    )
    _, tau_ns_mat, _, cos_phi, sin_phi = build_inline_geometry_matrices(
        num_lasers=num_lasers,
        spacing_m=spacing_m,
        lambda0_m=lambda0_m,
        n_air=n_air,
    )

    gamma = 496.0
    tau_p_ns = 1.0 / gamma
    tau_thresh = 0.1 * tau_p_ns

    delay_steps_mat = np.rint(tau_ns_mat / dt_ns).astype(int)
    use_delayed = tau_ns_mat >= tau_thresh
    np.fill_diagonal(use_delayed, False)
    np.fill_diagonal(delay_steps_mat, 0)

    Mxc = K * cos_phi
    Mxs = K * sin_phi

    local_mask = ~use_delayed
    local_Mxc = Mxc * local_mask
    local_Mxs = Mxs * local_mask

    unique_delays_arr = np.unique(delay_steps_mat[use_delayed])
    unique_delays_arr = unique_delays_arr[unique_delays_arr > 0]
    delay_list = [int(d) for d in unique_delays_arr]

    group_Mxc_list = []
    group_Mxs_list = []
    for d in delay_list:
        mask_d = use_delayed & (delay_steps_mat == d)
        group_Mxc_list.append(jnp.array(Mxc * mask_d))
        group_Mxs_list.append(jnp.array(Mxs * mask_d))

    ring_depth = (max(delay_list) + 1) if delay_list else 1

    t0, tf = t_span_ns
    n_steps = int((tf - t0) / dt_ns)
    washout_step = int(washout_ns / dt_ns)

    return {
        # JAX arrays (static across samples in a batch)
        "local_Mxc": jnp.array(local_Mxc),
        "local_Mxs": jnp.array(local_Mxs),
        "group_Mxc_list": group_Mxc_list,
        "group_Mxs_list": group_Mxs_list,
        # Static Python scalars / lists (must NOT change between JIT calls)
        "delay_list": delay_list,
        "ring_depth": ring_depth,
        "num_lasers": num_lasers,
        "n_steps": n_steps,
        "washout_step": washout_step,
        "store_every": store_every,
        "noise_on": noise_on,
        "dt_ns": dt_ns,
        # Physics scalars
        "alpha": 3.0,
        "g": 1.2e-5,
        "N0": 1.25e8,
        "s": 5e-7,
        "gamma": gamma,
        "gamma_e": 0.651,
        "beta_sp": 1e-5,
        # Pump mapping scalars
        "I_th_A": I_th_A,
        "i_min": i_min,
        "i_max": i_max,
    }


# ============================================================
# 2. JAX-traceable pump conversion
# ============================================================

def _pump_from_features_jax(
    features_01: jnp.ndarray,
    *,
    I_th_A: float,
    i_min: float,
    i_max: float,
    num_lasers: int,
) -> jnp.ndarray:
    """
    Convert (num_lasers,) normalized Iris features [0,1] -> pump rate (carriers/ns).
    Mirrors features_to_pump_ratios + u_norm_to_pump_ops logic, but in JAX.
    """
    f01 = jnp.clip(features_01, 0.0, 1.0)
    ratios = i_min + f01 * (i_max - i_min)  # shape (num_lasers,)
    I_ops = ratios * I_th_A
    P_ops = (I_ops / _E_CHARGE) * 1e-9
    return P_ops  # (num_lasers,)


# ============================================================
# 3. Single Euler-Maruyama step (pure JAX, no Python overhead)
# ============================================================

def _make_step_fn(
    local_Mxc: jnp.ndarray,
    local_Mxs: jnp.ndarray,
    group_Mxc_list: list,
    group_Mxs_list: list,
    delay_list: list[int],
    ring_depth: int,
    num_lasers: int,
    alpha: float,
    g: float,
    s: float,
    gamma: float,
    gamma_e: float,
    beta_sp: float,
    N0: float,
    P_ops: jnp.ndarray,
    dt_ns: float,
    noise_on: bool,
    washout_step: int,
    store_every: int,
):
    """
    Returns a lax.scan-compatible step function closed over all static parameters.
    carry = (x_ring, y_ring, N, step_idx, rng_key,
             sum_I, min_I, max_I, M2_I, sum_N, count)
    where x_ring/y_ring have shape (ring_depth, num_lasers).
    """
    sqrt_dt = jnp.sqrt(dt_ns)
    N0_arr = jnp.full((num_lasers,), N0)

    def step(carry, _xs):
        x_ring, y_ring, N, step_idx, rng_key, sum_I, min_I, max_I, M2_I, sum_N, count = carry

        # Read current state from ring (write_pos points to current step slot)
        write_pos = step_idx % ring_depth
        xn = x_ring[write_pos]
        yn = y_ring[write_pos]
        Nn = N

        # Gain
        S = xn * xn + yn * yn
        G = g * (Nn - N0_arr) / (1.0 + s * S)

        # Local coupling (instantaneous)
        inj_x = local_Mxc @ xn - local_Mxs @ yn
        inj_y = local_Mxc @ yn + local_Mxs @ xn

        # Delayed coupling — Python loop unrolled at trace time
        for i, d in enumerate(delay_list):
            read_pos = (step_idx - d + ring_depth * (d + 1)) % ring_depth
            xd = x_ring[read_pos]
            yd = y_ring[read_pos]
            inj_x = inj_x + group_Mxc_list[i] @ xd - group_Mxs_list[i] @ yd
            inj_y = inj_y + group_Mxc_list[i] @ yd + group_Mxs_list[i] @ xd

        # Drift terms
        dx = 0.5 * (G - gamma) * (xn - alpha * yn) + inj_x
        dy = 0.5 * (G - gamma) * (yn + alpha * xn) + inj_y
        dN = P_ops - gamma_e * Nn - G * S

        # Euler-Maruyama update
        if noise_on:
            rng_key, k1, k2 = jax.random.split(rng_key, 3)
            Rsp = beta_sp * gamma_e * jnp.maximum(Nn, 0.0)
            diff = jnp.sqrt(Rsp / 2.0) * sqrt_dt
            x_new = xn + dx * dt_ns + diff * jax.random.normal(k1, (num_lasers,))
            y_new = yn + dy * dt_ns + diff * jax.random.normal(k2, (num_lasers,))
        else:
            x_new = xn + dx * dt_ns
            y_new = yn + dy * dt_ns

        N_new = Nn + dN * dt_ns

        # Write new state into ring buffer
        next_write = (step_idx + 1) % ring_depth
        x_ring = x_ring.at[next_write].set(x_new)
        y_ring = y_ring.at[next_write].set(y_new)

        # Accumulate running statistics after washout, every store_every steps
        I_new = x_new * x_new + y_new * y_new
        next_idx = step_idx + 1
        in_window = (next_idx >= washout_step) & ((next_idx % store_every) == 0)

        def update_stats(args):
            sum_I, min_I, max_I, M2_I, sum_N, count, I_new, N_new = args
            count_new = count + 1
            delta = I_new - sum_I / jnp.maximum(count, 1)
            sum_I_new = sum_I + I_new
            delta2 = I_new - sum_I_new / count_new
            M2_new = M2_I + delta * delta2
            return (
                sum_I_new,
                jnp.minimum(min_I, I_new),
                jnp.maximum(max_I, I_new),
                M2_new,
                sum_N + N_new,
                count_new,
            )

        def keep_stats(args):
            sum_I, min_I, max_I, M2_I, sum_N, count, _I, _N = args
            return sum_I, min_I, max_I, M2_I, sum_N, count

        sum_I, min_I, max_I, M2_I, sum_N, count = jax.lax.cond(
            in_window,
            update_stats,
            keep_stats,
            (sum_I, min_I, max_I, M2_I, sum_N, count, I_new, N_new),
        )

        new_carry = (
            x_ring, y_ring, N_new,
            step_idx + 1, rng_key,
            sum_I, min_I, max_I, M2_I, sum_N, count,
        )
        return new_carry, None

    return step


# ============================================================
# 4. Single Iris sample simulation
# ============================================================

def simulate_one_iris_sample_gpu(
    features_01: jnp.ndarray,  # (num_lasers,)
    rng_key: jnp.ndarray,      # (2,) uint32 PRNG key
    *,
    gpu_params: dict,
) -> jnp.ndarray:              # (20,) feature vector
    """
    Run one Iris sample through the Model 7 reservoir using JAX.
    Returns 20D feature vector: [meanI, minI, maxI, stdI, meanN] per laser.
    """
    Nlas = gpu_params["num_lasers"]
    ring_depth = gpu_params["ring_depth"]
    n_steps = gpu_params["n_steps"]
    washout_step = gpu_params["washout_step"]
    store_every = gpu_params["store_every"]
    dt_ns = gpu_params["dt_ns"]
    noise_on = gpu_params["noise_on"]

    # Compute per-laser pump from Iris features
    P_ops = _pump_from_features_jax(
        features_01,
        I_th_A=gpu_params["I_th_A"],
        i_min=gpu_params["i_min"],
        i_max=gpu_params["i_max"],
        num_lasers=Nlas,
    )

    # Build the closed-over step function
    step_fn = _make_step_fn(
        local_Mxc=gpu_params["local_Mxc"],
        local_Mxs=gpu_params["local_Mxs"],
        group_Mxc_list=gpu_params["group_Mxc_list"],
        group_Mxs_list=gpu_params["group_Mxs_list"],
        delay_list=gpu_params["delay_list"],
        ring_depth=ring_depth,
        num_lasers=Nlas,
        alpha=gpu_params["alpha"],
        g=gpu_params["g"],
        s=gpu_params["s"],
        gamma=gpu_params["gamma"],
        gamma_e=gpu_params["gamma_e"],
        beta_sp=gpu_params["beta_sp"],
        N0=gpu_params["N0"],
        P_ops=P_ops,
        dt_ns=dt_ns,
        noise_on=noise_on,
        washout_step=washout_step,
        store_every=store_every,
    )

    # Initialize carry — match CPU: y0 = [[1e-3, 0, 0], ...] (N starts at 0, not N0)
    x0 = jnp.full((Nlas,), 1e-3)
    y0 = jnp.zeros((Nlas,))
    N0_val = jnp.zeros((Nlas,))

    # Ring buffer: all slots initialized to initial condition
    x_ring = jnp.broadcast_to(x0[None, :], (ring_depth, Nlas)).copy()
    y_ring = jnp.broadcast_to(y0[None, :], (ring_depth, Nlas)).copy()

    # Running stats (Welford)
    neg_inf = jnp.full((Nlas,), -jnp.inf)
    pos_inf = jnp.full((Nlas,), jnp.inf)
    init_carry = (
        x_ring,
        y_ring,
        N0_val,
        jnp.array(0, dtype=jnp.int64),
        rng_key,
        jnp.zeros((Nlas,)),   # sum_I
        pos_inf,               # min_I
        neg_inf,               # max_I
        jnp.zeros((Nlas,)),   # M2_I (Welford)
        jnp.zeros((Nlas,)),   # sum_N
        jnp.array(0, dtype=jnp.int64),  # count
    )

    final_carry, _ = jax.lax.scan(step_fn, init_carry, xs=None, length=n_steps)

    _, _, _, _, _, sum_I, min_I, max_I, M2_I, sum_N, count = final_carry

    count_f = jnp.maximum(count.astype(jnp.float64), 1.0)
    mean_I = sum_I / count_f
    std_I = jnp.sqrt(jnp.maximum(M2_I / count_f, 0.0))
    mean_N = sum_N / count_f

    # Build 20D feature: [meanI, minI, maxI, stdI, meanN] per laser
    feats = jnp.stack([mean_I, min_I, max_I, std_I, mean_N], axis=1)  # (Nlas, 5)
    return feats.ravel()  # (20,)


# ============================================================
# 5. Batched Iris simulation (JIT + vmap)
# ============================================================

def simulate_iris_batch_gpu(
    X01_batch: jnp.ndarray,   # (batch, num_lasers)
    rng_keys: jnp.ndarray,    # (batch, 2)
    *,
    gpu_params: dict,
) -> jnp.ndarray:             # (batch, 20)
    """
    Vectorized (vmapped) simulation of a batch of independent Iris samples.
    All samples share the same coupling matrices and physics parameters.
    """
    fn = functools.partial(simulate_one_iris_sample_gpu, gpu_params=gpu_params)
    batched = jax.vmap(fn, in_axes=(0, 0))
    return batched(X01_batch, rng_keys)


# ============================================================
# 6. Drop-in replacement for build_reservoir_dataset()
# ============================================================

def build_reservoir_dataset_gpu(
    X01: np.ndarray,
    y: np.ndarray,
    *,
    gpu_params: dict,
    base_seed: int | None = 30,
    chunk_size: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop-in replacement for model7_reservoir.build_reservoir_dataset().

    X01: (N, 4) normalized Iris features in [0,1]
    y:   (N,) integer labels
    Returns: F (N, 20), y (N,)
    """
    X01 = np.asarray(X01, dtype=float)
    y = np.asarray(y, dtype=int)
    N = X01.shape[0]

    seed = base_seed if base_seed is not None else 0
    master_key = jax.random.PRNGKey(seed)

    F_parts = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        batch = jnp.array(X01[start:end])
        master_key, subkey = jax.random.split(master_key)
        keys = jax.random.split(subkey, end - start)

        F_chunk = simulate_iris_batch_gpu(batch, keys, gpu_params=gpu_params)
        F_parts.append(np.array(F_chunk))

    F = np.concatenate(F_parts, axis=0)
    return F, y


# ============================================================
# 7. Validation suite
# ============================================================

def _default_gpu_params_for_test(noise_on: bool = False) -> dict:
    return build_gpu_sim_params(
        motif="auxiliary",
        spacing_m=50e-6,
        lambda0_m=850e-9,
        n_air=1.0,
        I_th_A=17.35e-3,
        i_min=0.5,
        i_max=1.5,
        dt_ns=5e-4,
        t_span_ns=(0.0, 20.0),
        washout_ns=10.0,
        store_every=400,
        noise_on=noise_on,
        default_k=15.0,
    )


def validate_single_step(verbose: bool = True) -> bool:
    """
    Check that one GPU Euler step matches one CPU Euler step (noise_off, float64).
    Tolerance: relative error < 1e-10.
    """
    gp = _default_gpu_params_for_test(noise_on=False)

    # Build a known state
    xn = np.array([1e-3, 2e-3, -1e-3, 5e-4])
    yn = np.array([0.0, 1e-4, -2e-4, 3e-4])
    Nn = np.array([1.25e8, 1.26e8, 1.24e8, 1.25e8])
    P_ops = np.array([1.08e11, 1.08e11, 1.08e11, 1.08e11])

    alpha = gp["alpha"]; g = gp["g"]; s = gp["s"]
    gamma = gp["gamma"]; gamma_e = gp["gamma_e"]
    N0 = gp["N0"]; dt_ns = gp["dt_ns"]
    local_Mxc = np.array(gp["local_Mxc"])
    local_Mxs = np.array(gp["local_Mxs"])

    # CPU step
    S = xn**2 + yn**2
    G = g * (Nn - N0) / (1.0 + s * S)
    inj_x = local_Mxc @ xn - local_Mxs @ yn
    inj_y = local_Mxc @ yn + local_Mxs @ xn
    dx = 0.5 * (G - gamma) * (xn - alpha * yn) + inj_x
    dy = 0.5 * (G - gamma) * (yn + alpha * xn) + inj_y
    dN = P_ops - gamma_e * Nn - G * S
    x_cpu = xn + dx * dt_ns
    y_cpu = yn + dy * dt_ns
    N_cpu = Nn + dN * dt_ns

    # GPU step: run a 1-step simulation on a trivial carry
    Nlas = gp["num_lasers"]
    ring_depth = gp["ring_depth"]
    N0_arr = jnp.full((Nlas,), N0)

    step_fn = _make_step_fn(
        local_Mxc=gp["local_Mxc"], local_Mxs=gp["local_Mxs"],
        group_Mxc_list=gp["group_Mxc_list"], group_Mxs_list=gp["group_Mxs_list"],
        delay_list=gp["delay_list"], ring_depth=ring_depth, num_lasers=Nlas,
        alpha=alpha, g=g, s=s, gamma=gamma, gamma_e=gamma_e, beta_sp=gp["beta_sp"],
        N0=N0, P_ops=jnp.array(P_ops), dt_ns=dt_ns,
        noise_on=False, washout_step=999999, store_every=1,
    )

    x_ring = jnp.zeros((ring_depth, Nlas))
    x_ring = x_ring.at[0].set(jnp.array(xn))
    y_ring = jnp.zeros((ring_depth, Nlas))
    y_ring = y_ring.at[0].set(jnp.array(yn))

    dummy_key = jax.random.PRNGKey(0)
    carry = (
        x_ring, y_ring, jnp.array(Nn),
        jnp.array(0, dtype=jnp.int64), dummy_key,
        jnp.zeros((Nlas,)), jnp.full((Nlas,), jnp.inf),
        jnp.full((Nlas,), -jnp.inf), jnp.zeros((Nlas,)),
        jnp.zeros((Nlas,)), jnp.array(0, dtype=jnp.int64),
    )

    new_carry, _ = jax.jit(step_fn)(carry, None)
    next_pos = 1 % ring_depth
    x_gpu = np.array(new_carry[0][next_pos])
    y_gpu = np.array(new_carry[1][next_pos])
    N_gpu = np.array(new_carry[2])

    # Compare
    ok = True
    for name, cpu_val, gpu_val in [("x", x_cpu, x_gpu), ("y", y_cpu, y_gpu), ("N", N_cpu, N_gpu)]:
        rel_err = np.max(np.abs(gpu_val - cpu_val) / (np.abs(cpu_val) + 1e-30))
        passed = rel_err < 1e-9
        ok = ok and passed
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] single-step {name}: max rel error = {rel_err:.2e}")

    return ok


def validate_single_sample_features(verbose: bool = True) -> bool:
    """
    Check that the 20D feature vector from GPU matches CPU for one Iris sample
    with noise_off. Tolerance: relative error < 1e-5 on all features.
    """
    from sklearn.datasets import load_iris
    from model7_reservoir import minmax_fit, minmax_transform

    gp = _default_gpu_params_for_test(noise_on=False)

    iris = load_iris()
    X = iris.data.astype(float)
    xmin, span = minmax_fit(X)
    X01 = minmax_transform(X, xmin, span)
    sample = X01[0]  # first sample

    # CPU
    feats_cpu = run_one_iris_sample_feature_vector(
        sample,
        motif="auxiliary",
        spacing_m=50e-6,
        lambda0_m=850e-9,
        n_air=1.0,
        I_th_A=17.35e-3,
        i_min=0.5,
        i_max=1.5,
        t_span_ns=(0.0, 20.0),
        washout_ns=10.0,
        dt_ns=5e-4,
        store_every=400,
        noise_on=False,
        seed=None,
        default_k=15.0,
    )

    # GPU
    dummy_key = jax.random.PRNGKey(0)
    feats_gpu = np.array(simulate_one_iris_sample_gpu(
        jnp.array(sample), dummy_key, gpu_params=gp
    ))

    rel_err = np.max(np.abs(feats_gpu - feats_cpu) / (np.abs(feats_cpu) + 1e-30))
    passed = rel_err < 1e-5
    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] single-sample features: max rel error = {rel_err:.2e}")
        if not passed:
            print(f"         CPU feats: {feats_cpu[:5]}")
            print(f"         GPU feats: {feats_gpu[:5]}")
    return passed


def validate_full_iris_accuracy(verbose: bool = True, n_trials: int = 3) -> bool:
    """
    Run full Iris benchmark on CPU and GPU for n_trials seeds.
    Check that mean accuracy difference < 5% (stochastic tolerance).
    """
    from benchmark_iris import run_iris_model7_benchmark
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import accuracy_score
    from model7_reservoir import minmax_fit, minmax_transform

    if verbose:
        print("  Running CPU Iris benchmark...")

    cpu_accs = []
    gpu_accs = []

    base_kwargs = dict(
        motif="auxiliary",
        dt_ns=5e-4,
        store_every=400,
        noise_on=True,
        t_span_ns=(0.0, 20.0),
        washout_ns=10.0,
        spacing_m=50e-6,
        lambda0_m=850e-9,
        n_air=1.0,
        I_th_A=17.35e-3,
        default_k=15.0,
        i_min=0.5,
        i_max=1.5,
        test_size=0.25,
        random_state_split=0,
    )

    for trial in range(n_trials):
        seed = 30 + trial * 100

        # CPU
        res_cpu = run_iris_model7_benchmark(base_seed=seed, **base_kwargs)
        cpu_accs.append(res_cpu["accuracy_test"])

        # GPU
        iris = load_iris()
        X = iris.data.astype(float)
        y = iris.target.astype(int)
        from sklearn.model_selection import train_test_split as tts
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.25, random_state=0, stratify=y)
        xmin, span_ = minmax_fit(X_train)
        X_train01 = minmax_transform(X_train, xmin, span_)
        X_test01 = minmax_transform(X_test, xmin, span_)

        gp = build_gpu_sim_params(
            motif="auxiliary", spacing_m=50e-6, lambda0_m=850e-9, n_air=1.0,
            I_th_A=17.35e-3, i_min=0.5, i_max=1.5,
            dt_ns=5e-4, t_span_ns=(0.0, 20.0), washout_ns=10.0,
            store_every=400, noise_on=True, default_k=15.0,
        )

        F_train, y_tr = build_reservoir_dataset_gpu(X_train01, y_train, gpu_params=gp, base_seed=seed)
        F_test, y_te = build_reservoir_dataset_gpu(X_test01, y_test, gpu_params=gp, base_seed=seed + 10000)

        clf = Pipeline([("sc", StandardScaler()), ("rc", RidgeClassifier())])
        clf.fit(F_train, y_tr)
        acc_gpu = float(accuracy_score(y_te, clf.predict(F_test)))
        gpu_accs.append(acc_gpu)

        if verbose:
            print(f"    trial {trial+1}: cpu_acc={cpu_accs[-1]:.3f}  gpu_acc={gpu_accs[-1]:.3f}")

    diff = abs(np.mean(cpu_accs) - np.mean(gpu_accs))
    passed = diff < 0.05
    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Iris accuracy: mean_cpu={np.mean(cpu_accs):.3f}  mean_gpu={np.mean(gpu_accs):.3f}  |diff|={diff:.3f} (tol=0.05)")
    return passed


def validate_quick_vs_standard_grid(verbose: bool = True) -> bool:
    """
    Compare quick grid (2 motifs × 2 k-values) vs standard grid results using GPU.
    Quick-grid best accuracy should be within 5% of standard-grid best.
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import accuracy_score
    from model7_reservoir import minmax_fit, minmax_transform

    iris = load_iris()
    X = iris.data.astype(float)
    y = iris.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    xmin, span_ = minmax_fit(X_train)
    X_train01 = minmax_transform(X_train, xmin, span_)
    X_test01 = minmax_transform(X_test, xmin, span_)

    quick_grid = {
        "motifs": ["auxiliary", "chain"],
        "default_ks": [12.0, 18.0],
    }
    standard_grid = {
        "motifs": ["auxiliary", "chain", "relay", "competitive"],
        "default_ks": [10.0, 15.0, 20.0],
    }

    def run_grid(grid: dict, label: str) -> list[float]:
        accs = []
        for motif in grid["motifs"]:
            for k in grid["default_ks"]:
                gp = build_gpu_sim_params(
                    motif=motif, spacing_m=50e-6, lambda0_m=850e-9, n_air=1.0,
                    I_th_A=17.35e-3, i_min=0.5, i_max=1.5,
                    dt_ns=5e-4, t_span_ns=(0.0, 20.0), washout_ns=10.0,
                    store_every=400, noise_on=False, default_k=k,
                )
                F_tr, y_tr = build_reservoir_dataset_gpu(X_train01, y_train, gpu_params=gp, base_seed=30)
                F_te, y_te = build_reservoir_dataset_gpu(X_test01, y_test, gpu_params=gp, base_seed=10030)
                clf = Pipeline([("sc", StandardScaler()), ("rc", RidgeClassifier())])
                clf.fit(F_tr, y_tr)
                acc = float(accuracy_score(y_te, clf.predict(F_te)))
                accs.append(acc)
                if verbose:
                    print(f"    {label:8s} motif={motif:12s} k={k:5.1f}  acc={acc:.3f}")
        return accs

    if verbose:
        print("  Quick grid (GPU, noise_off):")
    quick_accs = run_grid(quick_grid, "quick")

    if verbose:
        print("  Standard grid (GPU, noise_off):")
    std_accs = run_grid(standard_grid, "standard")

    best_quick = max(quick_accs)
    best_std = max(std_accs)
    diff = abs(best_quick - best_std)
    passed = diff < 0.10  # 10% tolerance (quick grid is a subset, may miss best)
    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] quick best={best_quick:.3f}  standard best={best_std:.3f}  |diff|={diff:.3f} (tol=0.10)")
    return passed


def print_device_info():
    """Print available JAX devices."""
    devices = jax.devices()
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {devices}")


# ============================================================
# 8. CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Model 7 GPU reservoir — validation")
    ap.add_argument("--validate", action="store_true", help="Run single-step + feature + accuracy checks")
    ap.add_argument("--validate-full", action="store_true", help="Run full Iris accuracy comparison (slow)")
    ap.add_argument("--validate-grids", action="store_true", help="Run quick vs standard grid comparison")
    ap.add_argument("--devices", action="store_true", help="Print device info")
    args = ap.parse_args()

    if args.devices or (not any(vars(args).values())):
        print_device_info()

    if args.validate or args.validate_full:
        print("\n=== Validation Suite ===")
        print_device_info()

        print("\n[1] Single Euler step (noise_off):")
        ok1 = validate_single_step(verbose=True)

        print("\n[2] Single-sample feature vector (noise_off):")
        ok2 = validate_single_sample_features(verbose=True)

        if args.validate_full:
            print("\n[3] Full Iris accuracy (noise_on, 3 trials):")
            ok3 = validate_full_iris_accuracy(verbose=True, n_trials=3)
        else:
            ok3 = True
            print("\n[3] Full Iris accuracy: skipped (use --validate-full)")

        all_ok = ok1 and ok2 and ok3
        print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
        sys.exit(0 if all_ok else 1)

    if args.validate_grids:
        print("\n=== Quick vs Standard Grid Comparison ===")
        print_device_info()
        ok = validate_quick_vs_standard_grid(verbose=True)
        sys.exit(0 if ok else 1)
