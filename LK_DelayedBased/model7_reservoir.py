"""
Model 7 — Four-laser inline reservoir simulator (CPU, NumPy).

Physics model
-------------
Four semiconductor lasers arranged in a 1D inline geometry, coupled via
time-delayed optical injection. Each laser obeys the Lang-Kobayashi rate
equations (field quadratures x, y and carrier number N), integrated with
Euler-Maruyama at a fixed time step.

Coupling convention: K[i, j] = coupling strength from laser j → laser i.
Delays are computed from the physical spacing between lasers; short links
(τ < 0.1 τ_p) are treated as instantaneous (local), longer links use the
actual round-trip delay.

Reservoir feature extraction (Iris)
-------------------------------------
Each Iris sample drives the four lasers via its four normalised features
mapped to per-laser pump currents.  After a washout transient the
simulation statistics are harvested into a 20-dimensional feature vector:

    [mean(I), min(I), max(I), std(I), mean(N)]  ×  4 lasers

This 20D vector is then passed to a Ridge classifier.

Public API
----------
Topology builders
    build_inline_motif_edges()
    build_inline_coupling_matrix_tunable()
    build_inline_geometry_matrices()

Simulator
    simulate_model7_network_auto()

Iris helpers
    minmax_fit() / minmax_transform()
    features_to_pump_ratios()
    default_physics_params_ops()
    run_one_iris_sample_feature_vector()
    build_reservoir_dataset()

Utilities
    print_motif_edge_list()
    format_matrix()
"""

from __future__ import annotations

import numpy as np

# Speed of light in vacuum (m/s)
C0 = 299_792_458.0

# Type alias for a directed edge (source, destination)
Edge = tuple[int, int]


# ============================================================
# Topology builders
# ============================================================

def build_inline_motif_edges(num_lasers: int, motif: str) -> set[Edge]:
    """Return the set of directed edges (j, i) for a named coupling motif.

    Supported motifs (scale to any even/odd num_lasers ≥ 2):

    relay        — bidirectional nearest-neighbour chain
    chain        — directed nearest-neighbour chain (left → right)
    competitive  — outer lasers feed the two centre lasers; centre pair
                   bidirectional
    auxiliary    — centre lasers drive the outer lasers; centre pair
                   bidirectional
    mixed_master — directed left-half chain, bidirectional right-half
    mixed_slave  — directed right-half chain, bidirectional left-half

    Parameters
    ----------
    num_lasers : int
        Number of lasers (≥ 2).
    motif : str
        One of the motif names listed above (case-insensitive).

    Returns
    -------
    set[Edge]
        Set of (j, i) pairs where j is the source and i is the destination.
    """
    N = int(num_lasers)
    if N < 2:
        raise ValueError("num_lasers must be >= 2")
    motif = motif.lower().strip()

    centers = [N // 2 - 1, N // 2] if N % 2 == 0 else [N // 2]
    cL, cR = centers[0], centers[-1]
    edges: set[Edge] = set()

    def add(j: int, i: int):
        if j != i:
            edges.add((j, i))

    def add_bi(a: int, b: int):
        add(a, b)
        add(b, a)

    if motif == "relay":
        for n in range(N - 1):
            add_bi(n, n + 1)
    elif motif == "chain":
        for n in range(N - 1):
            add(n, n + 1)
    elif motif == "competitive":
        for j in range(0, cL):
            add(j, cL)
        for j in range(cR + 1, N):
            add(j, cR)
        if len(centers) == 2:
            add_bi(cL, cR)
    elif motif == "auxiliary":
        for i in range(0, cL):
            add(cL, i)
        for i in range(cR + 1, N):
            add(cR, i)
        if len(centers) == 2:
            add_bi(cL, cR)
    elif motif == "mixed_master":
        for n in range(0, cL):
            add(n, n + 1)
        if len(centers) == 2:
            add_bi(cL, cR)
        for n in range(cR, N - 1):
            add_bi(n, n + 1)
    elif motif == "mixed_slave":
        for n in range(0, cL):
            add(n + 1, n)
        if len(centers) == 2:
            add_bi(cL, cR)
        for n in range(cR, N - 1):
            add_bi(n, n + 1)
    else:
        raise ValueError(
            f"Unknown motif '{motif}'. "
            "Choose: relay, chain, competitive, auxiliary, mixed_master, mixed_slave"
        )

    return edges


def build_inline_coupling_matrix_tunable(
    num_lasers: int,
    motif: str,
    *,
    default_k: float = 0.0,
    k_ji: dict[Edge, float] | None = None,
    strict: bool = False,
) -> np.ndarray:
    """Build the N×N coupling matrix K for a given motif.

    K[i, j] holds the coupling strength from laser j to laser i.
    Off-motif entries are zero; diagonal is always zero.

    Parameters
    ----------
    num_lasers : int
        Number of lasers.
    motif : str
        Coupling motif name (see ``build_inline_motif_edges``).
    default_k : float
        Strength applied to every motif edge that is not in ``k_ji``.
    k_ji : dict, optional
        Per-edge overrides: {(j, i): strength, ...}.
    strict : bool
        If True, raise if ``k_ji`` contains edges outside the motif.

    Returns
    -------
    np.ndarray, shape (num_lasers, num_lasers)
    """
    N = int(num_lasers)
    edges = build_inline_motif_edges(N, motif)
    k_ji = {} if k_ji is None else dict(k_ji)

    if strict:
        extra = set(k_ji.keys()) - edges
        if extra:
            raise ValueError(f"k_ji contains edges not in motif: {sorted(extra)}")

    K = np.zeros((N, N), dtype=float)
    for (j, i) in edges:
        K[i, j] = float(k_ji.get((j, i), default_k))
    np.fill_diagonal(K, 0.0)
    return K


def build_inline_geometry_matrices(
    num_lasers: int,
    spacing_m: float,
    lambda0_m: float,
    n_air: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise distance, delay, and phase matrices for inline lasers.

    Lasers are placed at positions x_i = i × spacing_m.

    Parameters
    ----------
    num_lasers : int
        Number of lasers.
    spacing_m : float
        Centre-to-centre spacing (m).
    lambda0_m : float
        Free-space wavelength (m).
    n_air : float
        Refractive index of the medium between lasers.

    Returns
    -------
    L      : np.ndarray (N, N)  pairwise distances (m)
    tau_ns : np.ndarray (N, N)  propagation delays (ns)
    phi    : np.ndarray (N, N)  round-trip phase (rad), in [0, 2π)
    cos_phi: np.ndarray (N, N)  cos(phi)
    sin_phi: np.ndarray (N, N)  sin(phi)
    """
    idx = np.arange(num_lasers)
    L = np.abs(idx[:, None] - idx[None, :]) * spacing_m
    tau_ns = (n_air * L / C0) * 1e9
    phi = np.mod(2.0 * np.pi * n_air * L / lambda0_m, 2.0 * np.pi)
    return L, tau_ns, phi, np.cos(phi), np.sin(phi)


# ============================================================
# Core ODE simulator
# ============================================================

def simulate_model7_network_auto(
    t_span_ns: tuple[float, float],
    dt_ns: float,
    y0: np.ndarray,
    params: dict,
    K: np.ndarray,
    tau_ns: np.ndarray,
    cos_phi: np.ndarray,
    sin_phi: np.ndarray,
    *,
    noise_on: bool = True,
    seed: int | None = None,
    store_every: int = 1,
    P_step: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Integrate the four-laser Lang-Kobayashi system with Euler-Maruyama.

    Per-laser equations
    -------------------
    S_i   = x_i² + y_i²                         (photon number)
    G_i   = g (N_i − N₀) / (1 + s S_i)          (modal gain)

    Injection into laser i:
      inj_x,i = Σ_j  K[i,j] [ cos φ_ij x_j(t−τ_ij) − sin φ_ij y_j(t−τ_ij) ]
      inj_y,i = Σ_j  K[i,j] [ cos φ_ij y_j(t−τ_ij) + sin φ_ij x_j(t−τ_ij) ]
    (local links use E_j(t) instead of the delayed value)

    Field / carrier ODEs:
      dx_i/dt = 0.5 (G_i − γ)(x_i − α y_i) + inj_x,i  +  √(R/2) ξ_x
      dy_i/dt = 0.5 (G_i − γ)(y_i + α x_i) + inj_y,i  +  √(R/2) ξ_y
      dN_i/dt = P_i − γ_e N_i − G_i S_i
    where R = β_sp γ_e max(N_i, 0) is the spontaneous emission rate.

    Link classification:
      τ_ij < 0.1 τ_p  →  local   (instantaneous field, E_j(t))
      otherwise        →  delayed (E_j(t − τ_ij), rounded to nearest step)

    Parameters
    ----------
    t_span_ns : (t_start, t_end) in nanoseconds.
    dt_ns     : Euler time step (ns).
    y0        : Initial state, shape (Nlas, 3). Columns: [x, y, N].
    params    : Physics dict (see ``default_physics_params_ops``).
    K         : Coupling matrix (Nlas, Nlas).
    tau_ns    : Pairwise delay matrix (Nlas, Nlas) in ns.
    cos_phi, sin_phi : Precomputed phase matrices (Nlas, Nlas).
    noise_on  : Include spontaneous emission noise (Euler-Maruyama).
    seed      : NumPy random seed for reproducibility.
    store_every : Decimate output — store one sample every this many steps.
    P_step    : Optional time-varying pump array, shape (n_steps+1, Nlas).
                If None, ``params["P"]`` is used as a constant pump.

    Returns
    -------
    t_out : (n_out,)          time vector (ns)
    x_out : (n_out, Nlas)     x-quadrature
    y_out : (n_out, Nlas)     y-quadrature
    N_out : (n_out, Nlas)     carrier number
    meta  : dict              simulation metadata
    """
    if seed is not None:
        np.random.seed(seed)

    t0, tf = t_span_ns
    n_steps = int((tf - t0) / dt_ns)
    Nlas = y0.shape[0]

    alpha   = float(params["alpha"])
    g       = float(params["g"])
    s       = float(params["s"])
    gamma   = float(params["gamma"])
    gamma_e = float(params["gamma_e"])
    beta_sp = float(params["beta_sp"])

    P_const = np.array(params["P"], dtype=float)
    if P_const.shape != (Nlas,):
        raise ValueError(f'params["P"] must have shape ({Nlas},), got {P_const.shape}')

    if P_step is not None:
        P_step = np.asarray(P_step, dtype=float)
        if P_step.shape != (n_steps + 1, Nlas):
            raise ValueError(
                f"P_step must have shape ({n_steps + 1}, {Nlas}), got {P_step.shape}"
            )

    N0 = params["N0"]
    N0 = float(N0) * np.ones(Nlas) if np.ndim(N0) == 0 else np.asarray(N0, dtype=float)

    tau_p_ns  = 1.0 / gamma
    tau_thresh = 0.1 * tau_p_ns

    delay_steps = np.rint(tau_ns / dt_ns).astype(int)
    use_delayed = tau_ns >= tau_thresh
    np.fill_diagonal(use_delayed, False)
    np.fill_diagonal(delay_steps, 0)

    Mxc = K * cos_phi
    Mxs = K * sin_phi
    local_Mxc = Mxc * (~use_delayed)
    local_Mxs = Mxs * (~use_delayed)

    unique_delays = np.unique(delay_steps[use_delayed])
    unique_delays = unique_delays[unique_delays > 0]

    group_Mxc = {}
    group_Mxs = {}
    for d in unique_delays:
        mask_d = use_delayed & (delay_steps == d)
        group_Mxc[d] = Mxc * mask_d
        group_Mxs[d] = Mxs * mask_d

    # Full history arrays (needed for delay look-back)
    x_full = np.zeros((n_steps + 1, Nlas), dtype=float)
    y_full = np.zeros((n_steps + 1, Nlas), dtype=float)
    N_full = np.zeros((n_steps + 1, Nlas), dtype=float)
    x_full[0] = y0[:, 0]
    y_full[0] = y0[:, 1]
    N_full[0] = y0[:, 2]

    sqrt_dt = np.sqrt(dt_ns)
    store_every = max(1, int(store_every))
    out_len = n_steps // store_every + 1

    t_out = np.empty(out_len, dtype=float)
    x_out = np.empty((out_len, Nlas), dtype=float)
    y_out = np.empty((out_len, Nlas), dtype=float)
    N_out = np.empty((out_len, Nlas), dtype=float)

    out_i = 0
    t_out[0] = t0
    x_out[0] = x_full[0]
    y_out[0] = y_full[0]
    N_out[0] = N_full[0]

    for n in range(n_steps):
        xn, yn, Nn = x_full[n], y_full[n], N_full[n]

        S = xn * xn + yn * yn
        G = g * (Nn - N0) / (1.0 + s * S)

        inj_x = local_Mxc @ xn - local_Mxs @ yn
        inj_y = local_Mxc @ yn + local_Mxs @ xn

        for d in unique_delays:
            if n - d >= 0:
                xd, yd = x_full[n - d], y_full[n - d]
            else:
                xd, yd = xn, yn
            inj_x += group_Mxc[d] @ xd - group_Mxs[d] @ yd
            inj_y += group_Mxc[d] @ yd + group_Mxs[d] @ xd

        dx = 0.5 * (G - gamma) * (xn - alpha * yn) + inj_x
        dy = 0.5 * (G - gamma) * (yn + alpha * xn) + inj_y

        P_now = P_step[n] if P_step is not None else P_const
        dN = P_now - gamma_e * Nn - G * S

        if noise_on:
            Rsp  = beta_sp * gamma_e * np.maximum(Nn, 0.0)
            diff = np.sqrt(Rsp / 2.0) * sqrt_dt
            x_full[n + 1] = xn + dx * dt_ns + diff * np.random.randn(Nlas)
            y_full[n + 1] = yn + dy * dt_ns + diff * np.random.randn(Nlas)
        else:
            x_full[n + 1] = xn + dx * dt_ns
            y_full[n + 1] = yn + dy * dt_ns

        N_full[n + 1] = Nn + dN * dt_ns

        if (n + 1) % store_every == 0:
            out_i += 1
            t_out[out_i] = t0 + (n + 1) * dt_ns
            x_out[out_i] = x_full[n + 1]
            y_out[out_i] = y_full[n + 1]
            N_out[out_i] = N_full[n + 1]

    meta = {
        "num_lasers":        Nlas,
        "noise_on":          bool(noise_on),
        "dt_ns":             dt_ns,
        "store_every":       store_every,
        "tau_p_ns":          tau_p_ns,
        "tau_threshold_ns":  0.1 * tau_p_ns,
        "unique_delay_steps": unique_delays.tolist(),
        "P_step_used":       P_step is not None,
    }
    return t_out, x_out, y_out, N_out, meta


# ============================================================
# Iris reservoir helpers
# ============================================================

def minmax_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature min and span from a training matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    xmin : per-feature minimum
    span : per-feature range (clamped to 1.0 where range is zero)
    """
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    span = np.where((xmax - xmin) > 0, (xmax - xmin), 1.0)
    return xmin, span


def minmax_transform(
    X: np.ndarray,
    xmin: np.ndarray,
    span: np.ndarray,
) -> np.ndarray:
    """Apply a fitted min-max normalisation to map X into [0, 1]."""
    return (X - xmin) / span


def features_to_pump_ratios(
    f01: np.ndarray,
    rmin: float = 0.5,
    rmax: float = 1.1,
) -> np.ndarray:
    """Map normalised features in [0, 1] to pump-current ratios in [rmin, rmax]."""
    return rmin + np.clip(f01, 0.0, 1.0) * (rmax - rmin)


def default_physics_params_ops(P_ops_per_laser: np.ndarray) -> dict:
    """Return the standard Model 7 physics parameter dictionary.

    Parameters
    ----------
    P_ops_per_laser : np.ndarray, shape (Nlas,)
        Per-laser pump rate in carriers/ns.

    Returns
    -------
    dict with keys: alpha, g, N0, s, gamma, gamma_e, beta_sp, P
    """
    return {
        "alpha":   3.0,
        "g":       1.2e-5,
        "N0":      1.25e8,
        "s":       5e-7,
        "gamma":   496.0,
        "gamma_e": 0.651,
        "beta_sp": 1e-5,
        "P":       np.asarray(P_ops_per_laser, dtype=float),
    }


def run_one_iris_sample_feature_vector(
    sample_features_01: np.ndarray,
    *,
    motif: str,
    spacing_m: float,
    lambda0_m: float,
    n_air: float,
    I_th_A: float,
    i_min: float,
    i_max: float,
    t_span_ns: tuple[float, float],
    washout_ns: float,
    dt_ns: float,
    store_every: int,
    noise_on: bool,
    seed: int | None,
    default_k: float,
) -> np.ndarray:
    """Simulate one Iris sample through the reservoir and return a 20D feature vector.

    The four normalised Iris features are mapped to per-laser pump currents,
    the four-laser system is simulated for ``t_span_ns`` nanoseconds, and the
    first ``washout_ns`` nanoseconds are discarded as transient.  The remaining
    time series is summarised as:

        [mean(I), min(I), max(I), std(I), mean(N)]  for each laser   (20 values)

    where I = x² + y² is the photon number.

    Parameters
    ----------
    sample_features_01 : np.ndarray, shape (4,)
        Normalised Iris feature vector in [0, 1].
    motif : str
        Laser coupling topology.
    spacing_m : float
        Laser spacing (m).
    lambda0_m : float
        Emission wavelength (m).
    n_air : float
        Refractive index of medium.
    I_th_A : float
        Threshold current (A).
    i_min, i_max : float
        Pump-current ratio bounds (units of I_th).
    t_span_ns : (t0, tf)
        Simulation window in nanoseconds.
    washout_ns : float
        Transient to discard (ns); must be < tf.
    dt_ns : float
        Euler integration step (ns).
    store_every : int
        Decimation factor for stored output.
    noise_on : bool
        Include spontaneous emission noise.
    seed : int or None
        Random seed for reproducibility.
    default_k : float
        Uniform coupling strength applied to all motif edges.

    Returns
    -------
    np.ndarray, shape (20,)
    """
    num_lasers = 4
    sample_features_01 = np.asarray(sample_features_01, dtype=float)
    if sample_features_01.shape != (4,):
        raise ValueError(f"Expected shape (4,), got {sample_features_01.shape}")

    pump_ratios = features_to_pump_ratios(sample_features_01, rmin=i_min, rmax=i_max)
    e = 1.602e-19
    P_ops = (pump_ratios * I_th_A / e) * 1e-9

    params = default_physics_params_ops(P_ops)
    K = build_inline_coupling_matrix_tunable(num_lasers, motif, default_k=default_k)

    y0 = np.zeros((num_lasers, 3), dtype=float)
    y0[:, 0] = 1e-3  # small initial field

    _, tau_ns, _, cos_phi, sin_phi = build_inline_geometry_matrices(
        num_lasers, spacing_m, lambda0_m, n_air
    )

    t, x, y, Ncar, _ = simulate_model7_network_auto(
        t_span_ns=t_span_ns,
        dt_ns=dt_ns,
        y0=y0,
        params=params,
        K=K,
        tau_ns=tau_ns,
        cos_phi=cos_phi,
        sin_phi=sin_phi,
        noise_on=noise_on,
        seed=seed,
        store_every=store_every,
        P_step=None,
    )

    mask = t >= washout_ns
    if not np.any(mask):
        raise ValueError(
            "washout_ns too large: no stored samples remain. "
            "Reduce washout_ns or store_every, or increase t_span_ns."
        )

    Iw = (x**2 + y**2)[mask]
    Nw = Ncar[mask]

    feats = []
    for k in range(num_lasers):
        feats.extend([
            float(np.mean(Iw[:, k])),
            float(np.min(Iw[:, k])),
            float(np.max(Iw[:, k])),
            float(np.std(Iw[:, k])),
            float(np.mean(Nw[:, k])),
        ])
    return np.asarray(feats, dtype=float)


def build_reservoir_dataset(
    X01: np.ndarray,
    y: np.ndarray,
    *,
    motif: str,
    spacing_m: float,
    lambda0_m: float,
    n_air: float,
    I_th_A: float,
    i_min: float,
    i_max: float,
    t_span_ns: tuple[float, float],
    washout_ns: float,
    dt_ns: float,
    store_every: int,
    noise_on: bool,
    base_seed: int | None,
    default_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a reservoir feature matrix for the full Iris dataset (CPU).

    Calls ``run_one_iris_sample_feature_vector`` sequentially for each row of
    ``X01``.  For GPU-accelerated batch processing use
    ``model7_reservoir_gpu.build_reservoir_dataset_gpu`` instead.

    Parameters
    ----------
    X01 : np.ndarray, shape (N, 4)
        Normalised Iris features in [0, 1].
    y : np.ndarray, shape (N,)
        Integer class labels.
    base_seed : int or None
        If provided, sample i uses seed ``base_seed + i`` for reproducibility.
    (remaining kwargs as in ``run_one_iris_sample_feature_vector``)

    Returns
    -------
    F : np.ndarray, shape (N, 20)   reservoir feature matrix
    y : np.ndarray, shape (N,)      labels (unchanged)
    """
    X01 = np.asarray(X01, dtype=float)
    y   = np.asarray(y,   dtype=int)
    N   = X01.shape[0]
    F   = np.zeros((N, 20), dtype=float)

    for i in range(N):
        seed = None if base_seed is None else (base_seed + i)
        F[i] = run_one_iris_sample_feature_vector(
            X01[i],
            motif=motif, spacing_m=spacing_m, lambda0_m=lambda0_m, n_air=n_air,
            I_th_A=I_th_A, i_min=i_min, i_max=i_max,
            t_span_ns=t_span_ns, washout_ns=washout_ns, dt_ns=dt_ns,
            store_every=store_every, noise_on=noise_on, seed=seed,
            default_k=default_k,
        )
    return F, y


# ============================================================
# Utilities
# ============================================================

def print_motif_edge_list(num_lasers: int, motif: str) -> None:
    """Print the directed edge list for a motif (1-based laser indices)."""
    edges = sorted(build_inline_motif_edges(num_lasers, motif))
    print(f"\nEdges for motif='{motif}' (1-based labels):")
    for j, i in edges:
        print(f"  {j + 1} -> {i + 1}")


def format_matrix(K: np.ndarray, fmt: str = "{:10.3f}") -> str:
    """Return a human-readable string representation of a matrix."""
    return "\n".join(" ".join(fmt.format(v) for v in row) for row in K)
