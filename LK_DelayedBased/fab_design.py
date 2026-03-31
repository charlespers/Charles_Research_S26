"""
Space-filling fab sampling (Latin Hypercube) for continuous knobs + categorical assignment.
"""

from __future__ import annotations

from itertools import cycle
from typing import Any

import numpy as np

try:
    from scipy.stats import qmc
except ImportError:  # pragma: no cover
    qmc = None


def _scale_lhc(unit: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    span = np.maximum(highs - lows, np.abs(lows) * 1e-6 + 1e-12)
    return lows + unit * span


def latin_hypercube_fab_samples(
    n_samples: int,
    *,
    default_k_bounds: tuple[float, float] = (8.0, 22.0),
    i_min_bounds: tuple[float, float] = (0.4, 0.6),
    i_max_bounds: tuple[float, float] = (1.3, 1.6),
    spacing_m_bounds: tuple[float, float] = (44e-6, 56e-6),
    I_th_A_bounds: tuple[float, float] | None = None,
    lambda0_m_bounds: tuple[float, float] | None = None,
    dt_ns_bounds: tuple[float, float] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Draw n_samples LHC points in the continuous fab box. Optional bounds default to None
    -> skip that dimension (caller merges with fixed dict).
    """
    if qmc is None:
        raise ImportError("latin_hypercube_fab_samples requires scipy>=1.7")

    cols: list[tuple[str, tuple[float, float]]] = [
        ("default_k", default_k_bounds),
        ("i_min", i_min_bounds),
        ("i_max", i_max_bounds),
        ("spacing_m", spacing_m_bounds),
    ]
    if I_th_A_bounds is not None:
        cols.append(("I_th_A", I_th_A_bounds))
    if lambda0_m_bounds is not None:
        cols.append(("lambda0_m", lambda0_m_bounds))
    if dt_ns_bounds is not None:
        cols.append(("dt_ns", dt_ns_bounds))

    d = len(cols)
    lows = np.array([c[1][0] for c in cols], dtype=float)
    highs = np.array([c[1][1] for c in cols], dtype=float)

    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit = sampler.random(n=n_samples)
    X = _scale_lhc(unit, lows, highs)

    rows: list[dict[str, Any]] = []
    for i in range(n_samples):
        row = {cols[j][0]: float(X[i, j]) for j in range(d)}
        rows.append(row)
    return rows


def assign_categoricals_lhc_rows(
    rows: list[dict[str, Any]],
    *,
    motifs: list[str],
    noise_flags: list[bool],
    store_every_values: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Round-robin motifs and noise_flags onto LHC rows; optional store_every cycle."""
    out = []
    mcy = cycle(motifs)
    ncy = cycle(noise_flags)
    scy = cycle(store_every_values) if store_every_values else cycle([None])
    for r in rows:
        rr = dict(r)
        rr["motif"] = next(mcy)
        rr["noise_on"] = next(ncy)
        se = next(scy)
        if se is not None:
            rr["store_every"] = int(se)
        out.append(rr)
    return out
