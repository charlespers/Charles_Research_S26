"""
Programmatic Iris + Model 7 benchmark (used by IrisFlower+Reservior.py and eval_sweeps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import load_iris

from model7_reservoir import build_reservoir_dataset, minmax_fit, minmax_transform
from output_io import ensure_run_dir, new_run_id, save_json


def run_iris_model7_benchmark(
    *,
    motif: str = "auxiliary",
    dt_ns: float = 5e-4,
    store_every: int = 400,
    noise_on: bool = True,
    base_seed: int | None = 30,
    t_span_ns: tuple[float, float] = (0.0, 20.0),
    washout_ns: float = 10.0,
    spacing_m: float = 50e-6,
    lambda0_m: float = 850e-9,
    n_air: float = 1.0,
    I_th_A: float = 17.35e-3,
    default_k: float = 15.0,
    i_min: float = 0.5,
    i_max: float = 1.5,
    test_size: float = 0.25,
    random_state_split: int = 0,
    save_outputs: bool = False,
    outputs_base: Path | None = None,
    run_id: str | None = None,
    use_gpu: bool = False,
) -> dict[str, Any]:
    iris = load_iris()
    X = iris.data.astype(float)
    y = iris.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state_split,
        stratify=y,
    )

    xmin, span = minmax_fit(X_train)
    X_train01 = minmax_transform(X_train, xmin, span)
    X_test01 = minmax_transform(X_test, xmin, span)

    if use_gpu:
        from model7_reservoir_gpu import build_gpu_sim_params, build_reservoir_dataset_gpu
        _gp = build_gpu_sim_params(
            motif=motif, spacing_m=spacing_m, lambda0_m=lambda0_m, n_air=n_air,
            I_th_A=I_th_A, i_min=i_min, i_max=i_max, dt_ns=dt_ns,
            t_span_ns=t_span_ns, washout_ns=washout_ns, store_every=store_every,
            noise_on=noise_on, default_k=default_k,
        )
        F_train, y_tr = build_reservoir_dataset_gpu(
            X_train01, y_train, gpu_params=_gp, base_seed=base_seed,
        )
        test_seed = None if base_seed is None else (base_seed + 10_000)
        F_test, y_te = build_reservoir_dataset_gpu(
            X_test01, y_test, gpu_params=_gp, base_seed=test_seed,
        )
    else:
        F_train, y_tr = build_reservoir_dataset(
            X_train01,
            y_train,
            motif=motif,
            spacing_m=spacing_m,
            lambda0_m=lambda0_m,
            n_air=n_air,
            I_th_A=I_th_A,
            i_min=i_min,
            i_max=i_max,
            t_span_ns=t_span_ns,
            washout_ns=washout_ns,
            dt_ns=dt_ns,
            store_every=store_every,
            noise_on=noise_on,
            base_seed=base_seed,
            default_k=default_k,
        )

        test_seed = None if base_seed is None else (base_seed + 10_000)
        F_test, y_te = build_reservoir_dataset(
            X_test01,
            y_test,
            motif=motif,
            spacing_m=spacing_m,
            lambda0_m=lambda0_m,
            n_air=n_air,
            I_th_A=I_th_A,
            i_min=i_min,
            i_max=i_max,
            t_span_ns=t_span_ns,
            washout_ns=washout_ns,
            dt_ns=dt_ns,
            store_every=store_every,
            noise_on=noise_on,
            base_seed=test_seed,
            default_k=default_k,
        )

    clf = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeClassifier())])
    clf.fit(F_train, y_tr)
    y_pred = clf.predict(F_test)
    acc = float(accuracy_score(y_te, y_pred))

    out: dict[str, Any] = {
        "accuracy_test": acc,
        "y_test": y_te,
        "y_pred": y_pred,
    }

    if save_outputs:
        base = outputs_base or Path(__file__).resolve().parent
        rid = run_id or new_run_id("iris_baseline")
        out_dir = ensure_run_dir(base, "iris", "baseline", rid)
        config = {
            "benchmark": "iris",
            "run_id": rid,
            "motif": motif,
            "dt_ns": dt_ns,
            "store_every": store_every,
            "t_span_ns": list(t_span_ns),
            "washout_ns": washout_ns,
            "noise_on": noise_on,
            "base_seed": base_seed,
            "default_k": default_k,
            "spacing_m": spacing_m,
            "lambda0_m": lambda0_m,
            "n_air": n_air,
            "I_th_A": I_th_A,
            "i_min": i_min,
            "i_max": i_max,
            "test_size": test_size,
            "random_state_split": random_state_split,
        }
        save_json(out_dir / "run_config.json", config)
        save_json(out_dir / "metrics.json", {"accuracy_test": acc})
        out["output_dir"] = str(out_dir)

    return out
