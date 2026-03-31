"""
Fab recommender: workload descriptors → recommended laser-reservoir settings.

Pipeline: fab_sweep → sweep_long.csv → aggregate_optimal (robust / surrogate / Pareto)
→ train_recommender (LOOCV by scenario, MLP + HistGradientBoosting + kNN baselines).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from output_io import ensure_run_dir, new_run_id, save_json

SCENARIO_FEATURE_COLS = [
    "task_encoding",
    "bench_n_train",
    "bench_n_test",
    "bench_washout_macros",
    "bench_hold_ns",
    "log10_workload_tokens",
    "log10_sim_budget",
]

# Extended fab outputs (subset may exist in older CSVs)
OPTIMAL_COL_ORDER = [
    "optimal_default_k",
    "optimal_i_min",
    "optimal_i_max",
    "optimal_spacing_m",
    "optimal_noise_float",
    "optimal_I_th_A",
    "optimal_lambda0_m",
    "optimal_dt_ns",
    "optimal_store_every",
]

MODEL_BUNDLE_NAME = "fab_recommender_bundle.joblib"

FAB_NUMERIC_FOR_SURROGATE = [
    "default_k",
    "i_min",
    "i_max",
    "spacing_m",
    "noise_on",
    "I_th_A",
    "lambda0_m",
    "dt_ns",
    "store_every",
]


def _log10_workload(row: pd.Series) -> float:
    tok = (
        float(row["bench_n_train"])
        + float(row["bench_n_test"])
        + float(row["bench_washout_macros"])
        + 1.0
    )
    return float(np.log10(max(tok, 1.0)))


def _robust_score(
    metric_mean: float,
    metric_std: float,
    *,
    maximize: bool,
    penalty_lambda: float,
) -> float:
    s = float(metric_std) if not np.isnan(metric_std) else 0.0
    if maximize:
        return float(metric_mean) - penalty_lambda * s
    return float(metric_mean) + penalty_lambda * s


def _pareto_mask(
    quality: np.ndarray,
    latency: np.ndarray,
    *,
    maximize_quality: bool,
) -> np.ndarray:
    """Non-dominated: higher quality better, lower latency better."""
    q = np.asarray(quality, dtype=float)
    if not maximize_quality:
        q = -q
    lat = np.asarray(latency, dtype=float)
    n = len(q)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if q[j] >= q[i] and lat[j] <= lat[i] and (q[j] > q[i] or lat[j] < lat[i]):
                keep[i] = False
                break
    return keep


def aggregate_optimal(
    long_csv: Path,
    out_csv: Path | None = None,
    *,
    robust_penalty_lambda: float = 0.0,
    use_surrogate_optimum: bool = False,
    pareto_multi_objective: bool = False,
    pareto_front_csv: Path | None = None,
) -> Path:
    """
    One optimal fab row per scenario_tag.
    Uses metric_mean / metric_std when present; else metric.
    pareto_multi_objective: among Pareto-efficient (quality, latency) points, pick best robust score.
    """
    df = pd.read_csv(long_csv)
    if "scenario_tag" not in df.columns:
        raise ValueError("sweep_long.csv must contain scenario_tag")

    if "metric_mean" in df.columns:
        mcol, scol = "metric_mean", "metric_std"
    else:
        mcol, scol = "metric", "metric_std"
        if scol not in df.columns:
            df = df.copy()
            df["metric_std"] = 0.0

    if "latency_euler_steps" not in df.columns:
        df = df.copy()
        df["latency_euler_steps"] = 0

    if "log10_sim_budget" not in df.columns:
        df = df.copy()
        df["log10_sim_budget"] = 0.0

    pareto_rows: list[dict[str, Any]] = []
    rows_out: list[dict[str, Any]] = []

    for tag, g in df.groupby("scenario_tag", sort=False):
        g = g.reset_index(drop=True)
        mx = bool(g["maximize_metric"].iloc[0])
        lat = g["latency_euler_steps"].values.astype(float)
        mmean = g[mcol].values.astype(float)
        mstd = g[scol].values.astype(float) if scol in g.columns else np.zeros(len(g))

        scores = np.array(
            [
                _robust_score(float(mmean[i]), float(mstd[i]), maximize=mx, penalty_lambda=robust_penalty_lambda)
                for i in range(len(g))
            ]
        )

        def _pick_extreme(arr: np.ndarray) -> int:
            return int(np.argmax(arr) if mx else np.argmin(arr))

        best_rs = 0.0
        if pareto_multi_objective and len(g) > 1:
            pm = _pareto_mask(scores, lat, maximize_quality=mx)
            idx_p = np.where(pm)[0]
            if len(idx_p) == 0:
                pick = _pick_extreme(scores)
                best = g.iloc[pick]
                best_rs = float(scores[pick])
            else:
                for i in idx_p:
                    r = g.iloc[i].to_dict()
                    r["pareto_on_front"] = True
                    r["robust_score"] = float(scores[i])
                    pareto_rows.append(r)
                sub_scores = scores[idx_p]
                pick_local = _pick_extreme(sub_scores)
                best = g.iloc[idx_p[pick_local]]
                best_rs = float(sub_scores[pick_local])
        elif use_surrogate_optimum and len(g) >= 8:
            Xs, yv = _surrogate_xy(g, scores)
            from sklearn.ensemble import HistGradientBoostingRegressor

            sur = HistGradientBoostingRegressor(
                max_depth=4,
                max_iter=80,
                random_state=0,
                learning_rate=0.08,
            )
            sur.fit(Xs, yv)
            pred = sur.predict(Xs)
            pick = _pick_extreme(pred)
            best = g.iloc[pick]
            best_rs = float(scores[pick])
        else:
            pick = _pick_extreme(scores)
            best = g.iloc[pick]
            best_rs = float(scores[pick])

        row = _optimal_row_from_best(best, mx, best_rs)
        rows_out.append(row)

    out = pd.DataFrame(rows_out)
    if out_csv is None:
        out_csv = long_csv.parent / "optimal_per_scenario.csv"
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(out)} scenarios)")

    if pareto_front_csv is not None and pareto_rows:
        pd.DataFrame(pareto_rows).to_csv(pareto_front_csv, index=False)
        print(f"Wrote Pareto front {pareto_front_csv} ({len(pareto_rows)} rows)")

    return Path(out_csv)


def _surrogate_xy(g: pd.DataFrame, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    motif_le = LabelEncoder()
    mc = motif_le.fit_transform(g["motif"].astype(str))
    Xl = []
    for c in FAB_NUMERIC_FOR_SURROGATE:
        Xl.append(g[c].values.astype(float))
    Xl.append(mc.astype(float))
    X = np.column_stack(Xl)
    return X, scores


def _optimal_row_from_best(best: pd.Series, mx: bool, robust_score: float) -> dict[str, Any]:
    def g(c: str, default: float = float("nan")) -> float:
        return float(best[c]) if c in best.index and pd.notna(best[c]) else default

    row: dict[str, Any] = {
        "scenario_tag": str(best["scenario_tag"]),
        "task": str(best["task"]),
        "task_encoding": int(best["task_encoding"]),
        "bench_n_train": int(best["bench_n_train"]),
        "bench_n_test": int(best["bench_n_test"]),
        "bench_washout_macros": int(best["bench_washout_macros"]),
        "bench_hold_ns": float(best["bench_hold_ns"]),
        "log10_workload_tokens": _log10_workload(best),
        "log10_sim_budget": float(best.get("log10_sim_budget", 0.0)),
        "optimal_default_k": g("default_k"),
        "optimal_i_min": g("i_min"),
        "optimal_i_max": g("i_max"),
        "optimal_spacing_m": g("spacing_m"),
        "optimal_noise_float": g("noise_on"),
        "optimal_motif": str(best["motif"]),
        "best_metric": float(best.get("metric_mean", best.get("metric", 0))),
        "metric_std_at_optimum": float(best.get("metric_std", 0.0)),
        "metric_name": str(best["metric_name"]),
        "maximize_metric": mx,
        "robust_score": robust_score,
        "latency_euler_steps_at_optimum": int(best.get("latency_euler_steps", 0)),
    }
    for oc, src in [
        ("optimal_I_th_A", "I_th_A"),
        ("optimal_lambda0_m", "lambda0_m"),
        ("optimal_dt_ns", "dt_ns"),
        ("optimal_store_every", "store_every"),
    ]:
        if src in best.index:
            row[oc] = g(src)
    return row


def _active_optimal_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in OPTIMAL_COL_ORDER if c in df.columns]


def train_recommender(
    optimal_csv: Path,
    out_dir: Path | None = None,
    *,
    test_fraction: float = 0.2,
    random_state: int = 0,
    hidden_regressor: tuple[int, ...] = (128, 64),
    hidden_classifier: tuple[int, ...] = (64, 32),
    use_group_cv: bool = True,
    run_baselines: bool = True,
) -> Path:
    df = pd.read_csv(optimal_csv)
    if len(df) < 4:
        raise ValueError(f"Need at least 4 optimal scenarios; got {len(df)}")

    for c in SCENARIO_FEATURE_COLS:
        if c not in df.columns:
            if c == "log10_workload_tokens":
                df[c] = df.apply(_log10_workload, axis=1)
            elif c == "log10_sim_budget":
                df[c] = 0.0
            else:
                raise ValueError(f"Missing column {c}")

    ycols = _active_optimal_cols(df)
    if not ycols:
        raise ValueError("No optimal_* columns found")

    X = df[SCENARIO_FEATURE_COLS].values.astype(np.float64)
    Yc = df[ycols].values.astype(np.float64)
    groups = df["scenario_tag"].values
    le = LabelEncoder()
    ym = le.fit_transform(df["optimal_motif"].astype(str))

    n = len(df)
    if n < 12:
        hidden_regressor = (64, 32)
        hidden_classifier = (32, 16)

    n_tr = max(1, int(n * (1 - test_fraction)))
    use_es = n_tr >= 16

    sx = StandardScaler()
    sy = StandardScaler()
    X_train, X_test, y_train, y_test, ym_train, ym_test, g_train, g_test = train_test_split(
        X, Yc, ym, groups, test_size=test_fraction, random_state=random_state
    )
    X_train = sx.fit_transform(X_train)
    X_test = sx.transform(X_test)
    y_train_n = sy.fit_transform(y_train)
    y_test_n = sy.transform(y_test)

    reg_mlp = MLPRegressor(
        hidden_layer_sizes=hidden_regressor,
        max_iter=5000,
        random_state=random_state,
        early_stopping=use_es,
        validation_fraction=0.15 if use_es else 0.0,
        alpha=1e-4,
        learning_rate_init=1e-3,
    )
    reg_mlp.fit(X_train, y_train_n)
    y_pred_test = sy.inverse_transform(reg_mlp.predict(X_test))
    r2_mlp_holdout = float(r2_score(y_test, y_pred_test, multioutput="uniform_average"))
    mae_mlp = mean_absolute_error(y_test, y_pred_test, multioutput="raw_values").tolist()

    reg_hgb = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_depth=5,
            max_iter=120,
            random_state=random_state,
            learning_rate=0.06,
        )
    )
    reg_hgb.fit(X_train, y_train)
    y_pred_hgb = reg_hgb.predict(X_test)
    r2_hgb_holdout = float(r2_score(y_test, y_pred_hgb, multioutput="uniform_average"))

    knn_k = max(1, min(3, n // 3))
    knn = KNeighborsRegressor(n_neighbors=knn_k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    r2_knn_holdout = float(r2_score(y_test, y_pred_knn, multioutput="uniform_average"))
    knn_full = KNeighborsRegressor(n_neighbors=knn_k)
    knn_full.fit(sx.fit_transform(X), Yc)

    n_motif = len(np.unique(ym))
    clf = None
    acc_motif = float("nan")
    if n_motif >= 2:
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_classifier,
            max_iter=3000,
            random_state=random_state,
            early_stopping=use_es,
            validation_fraction=0.15 if use_es else 0.0,
            alpha=1e-4,
        )
        clf.fit(X_train, ym_train)
        acc_motif = float(accuracy_score(ym_test, clf.predict(X_test)))

    # Leave-one-scenario-out CV on continuous targets
    loocv_report: dict[str, Any] = {}
    if use_group_cv and len(np.unique(groups)) >= 3:
        Xf = sx.fit_transform(X)
        Yf = Yc
        logo = LeaveOneGroupOut()
        r2_mlp_cv: list[float] = []
        r2_hgb_cv: list[float] = []
        r2_knn_cv: list[float] = []
        for tr, te in logo.split(Xf, Yf, groups):
            sy_cv = StandardScaler()
            ytr = sy_cv.fit_transform(Yf[tr])
            yte = Yf[te]
            m = MLPRegressor(
                hidden_layer_sizes=hidden_regressor,
                max_iter=2000,
                random_state=random_state,
                early_stopping=False,
                alpha=1e-3,
            )
            if len(te) < 2:
                r2_mlp_cv.append(float("nan"))
                r2_hgb_cv.append(float("nan"))
                r2_knn_cv.append(float("nan"))
                continue
            m.fit(Xf[tr], ytr)
            pred = sy_cv.inverse_transform(m.predict(Xf[te]))
            r2_mlp_cv.append(r2_score(yte, pred, multioutput="uniform_average"))
            h = MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    max_depth=4, max_iter=80, random_state=random_state
                )
            )
            h.fit(Xf[tr], Yf[tr])
            r2_hgb_cv.append(r2_score(yte, h.predict(Xf[te]), multioutput="uniform_average"))
            kn = KNeighborsRegressor(n_neighbors=max(1, min(3, len(tr) // 2)))
            kn.fit(Xf[tr], Yf[tr])
            r2_knn_cv.append(r2_score(yte, kn.predict(Xf[te]), multioutput="uniform_average"))
        loocv_report = {
            "mlp_r2_mean": float(np.nanmean(r2_mlp_cv)),
            "mlp_r2_std": float(np.nanstd(r2_mlp_cv)),
            "hgb_r2_mean": float(np.nanmean(r2_hgb_cv)),
            "hgb_r2_std": float(np.nanstd(r2_hgb_cv)),
            "knn_r2_mean": float(np.nanmean(r2_knn_cv)),
            "knn_r2_std": float(np.nanstd(r2_knn_cv)),
            "n_splits": len(r2_mlp_cv),
        }

    baselines: dict[str, Any] = {}
    if run_baselines:
        y_mean = Yc.mean(axis=0, keepdims=True)
        y_bl = np.repeat(y_mean, len(y_test), axis=0)
        baselines["constant_r2_holdout"] = float(
            r2_score(y_test, y_bl, multioutput="uniform_average")
        )

    primary = "hist_gbrt"
    if loocv_report:
        if loocv_report.get("mlp_r2_mean", -1e9) >= loocv_report.get("hgb_r2_mean", -1e9):
            primary = "mlp"
    elif r2_mlp_holdout >= r2_hgb_holdout:
        primary = "mlp"
    else:
        primary = "hist_gbrt"

    reg_hgb_full = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_depth=5, max_iter=120, random_state=random_state, learning_rate=0.06
        )
    )
    reg_hgb_full.fit(sx.fit_transform(X), Yc)

    reg_mlp_full = MLPRegressor(
        hidden_layer_sizes=hidden_regressor,
        max_iter=5000,
        random_state=random_state,
        early_stopping=n >= 16,
        validation_fraction=0.15 if n >= 16 else 0.0,
        alpha=1e-4,
    )
    sy_full = StandardScaler()
    reg_mlp_full.fit(sx.fit_transform(X), sy_full.fit_transform(Yc))

    base = Path(__file__).resolve().parent
    out_dir = out_dir or ensure_run_dir(base, "analysis", "fab_recommender", new_run_id("train"))
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "scaler_X": sx,
        "scaler_Y_cont": sy_full,
        "label_encoder_motif": le,
        "mlp_continuous": reg_mlp_full,
        "hgb_continuous": reg_hgb_full,
        "knn_continuous": knn_full,
        "mlp_motif": clf,
        "scenario_feature_cols": SCENARIO_FEATURE_COLS,
        "continuous_fab_cols": ycols,
        "single_motif_fallback": clf is None,
        "primary_regressor": primary,
    }
    joblib.dump(bundle, out_dir / MODEL_BUNDLE_NAME)

    per_output = {
        "mae_mlp_holdout": dict(zip(ycols, mae_mlp)),
        "ycols": ycols,
    }

    report = {
        "n_scenarios": n,
        "holdout_fraction": test_fraction,
        "r2_continuous_mlp_holdout": r2_mlp_holdout,
        "r2_continuous_hgb_holdout": r2_hgb_holdout,
        "r2_continuous_knn_holdout": r2_knn_holdout,
        "motif_accuracy_holdout": acc_motif,
        "motif_classes": le.classes_.tolist(),
        "loocv_by_scenario": loocv_report,
        "baselines": baselines,
        "primary_regressor": primary,
        "per_output": per_output,
        "source_csv": str(optimal_csv.resolve()),
    }
    save_json(out_dir / "train_report.json", report)

    from sklearn.inspection import permutation_importance

    r = permutation_importance(
        reg_mlp, X_test, y_test_n, n_repeats=15, random_state=random_state, scoring="r2"
    )
    imp = dict(zip(SCENARIO_FEATURE_COLS, r.importances_mean.tolist()))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.barh(SCENARIO_FEATURE_COLS, [imp[c] for c in SCENARIO_FEATURE_COLS])
    ax.set_xlabel("Permutation importance (R²)")
    ax.set_title("Workload features (meta-MLP holdout)")
    plt.tight_layout()
    fig.savefig(out_dir / "permutation_importance_workload.png")
    plt.close(fig)

    print(f"Saved bundle to {out_dir / MODEL_BUNDLE_NAME}")
    print(json.dumps(report, indent=2))
    return out_dir


def workload_vector_from_dict(d: dict[str, Any]) -> np.ndarray:
    task = d["task"].lower().strip()
    if task == "iris":
        ts = float(d.get("test_size", 0.25))
        n_total = int(d.get("n_total", 150))
        n_te = int(round(n_total * ts))
        n_tr = n_total - n_te
        t_span = d.get("t_span_ns", [0.0, 20.0])
        hold = float(t_span[1] - t_span[0])
        te, bw = 0, 0
        enc = 0
    elif task in ("narma10", "narma"):
        task, enc = "narma10", 1
        n_tr = int(d["train_size"])
        n_te = int(d["test_size"])
        bw = int(d.get("washout", 200))
        hold = float(d.get("hold_ns", 20.0))
    elif task in ("memory_capacity", "memory"):
        task, enc = "memory_capacity", 2
        seq = int(d.get("sequence_length", 500))
        wo = int(d.get("washout", 120))
        n_tr, n_te, bw = max(0, seq - wo), 0, wo
        hold = float(d.get("hold_ns", 0.25))
    elif task == "mackey":
        enc = 3
        n_tr = int(d["train_size"])
        n_te = int(d["test_size"])
        bw = int(d.get("washout", 150))
        hold = float(d.get("hold_ns", 0.25))
    else:
        raise ValueError("task must be iris, narma10, memory_capacity, or mackey")

    row = pd.Series(
        {
            "task_encoding": float(enc),
            "bench_n_train": n_tr,
            "bench_n_test": n_te,
            "bench_washout_macros": bw,
            "bench_hold_ns": hold,
        }
    )
    row["log10_workload_tokens"] = _log10_workload(row)
    dt = float(d.get("dt_ns", 5e-4))
    fab = {"dt_ns": dt}
    sc = dict(d)
    sc["task"] = task
    row["log10_sim_budget"] = float(
        np.log10(max(1, _estimate_steps_quick(task, sc, fab)))
    )
    return np.array([[row[c] for c in SCENARIO_FEATURE_COLS]], dtype=np.float64)


def _estimate_steps_quick(task: str, sc: dict[str, Any], fab: dict[str, Any]) -> int:
    dt = float(fab.get("dt_ns", 5e-4))
    if task == "iris":
        ts = float(sc.get("test_size", 0.25))
        n_tr = int(150 * (1 - ts))
        tspan = sc.get("t_span_ns", (0.0, 20.0))
        steps = max(1, int(round((tspan[1] - tspan[0]) / dt)))
        return steps * n_tr
    if task == "narma10":
        T = int(sc["train_size"]) + int(sc["test_size"]) + int(sc.get("washout", 200))
        hold = float(sc.get("hold_ns", 20.0))
        return T * max(1, int(round(hold / dt)))
    if task == "memory_capacity":
        T = int(sc.get("sequence_length", 500))
        hold = float(sc.get("hold_ns", 0.25))
        return T * max(1, int(round(hold / dt)))
    if task == "mackey":
        T = int(sc["train_size"]) + int(sc["test_size"]) + int(sc.get("washout", 150))
        hold = float(sc.get("hold_ns", 0.25))
        return T * max(1, int(round(hold / dt)))
    return 1


def predict_from_bundle(bundle_path: Path, workload: dict[str, Any]) -> dict[str, Any]:
    bundle = joblib.load(bundle_path)
    sx: StandardScaler = bundle["scaler_X"]
    sy: StandardScaler = bundle["scaler_Y_cont"]
    le: LabelEncoder = bundle["label_encoder_motif"]
    ycols: list[str] = bundle["continuous_fab_cols"]
    primary = bundle.get("primary_regressor", "mlp")
    clf = bundle["mlp_motif"]
    single = bundle.get("single_motif_fallback", False)

    X = workload_vector_from_dict(workload)
    Xn = sx.transform(X)

    if primary == "hist_gbrt" and "hgb_continuous" in bundle:
        y_hat = bundle["hgb_continuous"].predict(Xn)[0]
    else:
        y_hat = sy.inverse_transform(bundle["mlp_continuous"].predict(Xn))[0]

    out: dict[str, Any] = {"workload": workload, "primary_regressor": primary}
    for i, c in enumerate(ycols):
        key = c.replace("optimal_", "recommended_")
        out[key] = float(y_hat[i])

    if "recommended_noise_float" in out:
        out["recommended_noise_on"] = bool(round(out["recommended_noise_float"]))

    if clf is None or single:
        out["recommended_motif"] = str(le.classes_[0])
    else:
        mid = int(clf.predict(Xn)[0])
        out["recommended_motif"] = str(le.inverse_transform([mid])[0])

    if "recommended_spacing_m" in out:
        out["recommended_spacing_um"] = float(out["recommended_spacing_m"]) * 1e6

    return out


def _cli_predict(args: argparse.Namespace) -> None:
    with open(args.workload_json, encoding="utf-8") as f:
        w = json.load(f)
    out = predict_from_bundle(Path(args.bundle).resolve(), w)
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fab recommender")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sa = sub.add_parser("aggregate")
    sa.add_argument("--long-csv", type=Path, required=True)
    sa.add_argument("--out", type=Path, default=None)
    sa.add_argument("--robust-lambda", type=float, default=0.0)
    sa.add_argument("--surrogate", action="store_true")
    sa.add_argument("--pareto", action="store_true")
    sa.add_argument("--pareto-front-csv", type=Path, default=None)

    def _agg(a):
        aggregate_optimal(
            Path(a.long_csv).resolve(),
            Path(a.out) if a.out else None,
            robust_penalty_lambda=a.robust_lambda,
            use_surrogate_optimum=a.surrogate,
            pareto_multi_objective=a.pareto,
            pareto_front_csv=Path(a.pareto_front_csv) if a.pareto_front_csv else None,
        )

    sa.set_defaults(func=_agg)

    st = sub.add_parser("train")
    st.add_argument("optimal_csv", type=Path)
    st.add_argument("--out-dir", type=Path, default=None)
    st.add_argument("--no-group-cv", action="store_true")
    st.add_argument("--no-baselines", action="store_true")
    st.set_defaults(
        func=lambda a: train_recommender(
            Path(a.optimal_csv).resolve(),
            Path(a.out_dir) if a.out_dir else None,
            use_group_cv=not a.no_group_cv,
            run_baselines=not a.no_baselines,
        )
    )

    sp = sub.add_parser("predict")
    sp.add_argument("--bundle", type=Path, required=True)
    sp.add_argument("--workload-json", type=Path, required=True)
    sp.set_defaults(func=_cli_predict)

    sp2 = sub.add_parser("pipeline")
    sp2.add_argument("--long-csv", type=Path, required=True)
    sp2.add_argument("--optimal-out", type=Path, default=None)
    sp2.add_argument("--train-out-dir", type=Path, default=None)
    sp2.add_argument("--robust-lambda", type=float, default=0.0)
    sp2.add_argument("--surrogate", action="store_true")
    sp2.add_argument("--pareto", action="store_true")

    def _pipe(a):
        o = aggregate_optimal(
            Path(a.long_csv).resolve(),
            Path(a.optimal_out) if a.optimal_out else None,
            robust_penalty_lambda=a.robust_lambda,
            use_surrogate_optimum=a.surrogate,
            pareto_multi_objective=a.pareto,
        )
        train_recommender(o, Path(a.train_out_dir) if a.train_out_dir else None)

    sp2.set_defaults(func=_pipe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
