#!/usr/bin/env python3
"""
MLP classification harness: workload descriptors → fab params (regressor)
and workload descriptors → best coupling motif (classifier).

Scoped to classification benchmarks (Iris, task_encoding == 0) only.

Inputs
------
--optimal-csv   Path to an optimal-per-scenario CSV (one row per workload
                scenario, each row has the best fab config for that scenario).
                Defaults to fixtures/optimal_per_scenario_synthetic.csv.

--sweep-csv     If provided, aggregate_optimal() is called first to extract
                the best config per scenario, then training proceeds.

Outputs (all in --output-dir)
------------------------------
regressor_cv_metrics.csv    LOOCV R² and MAE per fab parameter per model
classifier_cv_metrics.csv   LOOCV accuracy per model
classifier_confusion_matrix.npy  (n_classes, n_classes) confusion matrix for MLP
motif_label_classes.npy     String class names in LabelEncoder order
mlp_bundle.joblib           Final models + scaler + label encoder
mlp_summary.json            High-level performance summary

Usage
-----
  python mlp_classification_harness.py
  python mlp_classification_harness.py \\
      --optimal-csv fixtures/optimal_per_scenario_synthetic.csv
  python mlp_classification_harness.py \\
      --sweep-csv pipeline_iris/outputs/sweeps/fab_long/.../sweep_long.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fab_meta_model import OPTIMAL_COL_ORDER, SCENARIO_FEATURE_COLS, _log10_workload
from output_io import save_json

try:
    from fab_meta_model import aggregate_optimal
except ImportError:
    aggregate_optimal = None  # handled below

DEFAULT_OPTIMAL_CSV = Path("fixtures/optimal_per_scenario_synthetic.csv")
DEFAULT_OUTPUT_DIR  = Path("outputs/analysis/mlp_classification")


# ── data loading ──────────────────────────────────────────────────────────────

def _load_and_prepare(optimal_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(optimal_csv)
    if "log10_workload_tokens" not in df.columns:
        df["log10_workload_tokens"] = df.apply(_log10_workload, axis=1)
    if "log10_sim_budget" not in df.columns:
        df["log10_sim_budget"] = 0.0
    df = df[df["task_encoding"] == 0].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows with task_encoding==0 (classification) found.")
    return df


# ── LOOCV helpers ─────────────────────────────────────────────────────────────

def _loocv_regressor(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    make_model,
) -> tuple[list[float], list[float]]:
    """LOOCV grouped by scenario. Returns per-column (r2, mae) lists."""
    logo = LeaveOneGroupOut()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []

    for tr_idx, te_idx in logo.split(X, Y, groups):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        m = make_model()
        m.fit(X_tr, Y[tr_idx])
        preds.append(m.predict(X_te))
        trues.append(Y[te_idx])

    y_true = np.vstack(trues)
    y_pred = np.vstack(preds)
    r2s  = [float(r2_score(y_true[:, i], y_pred[:, i]))         for i in range(y_true.shape[1])]
    maes = [float(mean_absolute_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    return r2s, maes


def _loocv_classifier(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    make_model,
    n_classes: int,
) -> tuple[float, np.ndarray]:
    """LOOCV grouped by scenario. Returns (accuracy, confusion_matrix)."""
    logo = LeaveOneGroupOut()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []

    for tr_idx, te_idx in logo.split(X, y, groups):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        m = make_model()
        m.fit(X_tr, y[tr_idx])
        preds.append(m.predict(X_te))
        trues.append(y[te_idx])

    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    acc = float(accuracy_score(y_true, y_pred))
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return acc, cm


# ── main training routine ─────────────────────────────────────────────────────

def run(optimal_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_and_prepare(optimal_csv)
    print(f"Loaded {len(df)} classification scenarios from {optimal_csv}")

    if len(df) < 4:
        raise ValueError(
            f"LOOCV requires at least 4 scenarios; got {len(df)}. "
            "Pass a richer --optimal-csv or run the sweep first."
        )

    X      = df[SCENARIO_FEATURE_COLS].values.astype(np.float64)
    ycols  = [c for c in OPTIMAL_COL_ORDER if c in df.columns]
    Y_reg  = df[ycols].values.astype(np.float64)
    groups = df["scenario_tag"].values

    le      = LabelEncoder()
    y_mot   = le.fit_transform(df["optimal_motif"].astype(str))
    n_cls   = len(le.classes_)
    motifs  = list(le.classes_)
    knn_k   = max(1, min(3, len(df) // 3))

    print(f"Regressor targets : {ycols}")
    print(f"Motif classes ({n_cls}): {motifs}")
    print(f"LOOCV folds       : {len(df)}")
    print()

    # ── Regressor LOOCV ───────────────────────────────────────────────────
    reg_factories = {
        "MLP": lambda: MultiOutputRegressor(
            MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=1000,
                random_state=42, alpha=1e-4, learning_rate_init=1e-3,
            )
        ),
        "HistGBT": lambda: MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_depth=4, max_iter=100, random_state=42, learning_rate=0.08,
            )
        ),
        "kNN": lambda: KNeighborsRegressor(n_neighbors=knn_k),
    }

    reg_rows: list[dict] = []
    for name, factory in reg_factories.items():
        print(f"  Regressor [{name:7s}] ...", end="  ", flush=True)
        r2s, maes = _loocv_regressor(X, Y_reg, groups, factory)
        for i, col in enumerate(ycols):
            reg_rows.append({
                "model":     name,
                "param_name": col,
                "r2_mean":   round(r2s[i],  4),
                "r2_std":    0.0,
                "mae_mean":  round(maes[i], 8),
                "mae_std":   0.0,
            })
        mean_r2 = float(np.mean(r2s))
        print(f"mean R²={mean_r2:+.3f}")

    df_reg = pd.DataFrame(reg_rows)
    df_reg.to_csv(output_dir / "regressor_cv_metrics.csv", index=False)
    print(f"  → regressor_cv_metrics.csv\n")

    # ── Classifier LOOCV ──────────────────────────────────────────────────
    clf_factories = {
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=1000,
            random_state=42, alpha=1e-4,
        ),
        "HistGBT": lambda: HistGradientBoostingClassifier(
            max_depth=4, max_iter=100, random_state=42, learning_rate=0.08,
        ),
        "kNN": lambda: KNeighborsClassifier(n_neighbors=knn_k),
    }

    clf_rows: list[dict] = []
    cm_mlp: np.ndarray | None = None
    for name, factory in clf_factories.items():
        print(f"  Classifier [{name:7s}] ...", end="  ", flush=True)
        acc, cm = _loocv_classifier(X, y_mot, groups, factory, n_cls)
        clf_rows.append({"model": name, "accuracy_mean": round(acc, 4), "accuracy_std": 0.0})
        print(f"accuracy={acc:.3f}")
        if name == "MLP":
            cm_mlp = cm

    df_clf = pd.DataFrame(clf_rows)
    df_clf.to_csv(output_dir / "classifier_cv_metrics.csv", index=False)
    np.save(output_dir / "classifier_confusion_matrix.npy", cm_mlp)
    np.save(output_dir / "motif_label_classes.npy", np.array(motifs))
    print(f"  → classifier_cv_metrics.csv + confusion_matrix.npy\n")

    # ── Final models trained on all data ──────────────────────────────────
    sc_final = StandardScaler()
    X_sc = sc_final.fit_transform(X)

    reg_final = MultiOutputRegressor(
        MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=1000,
            random_state=42, alpha=1e-4, learning_rate_init=1e-3,
        )
    )
    reg_final.fit(X_sc, Y_reg)

    clf_final = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, alpha=1e-4
    )
    clf_final.fit(X_sc, y_mot)

    # Permutation importance for Figure 6
    pi = permutation_importance(
        clf_final, X_sc, y_mot, n_repeats=30, random_state=42, scoring="accuracy"
    )
    pi_means = pi.importances_mean.tolist()
    pi_stds  = pi.importances_std.tolist()

    bundle = {
        "regressor":         reg_final,
        "classifier":        clf_final,
        "scaler":            sc_final,
        "label_encoder":     le,
        "feature_cols":      SCENARIO_FEATURE_COLS,
        "target_cols":       ycols,
        "motif_classes":     motifs,
        "permutation_importance_means": pi_means,
        "permutation_importance_stds":  pi_stds,
    }
    joblib.dump(bundle, output_dir / "mlp_bundle.joblib")
    print(f"  → mlp_bundle.joblib")

    # ── summary ───────────────────────────────────────────────────────────
    mlp_r2  = float(df_reg[df_reg["model"] == "MLP"]["r2_mean"].mean())
    mlp_acc = float(df_clf[df_clf["model"] == "MLP"]["accuracy_mean"].iloc[0])
    save_json(
        output_dir / "mlp_summary.json",
        {
            "n_scenarios":              len(df),
            "n_motif_classes":          n_cls,
            "motif_classes":            motifs,
            "mlp_regressor_mean_r2":    round(mlp_r2, 4),
            "mlp_classifier_accuracy":  round(mlp_acc, 4),
            "feature_cols":             SCENARIO_FEATURE_COLS,
            "target_cols":              ycols,
        },
    )
    print(f"  → mlp_summary.json\n")

    print("Results:")
    print(f"  MLP regressor  mean R² : {mlp_r2:+.3f}")
    print(f"  MLP classifier accuracy: {mlp_acc:.3f}")
    print(f"\nAll outputs in {output_dir}/")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MLP harness for fab-metadata prediction on classification benchmarks"
    )
    ap.add_argument(
        "--optimal-csv", type=Path, default=DEFAULT_OPTIMAL_CSV,
        help="Path to optimal_per_scenario CSV (one best-config row per scenario).",
    )
    ap.add_argument(
        "--sweep-csv", type=Path, default=None,
        help="If given, run aggregate_optimal() on this sweep CSV first.",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for all output artefacts.",
    )
    args = ap.parse_args()

    if args.sweep_csv is not None:
        if aggregate_optimal is None:
            raise ImportError("fab_meta_model.aggregate_optimal not available")
        tmp_optimal = args.output_dir / "optimal_per_scenario.csv"
        tmp_optimal.parent.mkdir(parents=True, exist_ok=True)
        args.optimal_csv = aggregate_optimal(args.sweep_csv, tmp_optimal)

    run(args.optimal_csv, args.output_dir)
