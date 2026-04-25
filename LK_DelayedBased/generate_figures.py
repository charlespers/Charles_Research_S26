#!/usr/bin/env python3
"""
Generate all 9 paper figures for the GPU-accelerated photonic reservoir paper.

Figures produced in --output-dir (default: paper/figures/):
  fig1_speedup.{pdf,png}            GPU vs CPU speedup (grouped bar)
  fig2_accuracy_heatmap.{pdf,png}   Fab sweep accuracy landscape (motif × k)
  fig3_baseline_comparison.{pdf,png} Iris accuracy: baselines vs reservoir
  fig4_regressor_r2.{pdf,png}       MLP regressor LOOCV R² per fab parameter
  fig5_motif_confusion.{pdf,png}    Motif classifier confusion matrix
  fig6_feature_importance.{pdf,png} Workload-feature permutation importance
  fig7_trajectories.{pdf,png}       Laser intensity trajectories under 3 motifs
  fig8_ablation.{pdf,png}           Accuracy / time vs dt_ns and store_every
  fig9_tsne.{pdf,png}               t-SNE of 20D reservoir features

Usage
-----
  # After running benchmark_timing.py and mlp_classification_harness.py:
  python generate_figures.py

  # Custom paths:
  python generate_figures.py \\
      --timing-csv outputs/timing/timing_results.csv \\
      --sweep-csv pipeline_iris/outputs/sweeps/fab_long/.../sweep_long.csv \\
      --mlp-dir outputs/analysis/mlp_classification \\
      --output-dir paper/figures
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=UserWarning)

# ── Style constants (NeurIPS-friendly) ───────────────────────────────────────

FIGSIZE_HALF = (3.5, 2.8)
FIGSIZE_FULL = (7.0, 2.8)
FIGSIZE_WIDE = (7.0, 4.5)
FIGSIZE_SQ   = (3.5, 3.2)
DPI          = 300

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         DPI,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})


def _save(fig: plt.Figure, path_stem: Path) -> None:
    for ext in ("pdf", "png"):
        out = path_stem.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path_stem.name}.{{pdf,png}}")


# ── Figure 1: GPU vs CPU speedup ─────────────────────────────────────────────

def fig1_speedup(timing_csv: Path, out: Path) -> None:
    df = pd.read_csv(timing_csv)
    configs = df["config"].unique().tolist()
    batch_sizes = sorted(df["batch_size"].unique())

    fig, axes = plt.subplots(1, len(configs), figsize=(7.0, 2.6), sharey=False)
    if len(configs) == 1:
        axes = [axes]

    for ax, cfg in zip(axes, configs):
        sub = df[df["config"] == cfg].sort_values("batch_size")
        xs = np.arange(len(batch_sizes))
        w = 0.28

        ax.bar(xs - w, sub["t_cpu"].values,        width=w, label="CPU",         color=COLORS[0], alpha=0.85)
        ax.bar(xs,     sub["t_gpu_warmup"].values,  width=w, label="GPU (warmup)", color=COLORS[1], alpha=0.85)
        ax.bar(xs + w, sub["t_gpu_cached"].values,  width=w, label="GPU (cached)", color=COLORS[2], alpha=0.85)

        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(i + w, row["t_gpu_cached"] + 0.01,
                    f"{row['speedup']:.0f}×", ha="center", va="bottom",
                    fontsize=7, color=COLORS[2], fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels(batch_sizes)
        ax.set_xlabel("Batch size (samples)")
        ax.set_ylabel("Wall time (s)" if ax is axes[0] else "")
        cfg_label = cfg.replace("_", " ")
        ax.set_title(cfg_label, fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3,
               bbox_to_anchor=(1.0, 1.05), fontsize=8)
    fig.suptitle("CPU vs GPU Simulation Time", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, out)


# ── Figure 2: Accuracy landscape heatmap ────────────────────────────────────

def fig2_accuracy_heatmap(sweep_csv: Path, out: Path) -> None:
    df = pd.read_csv(sweep_csv)
    metric_col = "metric_mean" if "metric_mean" in df.columns else "metric"

    motifs = sorted(df["motif"].unique())
    ks = sorted(df["default_k"].unique())

    # Build heatmap matrix (avg over other params)
    mat = np.full((len(motifs), len(ks)), np.nan)
    for i, m in enumerate(motifs):
        for j, k in enumerate(ks):
            vals = df[(df["motif"] == m) & (df["default_k"] == k)][metric_col].dropna()
            if len(vals):
                mat[i, j] = vals.mean()

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    cmap = LinearSegmentedColormap.from_list("acc", ["#E3F2FD", "#1565C0"])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.85, vmax=1.0)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"{k:.0f}" for k in ks])
    ax.set_yticks(range(len(motifs)))
    ax.set_yticklabels(motifs)
    ax.set_xlabel("Coupling strength $k$")
    ax.set_ylabel("Coupling motif")
    ax.set_title("Iris accuracy landscape (fab sweep)")
    fig.colorbar(im, ax=ax, label="Accuracy")

    # Annotate cells with values
    for i in range(len(motifs)):
        for j in range(len(ks)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if mat[i, j] > 0.93 else "black")

    fig.tight_layout()
    _save(fig, out)


# ── Figure 3: Baseline comparison ────────────────────────────────────────────

def fig3_baseline_comparison(sweep_csv: Path | None, out: Path) -> None:
    from sklearn.datasets import load_iris
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from model7_reservoir import build_reservoir_dataset, minmax_fit, minmax_transform

    try:
        import jax as _jax
        _jax.config.update("jax_enable_x64", True)
        from model7_reservoir_gpu import build_gpu_sim_params, build_reservoir_dataset_gpu
        _use_gpu = True
    except ImportError:
        _use_gpu = False

    iris = load_iris()
    X, y = iris.data.astype(float), iris.target.astype(int)

    # Best fab params from sweep (or use defaults if no sweep available)
    if sweep_csv is not None and sweep_csv.exists():
        df_sw = pd.read_csv(sweep_csv)
        mc = "metric_mean" if "metric_mean" in df_sw.columns else "metric"
        best = df_sw.loc[df_sw[mc].idxmax()]
        best_motif   = str(best["motif"])
        best_k       = float(best["default_k"])
        best_i_min   = float(best["i_min"])
        best_i_max   = float(best["i_max"])
        best_spacing = float(best["spacing_m"])
        best_noise   = bool(best["noise_on"])
    else:
        best_motif, best_k, best_i_min, best_i_max = "auxiliary", 15.0, 0.5, 1.5
        best_spacing, best_noise = 50e-6, False

    if _use_gpu:
        gpu_params = build_gpu_sim_params(
            motif=best_motif, spacing_m=best_spacing, lambda0_m=850e-9, n_air=1.0,
            I_th_A=17.35e-3, i_min=best_i_min, i_max=best_i_max,
            dt_ns=5e-4, t_span_ns=(0.0, 20.0), washout_ns=10.0,
            store_every=400, noise_on=best_noise, default_k=best_k,
        )
    else:
        print("  (JAX not available — using CPU reservoir for baseline comparison)")
        gpu_params = None

    cpu_kw = dict(
        motif=best_motif, spacing_m=best_spacing, lambda0_m=850e-9, n_air=1.0,
        I_th_A=17.35e-3, i_min=best_i_min, i_max=best_i_max,
        dt_ns=5e-4, t_span_ns=(0.0, 20.0), washout_ns=10.0,
        store_every=400, noise_on=best_noise, default_k=best_k,
    )

    # On CPU use reduced iterations to keep runtime under ~10 min;
    # on GPU (Adroit) use full 5×5 for publication-quality error bars.
    if _use_gpu:
        N_REPS, N_FOLDS = 5, 5
    else:
        N_REPS, N_FOLDS = 1, 3

    models = {
        "Ridge":     Pipeline([("sc", StandardScaler()), ("m", RidgeClassifier())]),
        "MLP":       Pipeline([("sc", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))]),
        "RF":        Pipeline([("sc", StandardScaler()), ("m", RandomForestClassifier(n_estimators=100, random_state=42))]),
        "GBT":       Pipeline([("sc", StandardScaler()), ("m", GradientBoostingClassifier(n_estimators=100, random_state=42))]),
    }

    results: dict[str, list[float]] = {k: [] for k in list(models.keys()) + ["Reservoir"]}

    print(f"  Running baseline experiments ({N_REPS}×{N_FOLDS}-fold, {'GPU' if _use_gpu else 'CPU'})...")
    for rep in range(N_REPS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
        for tr, te in skf.split(X, y):
            xmin, span = minmax_fit(X[tr])
            X_tr01 = minmax_transform(X[tr], xmin, span)
            X_te01 = minmax_transform(X[te], xmin, span)

            for name, pipe in models.items():
                pipe.fit(X[tr], y[tr])
                results[name].append(accuracy_score(y[te], pipe.predict(X[te])))

            # Reservoir — GPU if available, else CPU
            if _use_gpu:
                F_tr, _ = build_reservoir_dataset_gpu(X_tr01, y[tr], gpu_params=gpu_params, base_seed=30 + rep)
                F_te, _ = build_reservoir_dataset_gpu(X_te01, y[te], gpu_params=gpu_params, base_seed=10030 + rep)
            else:
                F_tr, _ = build_reservoir_dataset(X_tr01, y[tr], base_seed=30 + rep, **cpu_kw)
                F_te, _ = build_reservoir_dataset(X_te01, y[te], base_seed=10030 + rep, **cpu_kw)
            clf_r = Pipeline([("sc", StandardScaler()), ("m", RidgeClassifier())])
            clf_r.fit(F_tr, y[tr])
            results["Reservoir"].append(accuracy_score(y[te], clf_r.predict(F_te)))

    names = list(results.keys())
    means = [np.mean(v) for v in results.values()]
    stds  = [np.std(v) / np.sqrt(len(v)) for v in results.values()]  # SEM
    colors = [COLORS[3] if n == "Reservoir" else COLORS[0] for n in names]

    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    xs = np.arange(len(names))
    bars = ax.bar(xs, means, yerr=stds, capsize=4, color=colors, alpha=0.85, width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.85, 1.01)
    ax.set_title("Iris classification: baselines vs reservoir")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _save(fig, out)


# ── Figure 4: MLP regressor LOOCV R² ─────────────────────────────────────────

def fig4_regressor_r2(mlp_dir: Path, out: Path) -> None:
    csv = mlp_dir / "regressor_cv_metrics.csv"
    df = pd.read_csv(csv)

    params = df["param_name"].unique().tolist()
    models = df["model"].unique().tolist()
    short = {
        "optimal_default_k":    "k",
        "optimal_i_min":        "i_min",
        "optimal_i_max":        "i_max",
        "optimal_spacing_m":    "spacing",
        "optimal_noise_float":  "noise",
        "optimal_I_th_A":       "I_th",
        "optimal_lambda0_m":    "λ₀",
        "optimal_dt_ns":        "dt",
        "optimal_store_every":  "store",
    }

    CLIP_MIN = -1.0  # values below this are shown clipped with a marker

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    n = len(params)
    w = 0.25
    xs = np.arange(n)
    for mi, (model, color) in enumerate(zip(models, COLORS)):
        sub = df[df["model"] == model].set_index("param_name")
        r2s = [float(sub.loc[p, "r2_mean"]) if p in sub.index else 0.0 for p in params]
        r2s_clipped = [max(v, CLIP_MIN) for v in r2s]
        bars = ax.barh(xs + mi * w - w, r2s_clipped, height=w,
                       label=model, color=color, alpha=0.85)
        # Mark bars that were clipped (actual value << -1)
        for xi, (raw, clipped) in enumerate(zip(r2s, r2s_clipped)):
            if raw < CLIP_MIN - 0.05:
                ax.text(CLIP_MIN - 0.02, xs[xi] + mi * w - w,
                        "◀", ha="right", va="center", fontsize=6, color=color)

    ax.set_yticks(xs)
    ax.set_yticklabels([short.get(p, p) for p in params])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LOOCV R²  (◀ = off-scale)")
    ax.set_title("MLP regressor: fab-parameter prediction accuracy (LOOCV)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(CLIP_MIN, 1.05)
    fig.tight_layout()
    _save(fig, out)


# ── Figure 5: Motif classifier confusion matrix ───────────────────────────────

def fig5_motif_confusion(mlp_dir: Path, out: Path) -> None:
    cm_path     = mlp_dir / "classifier_confusion_matrix.npy"
    labels_path = mlp_dir / "motif_label_classes.npy"
    cm     = np.load(cm_path, allow_pickle=True).astype(float)
    labels = np.load(labels_path, allow_pickle=True).tolist()

    # Normalise per true class
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    cmap = LinearSegmentedColormap.from_list("conf", ["white", "#1565C0"])
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted motif")
    ax.set_ylabel("True motif")
    ax.set_title("Motif classifier confusion matrix (LOOCV)")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm_norm[i, j] > 0.6 else "black")

    fig.colorbar(im, ax=ax, label="Fraction")
    fig.tight_layout()
    _save(fig, out)


# ── Figure 6: Feature importance ─────────────────────────────────────────────

def fig6_feature_importance(mlp_dir: Path, out: Path) -> None:
    import joblib
    bundle = joblib.load(mlp_dir / "mlp_bundle.joblib")
    means  = np.array(bundle["permutation_importance_means"])
    stds   = np.array(bundle["permutation_importance_stds"])
    feat   = bundle["feature_cols"]

    short = {
        "task_encoding":        "Task type",
        "bench_n_train":        "N train",
        "bench_n_test":         "N test",
        "bench_washout_macros": "Washout",
        "bench_hold_ns":        "Hold (ns)",
        "log10_workload_tokens":"log₁₀ tokens",
        "log10_sim_budget":     "log₁₀ budget",
    }

    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    ys = np.arange(len(feat))
    ax.barh(ys, means[order], xerr=stds[order], capsize=4,
            color=COLORS[0], alpha=0.85, height=0.55)
    ax.set_yticks(ys)
    ax.set_yticklabels([short.get(feat[i], feat[i]) for i in order])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Permutation importance (motif classifier)")
    ax.set_title("Workload feature importance")
    fig.tight_layout()
    _save(fig, out)


# ── Figure 7: Laser intensity trajectories ────────────────────────────────────

def fig7_trajectories(out: Path) -> None:
    from sklearn.datasets import load_iris
    from model7_reservoir import (
        build_inline_coupling_matrix_tunable,
        build_inline_geometry_matrices,
        default_physics_params_ops,
        features_to_pump_ratios,
        minmax_fit,
        minmax_transform,
        simulate_model7_network_auto,
    )

    iris = load_iris()
    X = iris.data.astype(float)
    xmin, span = minmax_fit(X)
    X01 = minmax_transform(X, xmin, span)
    sample = X01[0]  # setosa representative

    motifs_to_show = ["auxiliary", "chain", "relay"]
    num_lasers = 4
    dt_ns = 5e-4
    t_span_ns = (0.0, 20.0)
    washout_ns = 10.0
    store_every = 20  # finer resolution for trajectory plot

    pump_ratios = features_to_pump_ratios(sample, rmin=0.5, rmax=1.5)
    e = 1.602e-19
    I_th_A = 17.35e-3
    P_ops = (pump_ratios * I_th_A / e) * 1e-9
    params = default_physics_params_ops(P_ops)

    y0 = np.zeros((num_lasers, 3))
    y0[:, 0] = 1e-3

    fig, axes = plt.subplots(
        2, len(motifs_to_show), figsize=(7.0, 4.0),
        sharex=True,
    )

    for col, motif in enumerate(motifs_to_show):
        K = build_inline_coupling_matrix_tunable(num_lasers, motif, default_k=15.0)
        _, tau_ns, _, cos_phi, sin_phi = build_inline_geometry_matrices(
            num_lasers, 50e-6, 850e-9, 1.0
        )
        t, x, y, N, _ = simulate_model7_network_auto(
            t_span_ns=t_span_ns, dt_ns=dt_ns, y0=y0,
            params=params, K=K, tau_ns=tau_ns,
            cos_phi=cos_phi, sin_phi=sin_phi,
            noise_on=True, seed=col, store_every=store_every, P_step=None,
        )
        S = (x ** 2 + y ** 2)  # photon intensity per laser

        ax_s = axes[0, col]
        ax_n = axes[1, col]

        for laser_idx in range(num_lasers):
            ax_s.plot(t, S[:, laser_idx],
                      color=COLORS[laser_idx], linewidth=0.8, alpha=0.85,
                      label=f"L{laser_idx+1}")
            ax_n.plot(t, N[:, laser_idx] / 1e8,
                      color=COLORS[laser_idx], linewidth=0.8, alpha=0.85)

        ax_s.axvline(washout_ns, color="gray", linestyle=":", linewidth=0.8)
        ax_n.axvline(washout_ns, color="gray", linestyle=":", linewidth=0.8)
        ax_s.set_title(f"{motif}", fontsize=9)
        # Cap y-axis at 2× post-washout median max to suppress initial transient spike
        S_post = S[t >= washout_ns]
        if S_post.size > 0:
            cap = max(float(np.percentile(S_post, 99)) * 3, 1e-6)
            ax_s.set_ylim(bottom=0, top=cap)
        if col == 0:
            ax_s.set_ylabel("Intensity $S_i$")
            ax_n.set_ylabel("Carriers $N_i$ (×10⁸)")
        ax_n.set_xlabel("Time (ns)")

    axes[0, -1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Photon intensity and carrier dynamics (Iris sample 0)", fontsize=9)
    fig.tight_layout()
    _save(fig, out)


# ── Figure 8: Ablation: dt_ns and store_every ────────────────────────────────

def fig8_ablation(out: Path) -> None:
    import time as _time
    from sklearn.datasets import load_iris
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from model7_reservoir import build_reservoir_dataset, minmax_fit, minmax_transform

    try:
        import jax as _jax
        _jax.config.update("jax_enable_x64", True)
        from model7_reservoir_gpu import build_gpu_sim_params, build_reservoir_dataset_gpu
        _use_gpu = True
    except ImportError:
        _use_gpu = False
        print("  (JAX not available — using CPU for ablation; will be slow)")

    iris = load_iris()
    X = iris.data.astype(float)
    y = iris.target.astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    xmin, span = minmax_fit(X_tr)
    X_tr01 = minmax_transform(X_tr, xmin, span)
    X_te01 = minmax_transform(X_te, xmin, span)

    base_cfg = dict(
        motif="auxiliary", spacing_m=50e-6, lambda0_m=850e-9, n_air=1.0,
        I_th_A=17.35e-3, i_min=0.5, i_max=1.5, washout_ns=10.0,
        noise_on=False, default_k=15.0,
    )

    def run_one(dt_ns, store_every_val):
        kw = dict(dt_ns=dt_ns, t_span_ns=(0.0, 20.0), store_every=store_every_val, **base_cfg)
        t0 = _time.perf_counter()
        if _use_gpu:
            gp = build_gpu_sim_params(**kw)
            F_tr, _ = build_reservoir_dataset_gpu(X_tr01, y_tr, gpu_params=gp, base_seed=30)
            F_te, _ = build_reservoir_dataset_gpu(X_te01, y_te, gpu_params=gp, base_seed=10030)
        else:
            F_tr, _ = build_reservoir_dataset(X_tr01, y_tr, base_seed=30, **kw)
            F_te, _ = build_reservoir_dataset(X_te01, y_te, base_seed=10030, **kw)
        elapsed = _time.perf_counter() - t0
        F_tr = np.asarray(F_tr)
        F_te = np.asarray(F_te)
        if np.any(~np.isfinite(F_tr)) or np.any(~np.isfinite(F_te)):
            return float("nan"), elapsed  # numerically unstable config
        clf = Pipeline([("sc", StandardScaler()), ("m", RidgeClassifier())])
        clf.fit(F_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(F_te))
        return float(acc), elapsed

    # Fine dt steps multiply simulation time; on CPU cap at 2e-4 to stay under ~10 min.
    if _use_gpu:
        dt_vals = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
    else:
        dt_vals = [1e-3, 5e-4, 2e-4]
    store_vals = [100, 200, 400, 800, 1600]

    print("  Ablation: sweeping dt_ns ...", flush=True)
    accs_dt, times_dt = [], []
    for dt in dt_vals:
        acc, t = run_one(dt, 400)
        accs_dt.append(acc)
        times_dt.append(t)
        tag = "unstable" if np.isnan(acc) else f"acc={acc:.3f}"
        print(f"    dt={dt:.0e}  {tag}  t={t:.2f}s", flush=True)

    print("  Ablation: sweeping store_every ...", flush=True)
    accs_se, times_se = [], []
    for se in store_vals:
        acc, t = run_one(5e-4, se)
        accs_se.append(acc)
        times_se.append(t)
        print(f"    store_every={se}  acc={acc:.3f}  t={t:.2f}s", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_WIDE)

    ax = axes[0, 0]
    # Filter NaN (numerically unstable) points; show only stable configs
    dt_arr   = np.array(dt_vals)
    accs_arr = np.array(accs_dt, dtype=float)
    stable   = np.isfinite(accs_arr)
    ax.plot(dt_arr[stable], accs_arr[stable], "o-", color=COLORS[0], markersize=5)
    if (~stable).any():
        ax.scatter(dt_arr[~stable], np.zeros(np.sum(~stable)),
                   marker="x", color="red", s=40, zorder=5, label="unstable")
        ax.legend(fontsize=7)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Time step dt (ns)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs time resolution")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    ax = axes[0, 1]
    ax.plot(dt_vals, times_dt, "s-", color=COLORS[1], markersize=5)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Time step dt (ns)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Simulation time vs time resolution")

    ax = axes[1, 0]
    ax.plot(store_vals, accs_se, "o-", color=COLORS[0], markersize=5)
    ax.set_xlabel("store_every (decimation)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs decimation rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    ax = axes[1, 1]
    ax.plot(store_vals, times_se, "s-", color=COLORS[1], markersize=5)
    ax.set_xlabel("store_every (decimation)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Simulation time vs decimation rate")

    fig.suptitle("Ablation: effect of simulation resolution parameters", fontsize=9)
    fig.tight_layout()
    _save(fig, out)


# ── Figure 9: t-SNE of reservoir state space ─────────────────────────────────

def fig9_tsne(sweep_csv: Path | None, out: Path) -> None:
    from sklearn.datasets import load_iris
    from sklearn.manifold import TSNE
    from model7_reservoir import build_reservoir_dataset, minmax_fit, minmax_transform

    try:
        import jax as _jax
        _jax.config.update("jax_enable_x64", True)
        from model7_reservoir_gpu import build_gpu_sim_params, build_reservoir_dataset_gpu
        _use_gpu = True
    except ImportError:
        _use_gpu = False
        print("  (JAX not available — using CPU reservoir for t-SNE; will be slow)")

    iris = load_iris()
    X = iris.data.astype(float)
    y = iris.target.astype(int)
    xmin, span = minmax_fit(X)
    X01 = minmax_transform(X, xmin, span)

    # Best params from sweep or default
    best_kw = dict(
        motif="auxiliary", spacing_m=50e-6, lambda0_m=850e-9, n_air=1.0,
        I_th_A=17.35e-3, i_min=0.5, i_max=1.5,
        dt_ns=5e-4, t_span_ns=(0.0, 20.0), washout_ns=10.0,
        store_every=400, noise_on=False, default_k=15.0,
    )
    if sweep_csv is not None and sweep_csv.exists():
        df_sw = pd.read_csv(sweep_csv)
        mc = "metric_mean" if "metric_mean" in df_sw.columns else "metric"
        best = df_sw.loc[df_sw[mc].idxmax()]
        best_kw = dict(
            motif=str(best["motif"]), spacing_m=float(best["spacing_m"]),
            lambda0_m=850e-9, n_air=1.0, I_th_A=float(best["I_th_A"]),
            i_min=float(best["i_min"]), i_max=float(best["i_max"]),
            dt_ns=5e-4, t_span_ns=(0.0, 20.0), washout_ns=10.0,
            store_every=400, noise_on=bool(best["noise_on"]),
            default_k=float(best["default_k"]),
        )

    print("  Computing reservoir features for all 150 Iris samples...", flush=True)
    if _use_gpu:
        gp = build_gpu_sim_params(**best_kw)
        F, _ = build_reservoir_dataset_gpu(X01, y, gpu_params=gp, base_seed=30)
        F = np.array(F)
    else:
        F, _ = build_reservoir_dataset(X01, y, base_seed=30, **best_kw)

    print("  Running t-SNE...", flush=True)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    Z = tsne.fit_transform(F)

    class_names = iris.target_names.tolist()
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    for cls_idx, (cls_name, color) in enumerate(zip(class_names, COLORS[:3])):
        mask = y == cls_idx
        ax.scatter(Z[mask, 0], Z[mask, 1], c=color, label=cls_name,
                   s=20, alpha=0.85, edgecolors="none")

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title("t-SNE: photonic reservoir features (all 150 Iris samples)")
    ax.legend(fontsize=8, markerscale=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    _save(fig, out)


# ── orchestrator ──────────────────────────────────────────────────────────────

def main(
    timing_csv: Path,
    sweep_csv: Path | None,
    mlp_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("fig1_speedup",           lambda: fig1_speedup(timing_csv, output_dir / "fig1_speedup"),
         timing_csv.exists()),
        ("fig2_accuracy_heatmap",  lambda: fig2_accuracy_heatmap(sweep_csv, output_dir / "fig2_accuracy_heatmap"),
         sweep_csv is not None and sweep_csv.exists()),
        ("fig3_baseline_comparison", lambda: fig3_baseline_comparison(sweep_csv, output_dir / "fig3_baseline_comparison"),
         True),
        ("fig4_regressor_r2",      lambda: fig4_regressor_r2(mlp_dir, output_dir / "fig4_regressor_r2"),
         (mlp_dir / "regressor_cv_metrics.csv").exists()),
        ("fig5_motif_confusion",   lambda: fig5_motif_confusion(mlp_dir, output_dir / "fig5_motif_confusion"),
         (mlp_dir / "classifier_confusion_matrix.npy").exists()),
        ("fig6_feature_importance", lambda: fig6_feature_importance(mlp_dir, output_dir / "fig6_feature_importance"),
         (mlp_dir / "mlp_bundle.joblib").exists()),
        ("fig7_trajectories",      lambda: fig7_trajectories(output_dir / "fig7_trajectories"),
         True),
        ("fig8_ablation",          lambda: fig8_ablation(output_dir / "fig8_ablation"),
         True),
        ("fig9_tsne",              lambda: fig9_tsne(sweep_csv, output_dir / "fig9_tsne"),
         True),
    ]

    for name, fn, available in tasks:
        if not available:
            print(f"[SKIP] {name} — required data not found")
            continue
        print(f"[GEN]  {name}")
        try:
            fn()
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Figures in {output_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate all 9 paper figures for the photonic reservoir paper"
    )
    ap.add_argument("--timing-csv",  type=Path, default=Path("outputs/timing/timing_results.csv"))
    ap.add_argument("--sweep-csv",   type=Path, default=None,
                    help="Path to sweep_long.csv from a fab sweep run")
    ap.add_argument("--mlp-dir",     type=Path, default=Path("outputs/analysis/mlp_classification"))
    ap.add_argument("--output-dir",  type=Path, default=Path("paper/figures"))
    ap.add_argument("--auto-sweep",  action="store_true",
                    help="Auto-discover most recent sweep_long.csv under pipeline_iris/outputs/")
    args = ap.parse_args()

    # Auto-discover sweep CSV
    if args.sweep_csv is None or not args.sweep_csv.exists():
        candidates = sorted(
            Path("pipeline_iris/outputs/sweeps").glob("fab_long*/*/sweep_long.csv"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        if candidates:
            args.sweep_csv = candidates[0]
            print(f"Auto-detected sweep CSV: {args.sweep_csv}")

    main(args.timing_csv, args.sweep_csv, args.mlp_dir, args.output_dir)
