
"""
Generador de PDF con gráficas para SuperCon / Ontología 1
----------------------------------------------------------
- Carga datos (Polars o Pandas) desde un archivo agregado (Vista A) o un CSV/Parquet.
- Genera PDF multi-página con:
  * Distribución de Tc (histograma + boxplot + violin)
  * Dispersión (scatter) y regresión lineal simple contra features clave
  * Importancia de características (si hay tabla de importancias o se entrena un RF rápido)

Requisitos mínimos:
- matplotlib, numpy
Opcionales:
- polars o pandas
- scikit-learn (si se quiere calcular importancias cuando no hay archivo de importancias)

Uso rápido (ejemplo):
---------------------
python supercon_report_pdf.py \
  --data path_a_df_A.parquet \
  --out  reporte_supercon_figs.pdf \
  --feat-importances-rf path_a_feat_imp_rf.csv \
  --feat-importances-xgb path_a_feat_imp_xgb.csv

Notas:
- Si no pasas tablas de importancias, el script intentará calcular importancias con RandomForestRegressor
  sobre columnas que empiecen por 'frac_' o 'mag_' (requiere scikit-learn).
- No se usa seaborn y no se establecen colores explícitos (cumpliendo la restricción).
"""

import argparse
import os
import math
import numpy as np

# Intento de importación: Polars -> Pandas
try:
    import polars as pl
    _HAS_POLARS = True
except Exception:
    _HAS_POLARS = False
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def _load_table(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    ext = os.path.splitext(path)[1].lower()
    if _HAS_POLARS:
        if ext in [".parquet"]:
            return "polars", pl.read_parquet(path)
        elif ext in [".csv", ".tsv"]:
            sep = "," if ext==".csv" else "\t"
            return "polars", pl.read_csv(path, separator=sep, infer_schema_length=2000)
    if _HAS_PANDAS:
        if ext in [".parquet"]:
            return "pandas", pd.read_parquet(path)
        elif ext in [".csv", ".tsv"]:
            sep = "," if ext==".csv" else "\t"
            return "pandas", pd.read_csv(path, sep=sep)
    # fallback: error
    raise RuntimeError("No se pudo cargar el archivo (falta Polars/Pandas o extensión no soportada).")

def _to_numpy_col(df, col: str):
    if isinstance(df, dict):
        return np.array(df.get(col, []))
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        return df.select(col).to_numpy().ravel() if col in df.columns else None
    if _HAS_PANDAS and isinstance(df, pd.DataFrame):
        return df[col].to_numpy() if col in df.columns else None
    return None

def _list_columns(df):
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        return df.columns
    if _HAS_PANDAS and isinstance(df, pd.DataFrame):
        return list(df.columns)
    return []

def _select_df(df, cols):
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        keep = [c for c in cols if c in df.columns]
        return df.select(keep)
    if _HAS_PANDAS and isinstance(df, pd.DataFrame):
        keep = [c for c in cols if c in df.columns]
        return df[keep]
    return None

def _impute_nan_zero(arr2d):
    arr = np.array(arr2d, dtype=float)
    m = np.isnan(arr)
    if m.any():
        arr[m] = 0.0
    return arr

def _maybe_calc_importances(df, target_col="Tc_K", topn=25):
    """
    Si no hay archivo de importancias, trata de calcularlas con RF rápido.
    Requiere scikit-learn.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return None

    cols = _list_columns(df)
    feat = [c for c in cols if c.startswith("frac_") or c.startswith("mag_")]
    if target_col not in cols or len(feat)==0:
        return None

    # X, y
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        X = df.select(feat).to_numpy()
        y = df.select(target_col).to_numpy().ravel()
    else:
        X = df[feat].to_numpy()
        y = df[target_col].to_numpy().ravel()

    X = _impute_nan_zero(X)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    try:
        model.fit(X, y)
        imps = model.feature_importances_
    except Exception:
        return None

    order = np.argsort(imps)[::-1][:topn]
    return [(feat[i], float(imps[i])) for i in order]

def _load_importance_table(path: str, topn=25):
    if not path or not os.path.exists(path):
        return None
    _, tbl = _load_table(path)
    cols = _list_columns(tbl)
    # Esperado: columns ~ ["feature","importance"] (y opcional "family")
    if "feature" not in cols or "importance" not in cols:
        return None
    if _HAS_POLARS and isinstance(tbl, pl.DataFrame):
        rows = list(zip(tbl["feature"].to_list(), tbl["importance"].to_list()))
    else:
        rows = list(zip(tbl["feature"].tolist(), tbl["importance"].tolist()))
    rows = sorted(rows, key=lambda x: x[1], reverse=True)[:topn]
    return rows

def plot_hist_box_violin(ax_hist, ax_box, ax_violin, data, title_prefix="Tc (K)"):
    # Histograma
    ax_hist.hist(data, bins=50)
    ax_hist.set_title(f"{title_prefix} — Histograma")
    ax_hist.set_xlabel("Tc (K)"); ax_hist.set_ylabel("Frecuencia")

    # Boxplot
    ax_box.boxplot([data], vert=True, showmeans=True)
    ax_box.set_title(f"{title_prefix} — Boxplot")
    ax_box.set_ylabel("Tc (K)")

    # Violin
    ax_violin.violinplot([data], showmeans=True, showmedians=True, showextrema=True)
    ax_violin.set_title(f"{title_prefix} — Violin")
    ax_violin.set_ylabel("Tc (K)")

def plot_scatter_and_regression(ax_sc, ax_reg, x, y, xname="feature", yname="Tc_K"):
    # Scatter
    ax_sc.scatter(x, y, s=10)
    ax_sc.set_title(f"{yname} vs {xname} — Scatter")
    ax_sc.set_xlabel(xname); ax_sc.set_ylabel(yname)

    # Regresión lineal simple
    # filtramos NaN/inf
    m = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[m], y[m]
    if len(xf) >= 2:
        coef = np.polyfit(xf, yf, 1)
        xline = np.linspace(np.nanmin(xf), np.nanmax(xf), 100)
        yline = coef[0]*xline + coef[1]
        ax_reg.scatter(xf, yf, s=10)
        ax_reg.plot(xline, yline)
        ax_reg.set_title(f"{yname} vs {xname} — Regresión")
        ax_reg.set_xlabel(xname); ax_reg.set_ylabel(yname)
        # R^2 simple
        ypred = coef[0]*xf + coef[1]
        ss_res = np.sum((yf-ypred)**2)
        ss_tot = np.sum((yf-np.mean(yf))**2)
        r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else float("nan")
        ax_reg.text(0.02, 0.95, f"R² = {r2:.3f}", transform=ax_reg.transAxes, verticalalignment='top')

def plot_feature_importance(ax, imp_rows, title="Importancia de características (Top-N)"):
    if not imp_rows or len(imp_rows) == 0:
        ax.text(0.5, 0.5, "Sin importancias disponibles", ha="center", va="center")
        ax.set_axis_off()
        return
    feats = [r[0] for r in imp_rows]
    vals = [r[1] for r in imp_rows]
    idx = np.arange(len(feats))
    ax.bar(idx, vals)
    ax.set_title(title)
    ax.set_xticks(idx)
    ax.set_xticklabels(feats, rotation=90)
    ax.set_ylabel("Importancia")
    ax.set_xlabel("Feature")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ruta a df_A (Parquet/CSV/TSV) con al menos Tc_K y features (frac_*, mag_*)")
    ap.add_argument("--out", default="reporte_supercon_figs.pdf", help="Ruta del PDF de salida")
    ap.add_argument("--feat-importances-rf", default=None, help="CSV/Parquet con columnas feature,importance (RF)")
    ap.add_argument("--feat-importances-xgb", default=None, help="CSV/Parquet con columnas feature,importance (XGB/HGB)")
    ap.add_argument("--xname", default="frac_O", help="Feature para scatter/regresión 1 (ej. frac_O)")
    ap.add_argument("--yname", default="Tc_K", help="Nombre de la columna objetivo (Tc_K)")
    ap.add_argument("--xname2", default="mag_thermal_conductivity_range", help="Feature para scatter/regresión 2")
    ap.add_argument("--topn", type=int, default=25, help="Top-N de importancias a graficar")
    args = ap.parse_args()

    _, df = _load_table(args.data)
    cols = _list_columns(df)

    if args.yname not in cols:
        raise ValueError(f"No se encontró la columna objetivo '{args.yname}' en {cols}")

    # Extrae Y y candidates X
    y = _to_numpy_col(df, args.yname)
    x1 = _to_numpy_col(df, args.xname) if args.xname in cols else None
    x2 = _to_numpy_col(df, args.xname2) if args.xname2 in cols else None

    # Carga/Calcula importancias
    imp_rf  = _load_importance_table(args.feat_importances_rf, topn=args.topn)
    if imp_rf is None:
        imp_rf = _maybe_calc_importances(df, target_col=args.yname, topn=args.topn)
    imp_xgb = _load_importance_table(args.feat_importances_xgb, topn=args.topn)

    with PdfPages(args.out) as pdf:
        # Página 1: Distribuciones Tc
        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.add_subplot(3,1,1)
        ax2 = fig1.add_subplot(3,1,2)
        ax3 = fig1.add_subplot(3,1,3)
        plot_hist_box_violin(ax1, ax2, ax3, y, title_prefix="Tc (K)")
        fig1.tight_layout()
        pdf.savefig(fig1); plt.close(fig1)

        # Página 2: Scatter/Regresión vs X1 (si existe)
        if x1 is not None:
            fig2 = plt.figure(figsize=(10, 8))
            ax21 = fig2.add_subplot(2,1,1)
            ax22 = fig2.add_subplot(2,1,2)
            plot_scatter_and_regression(ax21, ax22, x1, y, xname=args.xname, yname=args.yname)
            fig2.tight_layout()
            pdf.savefig(fig2); plt.close(fig2)

        # Página 3: Scatter/Regresión vs X2 (si existe)
        if x2 is not None:
            fig3 = plt.figure(figsize=(10, 8))
            ax31 = fig3.add_subplot(2,1,1)
            ax32 = fig3.add_subplot(2,1,2)
            plot_scatter_and_regression(ax31, ax32, x2, y, xname=args.xname2, yname=args.yname)
            fig3.tight_layout()
            pdf.savefig(fig3); plt.close(fig3)

        # Página 4: Importancias RF (si hay)
        fig4 = plt.figure(figsize=(10, 6))
        ax4 = fig4.add_subplot(1,1,1)
        plot_feature_importance(ax4, imp_rf, title=f"Importancias (RF) Top-{args.topn}")
        fig4.tight_layout()
        pdf.savefig(fig4); plt.close(fig4)

        # Página 5: Importancias XGB/HGB (si hay)
        fig5 = plt.figure(figsize=(10, 6))
        ax5 = fig5.add_subplot(1,1,1)
        plot_feature_importance(ax5, imp_xgb, title=f"Importancias (XGB/HGB) Top-{args.topn}")
        fig5.tight_layout()
        pdf.savefig(fig5); plt.close(fig5)

    print(f"PDF guardado en: {args.out}")


if __name__ == "__main__":
    main()
