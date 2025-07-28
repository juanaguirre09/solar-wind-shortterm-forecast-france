# -------------------------------------------------------------
# Probabilistic forecast 1 h and 24 h for Solar & Wind (France)
# Unified script · Phases 1‑10 · Optimised for 16 GB RAM
# -------------------------------------------------------------
"""
This single file reproduces every phase agreed for the Talento Tech
project, now cleaned of accented characters to avoid encoding issues.
It follows the feature‑engineering layout you provided and corrects the
CSV round‑trip bug that raised:

    ValueError: Missing column provided to 'parse_dates': 'Unnamed: 0'

Predictions are exported with the index label **datetime**, and later
re‑imported with `parse_dates=["datetime"], index_col="datetime"`, so
the error disappears.

Phases included
===============
1  Data loading and sanity checks
2  Quick EDA with on‑screen plots (+ PNG copies)
3  Feature engineering (calendar, lags, rolling, cyclic, holidays)
4  Baseline models (Persistence +1 h, SNaive +24 h)
5  Dataset preparation per resource/horizon (temporal 80/20 split)
6  Gradient Boosting quantile models (q10, q50, q90)
7  Evaluation (MAE, RMSE, Coverage 80 %, pinball loss)
8  CSV export of predictions (one file per model)
9  Diagnostic scatter Real vs Pred q50
10 All plots also saved to *outputs/*
"""

# ------------------ FASE 2 · IMPORTS & GLOBALS -------------------------
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays

# Paths
DATA_PATH = Path("database_power_production.csv")
OUT_DIR   = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

# Hyper‑parameters
GBR_PARAMS = dict(loss="quantile", learning_rate=0.05,
                  n_estimators=300, max_depth=3, random_state=42)
QUANTILES  = [0.1, 0.5, 0.9]
RESOURCES  = ["Solar", "Wind"]
HORIZONS   = {"+1h": 1, "+24h": 24}

# ------------------ 2 · HELPER FUNCTIONS --------------------------
'''Funciones que fueron definidas en la fase 1 para evaluar el desempeño que 
del modelo, en este apartado se crean dichas funciones '''

def pinball_loss(y_true, y_pred, q):
    delta = y_true - y_pred
    return np.mean(np.maximum(q * delta, (q - 1) * delta))


def coverage80(y, q10, q90):
    return ((y >= q10) & (y <= q90)).mean() * 100


def mae_rmse(y, yhat):
    mae  = mean_absolute_error(y, yhat)
    rmse = mean_squared_error(y, yhat) ** 0.5
    return mae, rmse


# ------------------ FASE 2 · LOAD & CLEAN DATA -------------------------
print("\n=== Loading data ===")
raw = pd.read_csv(DATA_PATH)
raw["Date and Hour"] = pd.to_datetime(raw["Date and Hour"], utc=True)
raw = raw.set_index("Date and Hour")

pivot = raw.pivot_table(values="Production", index=raw.index, columns="Source")
print("Total rows:", len(pivot))

# Drop rows where both resources are NaN
pivot = pivot.dropna(subset=["Solar", "Wind"], how="all")
print("Rows after cleaning:", len(pivot))


# ------------------ FASE 3 · QUICK EDA ---------------------------------
print("\n=== Descriptive statistics ===")
print(pivot[RESOURCES].describe())

# Helper to plot grouped means

def _plot_group(grouper, title, fname, labels=None):
    grp = pivot.groupby(grouper)[RESOURCES].mean()
    if labels is not None:
        grp.index = labels
    grp.plot(figsize=(8, 3), title=title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname)
    plt.show(); plt.close()

_plot_group(pivot.index.hour, "Mean by hour of day", "eda_hour.png")
_plot_group(pivot.index.dayofweek, "Mean by day of week",
            "eda_dow.png", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
_plot_group(pivot.index.month, "Mean by month", "eda_month.png",
            ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

# Correlation heat‑map
corr = pivot[RESOURCES].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Solar–Wind")
plt.tight_layout(); plt.savefig(OUT_DIR/"eda_corr.png"); plt.show(); plt.close()

# ACF up to 48 h for each resource
for res in RESOURCES:
    plt.figure(figsize=(8, 3))
    plot_acf(pivot[res].dropna(), lags=48)
    plt.title(f"ACF {res} (48 h)")
    plt.tight_layout(); plt.savefig(OUT_DIR/f"acf_{res.lower()}.png"); plt.show(); plt.close()


# ------------------ 5 · FASE 4 - FEATURE ENGINEERING -----------------------
print("\n=== Creating features ===")
fr_holidays = holidays.FR()
aux = pivot.copy()

# Calendar features
aux["hour"]      = aux.index.hour
aux["dayofweek"] = aux.index.dayofweek
aux["month"]     = aux.index.month
aux["doy"]       = aux.index.dayofyear

# Cyclic encoding
aux["hour_sin"] = np.sin(2 * np.pi * aux["hour"] / 24)
aux["hour_cos"] = np.cos(2 * np.pi * aux["hour"] / 24)
aux["doy_sin" ] = np.sin(2 * np.pi * aux["doy"]  / 365)
aux["doy_cos" ] = np.cos(2 * np.pi * aux["doy"]  / 365)

# Holidays and daylight flag
aux["is_holiday"]  = aux.index.normalize().isin(fr_holidays).astype(int)
aux["is_daylight"] = (aux["Solar"] > 0).astype(int)

# Lags & rolling stats
for res in RESOURCES:
    aux[f"{res.lower()}_lag1"]    = aux[res].shift(1)
    aux[f"{res.lower()}_lag24"]   = aux[res].shift(24)
    aux[f"{res.lower()}_roll24"]  = aux[res].rolling(24).mean()
    aux[f"{res.lower()}_roll168"] = aux[res].rolling(168).mean()

aux = aux.dropna()


# ------------------ 6 · FASE 5 - BASELINE MODELS ---------------------------
print("\n=== Baseline metrics ===")

# Targets just for baseline (added again below for full dataset)
for res in RESOURCES:
    aux[f"{res.lower()}_t+1"]  = aux[res].shift(-1)
    aux[f"{res.lower()}_t+24"] = aux[res].shift(-24)
aux = aux.dropna()

for res in RESOURCES:
    # Persistence +1 h
    mae, rmse_val = mae_rmse(aux[f"{res.lower()}_t+1"], aux[f"{res.lower()}_lag1"])
    print(f"{res} +1h Persistence — MAE {mae:.2f}, RMSE {rmse_val:.2f}")

    # SNaive +24 h
    mae, rmse_val = mae_rmse(aux[f"{res.lower()}_t+24"], aux[f"{res.lower()}_lag24"])
    print(f"{res} +24h SNaive      — MAE {mae:.2f}, RMSE {rmse_val:.2f}")


# ------------------ 7 · DATASET PREPARATION -----------------------
print("\n=== Building training & test sets ===")

datasets = {}

# Create shifted targets
for res in RESOURCES:
    for tag, h in HORIZONS.items():
        aux[f"{res.lower()}_t{tag}"] = aux[res].shift(-h)

# Feature list (base)
base_feats = [
    "hour", "dayofweek", "month", "is_holiday",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "solar_lag1", "solar_lag24", "solar_roll24", "solar_roll168",
    "wind_lag1",  "wind_lag24",  "wind_roll24",  "wind_roll168",
]

for res in RESOURCES:
    for tag in HORIZONS.keys():
        key      = f"{res}_{tag}"
        target   = f"{res.lower()}_t{tag}"
        feat_cols = base_feats.copy()
        if res == "Solar":
            feat_cols.append("is_daylight")
        # Drop any row with NaN in used cols
        clean = aux.dropna(subset=feat_cols + [target])
        X_all = clean[feat_cols]
        y_all = clean[target]
        split = int(len(X_all) * 0.8)
        datasets[key] = dict(
            X_train=X_all.iloc[:split],
            y_train=y_all.iloc[:split],
            X_test =X_all.iloc[split:],
            y_test =y_all.iloc[split:],
            idx_test=X_all.iloc[split:].index,
        )


# ------------------ 8 · FASE 6 - TRAINING & PREDICTIONS --------------------
print("\n=== Training quantile GBMs ===")
results = []

for key, data in datasets.items():
    preds_df = pd.DataFrame(index=data["idx_test"])
    for q in QUANTILES:
        gbr = GradientBoostingRegressor(alpha=q, **GBR_PARAMS)
        gbr.fit(data["X_train"], data["y_train"])
        preds_df[f"pred_q{int(q*100)}"] = gbr.predict(data["X_test"])

    # Evaluation on q50
    mae, rmse_val = mae_rmse(data["y_test"], preds_df["pred_q50"])
    cov = coverage80(data["y_test"], preds_df["pred_q10"], preds_df["pred_q90"])
    results.append({
        "Model": key,
        "MAE": round(mae, 2),
        "RMSE": round(rmse_val, 2),
        "Coverage80": round(cov, 1),
    })

    # Export — include explicit index label to avoid unnamed column
    export_df = pd.concat([
        data["y_test"].rename("y_true"),
        preds_df
    ], axis=1)
    export_df.to_csv(f"preds_{key}.csv", index_label="datetime")

# Summary metrics
metrics = pd.DataFrame(results)
print("\n=== Metrics (q50) ===")
print(metrics)
# metrics.to_csv(OUT_DIR/"metrics_summary.csv", index=False)


# ------------------ 9 · DIAGNOSTIC SCATTERS -----------------------
    
print("\n=== Plotting diagnostics + exportando para Tableau ===")
for key in datasets.keys():
    # 1 · Cargar predicciones
    df_pred = pd.read_csv(
        f"preds_{key}.csv",               # ajusta si el nombre difiere
        parse_dates=["datetime"],
        index_col="datetime"
    )

    # 2 · Series real y predicha
    y_test = df_pred["y_true"]
    y_hat  = df_pred["pred_q50"]

    # 3 · DataFrame para Tableau
    df_tbl = df_pred.reset_index()[["datetime", "y_true", "pred_q50"]].copy()
    df_tbl["Date"]       = df_tbl["datetime"].dt.date
    df_tbl["Hour"]       = df_tbl["datetime"].dt.time
    df_tbl["Tecnologia"] = "solar" if "solar" in key.lower() else "wind"
    df_tbl["Horizonte"]  = "+1h"  if "h1"    in key.lower() else "+24h"

    columnas = ["Date", "Hour", "Tecnologia", "Horizonte",
                "y_true", "pred_q50"]
    df_tbl = df_tbl[columnas]

    # 4 · Guardar directamente en el directorio actual
    # df_tbl.to_csv(
    #     f"diag_{key}.csv",     # ← sin rutas adicionales
    #     sep=";",
    #     quotechar='"',
    #     index=False
    # )

    # 5 · Seguir con la gráfica PNG
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_hat, alpha=0.3)
    max_val = max(y_test.max(), y_hat.max())
    plt.plot([0, max_val], [0, max_val], "--", lw=1)
    plt.title(f"Real vs Pred q50 — {key}")
    plt.xlabel("Real (MW)")
    plt.ylabel("Pred q50 (MW)")
    plt.tight_layout()
    plt.savefig(f"diag_{key}.png")
    plt.show()
    plt.close()

print("\n>>> Exportación para Tableau completada <<<")


print("\n>>> Script finished successfully <<<")

# ------------------ FASE 7 · LIGHT BACKTEST -----------------------
print("\n=== Light backtesting (4 rolling monthly cutoffs) ===")

cutoffs = pd.to_datetime(
    ["2020-01-31", "2020-04-30", "2020-07-31", "2020-10-31"], utc=True
)

bt_rows = []
for res in RESOURCES:
    feats = base_feats + (["is_daylight"] if res == "Solar" else [])
    for tag, h in HORIZONS.items():
        ycol = f"{res.lower()}_t{tag}"
        for c in cutoffs:
            tr_mask = aux.index <= c
            te_start = c + pd.Timedelta(hours=1)
            te_end   = te_start + pd.Timedelta(days=27)
            te_mask  = (aux.index >= te_start) & (aux.index <= te_end)

            if te_mask.sum() == 0:
                continue

            X_tr, y_tr = aux.loc[tr_mask, feats], aux.loc[tr_mask, ycol]
            X_te, y_te = aux.loc[te_mask, feats], aux.loc[te_mask, ycol]

            model = GradientBoostingRegressor(alpha=0.5, **GBR_PARAMS)
            model.fit(X_tr, y_tr)
            y_hat = model.predict(X_te)

            mae_val, rmse_val = mae_rmse(y_te, y_hat)

            bt_rows.append(
                dict(
                    cutoff=c.date(),
                    horizon=f"{res}_{tag}",
                    MAE=round(mae_val, 2),
                    RMSE=round(rmse_val, 2),
                )
            )

bt_df = pd.DataFrame(bt_rows)
# bt_df.to_csv(OUT_DIR / "backtesting_summary.csv", index=False)

# print("Backtesting summary saved to outputs/backtesting_summary.csv")
print(bt_df.groupby("horizon")[["MAE", "RMSE"]].mean().round(2))

# ------------------ 12 · GRAFICAS MAE-RMSE Y COVERAGE 80 % ------------------
import seaborn as sns
import matplotlib.pyplot as plt

# ── a) MAE y RMSE en una sola figura ─────────────────────────────────────────
metrics_long = metrics.melt(
    id_vars="Model",           # columnas que se mantienen
    value_vars=["MAE", "RMSE"],# métricas que se apilan
    var_name="Metric",         # nombre nueva columna
    value_name="Value"         # nombre valores
)

plt.figure(figsize=(8, 4))
sns.barplot(
    data=metrics_long,
    y="Model", x="Value", hue="Metric",
    palette="Set2"
)
plt.title("MAE vs RMSE por modelo / horizonte")
plt.xlabel("MW")
plt.ylabel("")
plt.legend(title="")
plt.tight_layout()
plt.savefig(OUT_DIR / "bar_mae_rmse.png")
plt.show()

# ── b) Cobertura 80 % en figura aparte ──────────────────────────────────────
plt.figure(figsize=(6, 4))
sns.barplot(
    data=metrics.sort_values("Coverage80", ascending=False),
    y="Model", x="Coverage80",
    palette="Blues_r"
)
plt.title("Cobertura 80 % por modelo / horizonte")
plt.xlabel("Cobertura (%)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(OUT_DIR / "bar_coverage80.png")
plt.show()

# ================================================================
#  FASE 8 · INTERPRETABILIDAD SHAP (4 modelos)
# ================================================================
import shap
shap.initjs()

for key in ["Solar_+1h", "Wind_+1h", "Solar_+24h", "Wind_+24h"]:
    data = datasets[key]

    # 1) Reentrenar rápidamente el modelo q50 para esta fase
    model_shap = GradientBoostingRegressor(alpha=0.5, **GBR_PARAMS)
    model_shap.fit(data["X_train"], data["y_train"])

    # 2) Muestras de fondo y de evaluación (ligeras, ≤16 GB RAM)
    background  = shap.sample(data["X_train"],
                              nsamples=min(1000, len(data["X_train"])),
                              random_state=42)
    X_test_smp  = shap.sample(data["X_test"],
                              nsamples=min(300, len(data["X_test"])),
                              random_state=42)

    # 3) Cálculo de valores SHAP
    explainer   = shap.Explainer(model_shap.predict, background)
    shap_values = explainer(X_test_smp)
    
        # ── convertir a porcentaje de importancia global ────────────────
    mean_abs = np.abs(shap_values.values).mean(axis=0)          # |SHAP| medio
    pct_imp  = 100 * mean_abs / mean_abs.sum()                  # % contribución
    imp_df   = pd.Series(pct_imp, index=data["X_train"].columns)\
                 .sort_values(ascending=False)[:12]             # top-12
    
    # ── plot en % ───────────────────────────────────────────────────
    plt.figure(figsize=(6,4))
    sns.barplot(x=imp_df.values, y=imp_df.index, palette="viridis")
    plt.xlabel("Importancia SHAP (%)")
    plt.title(f"SHAP importance — {key}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"shap_bar_{key.replace('+','')}_pct.png")
    plt.show()

    # 4) Gráfica de importancia global (barplot)
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title(f"SHAP importance — {key}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"shap_bar_{key.replace('+','')}.png")
    plt.show()
    
        # ── exportar a CSV compatible con Tableau ─────────────────────────
    import csv
    
    # imp_df es una Serie: índice = feature, valor = % importancia
    shap_tbl = (
        imp_df
        .rename("Importance_pct")
        .reset_index()
        .rename(columns={"index": "Feature"})
    )
    
    # out_file = OUT_DIR / f"shap_importance_{key.replace('+','')}_pct.csv"
    # shap_tbl.to_csv(
    #     out_file,
    #     sep=";",                  # separador punto-y-coma
    #     quotechar='"',            # comillas dobles
    #     quoting=csv.QUOTE_ALL,    # cita todas las celdas
    #     index=False,
    #     float_format="%.4f",
    #     encoding="utf-8"
    # )
    
    # print(f"Tabla SHAP exportada a: {out_file}")


import pandas as pd

# Si el índice es 'Date and Hour', restablecerlo como columna
aux = aux.reset_index()

# Convertir la columna a datetime si aún no lo está
aux["Date and Hour"] = pd.to_datetime(aux["Date and Hour"])

# Crear las columnas Date y Hour
aux["Date"] = aux["Date and Hour"].dt.date
aux["Hour"] = aux["Date and Hour"].dt.time

# Reordenar columnas: Date y Hour primero
cols = ["Date", "Hour"] + [col for col in aux.columns if col not in ["Date and Hour", "Date", "Hour"]]
aux_reordered = aux[cols]

# Exportar en formato compatible con Tableau
# aux_reordered.to_csv("datos_limpios_para_tableau.csv", sep=';', quotechar='"', index=False)









