# Solar-Wind Short-Term Forecast • French Grid  
**Probabilistic +1 h / +24 h renewable-power prediction   

> Hour-ahead and day-ahead forecasts of solar ☀️ & wind 💨 generation for mainland-France, built with a Quantile Gradient-Boosting Regressor (QGBR) and visualised in Tableau.

---

## Contents

| Path | Purpose |
|------|---------|
| `project_v5.py` | End-to-end Python pipeline – data ingest, feature engineering, QGBR training, back-testing, SHAP interpretability. |
| `dashboard_renovables.twbx`<br/>*(or `.twb`)* | Packaged Tableau dashboard that displays forecasts, 80 % uncertainty band, error diagnostics and feature importance. |

> **Dataset not included**  
> Download the raw CSV from Kaggle (see below) and place it next to the script before running.

---

## 1 · Background & Data Source

* **Dataset:** Hourly solar and wind production (MW) for mainland France from 2020-01-01 onward – published by *Commission de Régulation de l’Énergie (CRE)* to compute the reference price for variable premiums.  
  <https://www.kaggle.com/datasets/henriupton/wind-solar-electricity-production>  
* **Goal:** Provide probabilistic (+1 h and +24 h) forecasts so producers can anticipate revenue gaps covered by the CRE premium mechanism.  
* **Why QGBR?** Ensemble tree models handle non-linear weather relationships and, with quantile loss, output predictive intervals (q05…q95) instead of single points.

---

