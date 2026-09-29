# AQI Prediction System — Delhi NCR 🌫️

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end Air Quality Index (AQI) prediction and analysis for Delhi NCR using machine learning, time series forecasting, and unsupervised clustering — covering 23 station locations and 201,664 readings (2020–2025).

---

## 🌟 Overview

Delhi NCR consistently ranks among the world's most polluted regions. This project builds a complete ML pipeline to predict short-term (next-reading) AQI, forecast 30-day trends, and cluster the 23 monitoring stations into pollution tiers.

---

## 📂 Project Structure

```text
.
├── AQI_(5).ipynb                    # Main notebook
└── r2_comparison_pollutants.png     # R² comparison (pollutant-based model)
```

---

## 📊 Dataset

- **Source:** TODO — add dataset name/link
- **Stations:** 23 monitoring stations across Delhi, Noida, Greater Noida, Gurugram, Faridabad, Ghaziabad
- **Sampling:** 4 readings/day (6AM, 12PM, 6PM, 11PM), 2020-01-01 to 2025-12-31
- **Records:** 201,664 rows × 25 columns, no missing values
- **Features:** `pm25`, `pm10`, `no2`, `so2`, `co`, `o3`, `temperature`, `humidity`, `wind_speed`, `visibility`, `aqi`
- **Note:** AQI values are capped at 500 per CPCB scale
- The CSV (`delhi_ncr_aqi_dataset.csv`) is not included in this repo — place it in the project folder before running.

---

## 🧪 Models & Results

Each experiment below uses its own split, so scores are not directly comparable across sections.

### 1. Short-Term AQI Regression (Main Model)

Target: `aqi_next = aqi.shift(-1)` (AQI of the following reading, ~6 hours ahead) | Features: 6 pollutants | Chronological 80/20 split

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Linear Regression | 0.613 | 92.55 | 111.34 |
| **XGBoost** | **0.9521** | **28.82** | **39.15** |

> The large gap between linear and XGBoost on the same split points to a **non-linear** relationship between pollutants and AQI.

**Top features (XGBoost, cover importance):** PM2.5 ranks highest; CO, O3, SO2, PM10 and NO2 follow at similar levels.

### 2. Same-Day AQI from Pollutants (Separate Experiment)

Features: 6 pollutants | Same-day prediction | Random split

| Model | R² Score | MAE |
|-------|----------|-----|
| Lasso (α=1.0) | 0.6426 | 88.72 |
| Ridge (α=1.0) | 0.6438 | 88.26 |

### 3. Weather-Based Regression (Separate Experiment)

Features: `temperature`, `humidity`, `wind_speed`, `visibility` | Same-day prediction | Random split

| Model | R² Score | MAE |
|-------|----------|-----|
| Linear Regression | 0.830 | 57.94 |
| **ANN** (64→32→16→1) | **0.9469** | — |
| XGBoost | 0.9582 | 25.25 |

> Visibility has the strongest single correlation with AQI (r = −0.86), followed by temperature (r = −0.73) — high pollution proportionally reduces visibility.

### 4. AQI Category Classification

6 classes per CPCB scale: Good / Satisfactory / Moderate / Poor / Very Poor / Severe | Weather features | Random Forest (`class_weight='balanced'`)

| Metric | Value |
|--------|-------|
| Accuracy | 69.85% |
| Macro F1 | 0.64 |
| Severe (F1) | 0.88 |
| Good (recall / precision) | 0.89 / 0.42 |
| Satisfactory (recall) | 0.30 |

> Class balancing lifts recall on the rare "Good" class, but the middle categories (Satisfactory, Poor) remain the hardest to separate.

---

## 📅 Time Series Forecasting (Prophet)

- **Model:** Facebook Prophet with yearly + weekly seasonality, fitted on daily mean AQI
- **Forecast horizon:** 30 days (Jan 1–30, 2026)
- **Jan 26–30, 2026 predictions:** yhat 431–439, consistent with the winter peak in the training data
- **Yearly pattern:** Nov–Jan peak (~+200), Jul–Sep trough (~−200, monsoon)
- **Weekly pattern:** Weekdays higher than weekends (Prophet effect ≈ +4 vs −10; median AQI 242 vs 207)
- **Note:** `daily_seasonality=False` — dataset is 4×daily, not hourly
- **Note:** No holdout evaluation was run on the forecast

---

## 🗺️ Spatial Clustering

### K-Means (4 clusters on station-level pollutant profiles: PM2.5, PM10, NO2, CO)

| Tier | Avg AQI | Stations | Examples |
|------|---------|----------|----------|
| Highest | 288.8 | 4 | Anand Vihar, Jahangirpuri, Wazirpur, Bawana |
| High | 274.3 | 5 | Ghaziabad Loni, ITO, Okhla Phase 2, Punjabi Bagh |
| Moderate | 263.2 | 6 | Shadipur, Rohini, Noida Sec 62, Faridabad |
| Lowest | 251.0 | 8 | NSIT Dwarka, Mandir Marg, Siri Fort, Greater Noida |

> All tiers fall within the "Poor" AQI category (200–300). Tiers are ranked by mean AQI only — no land-use data was used, so they are **relative**, not absolute.

### Agglomerative Clustering (Hierarchical, Ward linkage)
Standardized AQI, PM2.5 and PM10 station means, four clusters. Produces the same four average-AQI levels as K-Means. Results are shown on an interactive Folium map with a heatmap overlay in the notebook.

---

## 🔄 Workflow

```
Raw CSV → EDA (correlation, seasonality, station ranking)
        → Feature Engineering (aqi.shift(-1) target)
        → Regression (Linear → XGBoost, Lasso/Ridge, ANN)
        → Classification (RandomForest, 6 AQI categories)
        → Forecasting (Prophet 30-day)
        → Clustering (KMeans + Agglomerative + Folium map)
```

---

## ⚠️ Limitations

- The main model predicts the AQI of the next row (date → station → hour order), about 6 hours ahead. For 23:00 readings, the next row is a different station, so it is a short-term model, not a strict next-day forecast.
- Experiments 2–4 use random splits on time-ordered data, so scores are optimistic.
- AQI is capped at 500, and the Prophet forecast has no holdout evaluation.

---

## 🚀 Getting Started

```bash
git clone https://github.com/bytecraft17/AQI-Project.git
cd AQI-Project
pip install pandas numpy scikit-learn xgboost prophet folium matplotlib seaborn tensorflow
jupyter notebook "AQI_(5).ipynb"
```

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `Facebook Prophet` · `TensorFlow/Keras` · `Folium` · `Matplotlib` · `Seaborn`

---

## 📄 License

MIT License
