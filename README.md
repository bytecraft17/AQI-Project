# AQI Prediction System — Delhi NCR 🌫️

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end Air Quality Index (AQI) prediction and analysis for Delhi NCR using machine learning, time series forecasting, and unsupervised clustering — built on real monitoring station data across 23 locations.

---

## 🌟 Overview

Delhi NCR consistently ranks among the world's most polluted regions. This project builds a complete ML pipeline to predict next-day AQI, forecast 30-day trends, and spatially cluster pollution zones across 23 monitoring stations.

---

## 📂 Project Structure

```text
.
├── AQI_Prediction.ipynb         # Main notebook (all cells verified)
├── delhi_ncr_aqi_dataset.csv    # Raw dataset (add your path)
└── plots/
    ├── correlation_heatmap.png
    ├── monthly_aqi_heatmap.png
    ├── clustering_map.html       # Interactive folium map
    ├── prophet_forecast.png
    └── performance_comparison.png
```

---

## 📊 Dataset

- **Source:** Delhi NCR air quality monitoring network
- **Stations:** 23 monitoring stations across Delhi, Noida, Gurgaon, Faridabad, Ghaziabad
- **Sampling:** 4 readings/day (6AM, 12PM, 6PM, 11PM)
- **Records:** 66,000+
- **Features:** `pm25`, `pm10`, `no2`, `so2`, `co`, `o3`, `temperature`, `humidity`, `wind_speed`, `visibility`, `aqi`
- **Note:** AQI values capped at 500 per CPCB scale (~22% of records at ceiling)

---

## 🧪 Models & Results

### 1. Next-Day AQI Regression (Main Model)

Target: `aqi_next_day = aqi.shift(-1)` | Features: 6 pollutants | Chronological split

| Model | R² Score | MAE |
|-------|----------|-----|
| Linear Regression | 0.613 | — |
| Lasso (α=1.0) | 0.643 | 88.72 |
| Ridge (α=1.0) | 0.644 | 88.26 |
| **XGBoost** | **0.9521** | — |

> Lasso/Ridge showed negligible gain over LR (+0.03 R²), confirming the problem is **non-linear** — not overfitting. XGBoost captures this non-linearity effectively.

**Top features (XGBoost):** PM2.5 is the dominant predictor, followed by PM10 and CO.

### 2. Weather-Based Regression (Separate Experiment)

Features: `temperature`, `humidity`, `wind_speed`, `visibility` | Same-day prediction

| Model | R² Score |
|-------|----------|
| Linear Regression | 0.830 |
| **ANN** (64→32→16→1) | **0.9469** |
| XGBoost | 0.9567 |

> Visibility has the strongest single correlation with AQI (r = −0.857) — high pollution proportionally reduces visibility. Weather-based models use random split (not chronological) — not directly comparable to next-day model.

### 3. AQI Category Classification

6 classes per CPCB scale: Good / Satisfactory / Moderate / Poor / Very Poor / Severe

| Model | Accuracy |
|-------|----------|
| RandomForest (baseline) | 73.98% |
| RandomForest (class_weight='balanced') | 69.85% |

> Balanced model preferred — "Good" category recall improved from 0.04 → 0.89.

---

## 📅 Time Series Forecasting (Prophet)

- **Model:** Facebook Prophet with yearly + weekly seasonality
- **Forecast horizon:** 30 days
- **Jan 26–30, 2026 predictions:** yhat 431–439 (realistic for peak winter pollution)
- **Yearly pattern:** Nov–Jan peak (+200), Jul–Sep trough (−200, monsoon)
- **Weekly pattern:** Weekdays ~5 AQI higher than weekends (traffic effect)
- **Note:** `daily_seasonality=False` — dataset is 4×daily, not hourly

---

## 🗺️ Spatial Clustering

### K-Means (4 clusters on pollutant profiles)

| Cluster | Avg AQI | Area Type | Example Stations |
|---------|---------|-----------|-----------------|
| Severe | 288.8 | Industrial/Hotspots | Anand Vihar, Jahangirpuri, Wazirpur |
| High | 274–279 | Traffic/Commercial | ITO, Okhla, Punjabi Bagh, Ghaziabad |
| Moderate | 260–266 | Mixed Use | Rohini, Faridabad, Noida Sec 62 |
| Low | 244–256 | Residential/Safe | NSIT Dwarka, Greater Noida, Siri Fort |

> All clusters fall within "Poor" AQI category (200–300) — labels are **relative**, not absolute.

### Agglomerative Clustering (Hierarchical, Ward linkage)
Interactive folium map with heatmap overlay — pollution hotspots clearly visible in North/East Delhi.

---

## 🔄 Workflow

```
Raw CSV → EDA (correlation, seasonality, station ranking)
        → Feature Engineering (aqi_next_day target)
        → Regression (LR → Lasso/Ridge → XGBoost)
        → Classification (RandomForest, 6 AQI categories)
        → Forecasting (Prophet 30-day)
        → Clustering (KMeans + Agglomerative + Folium map)
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/bytecraft17/AQI-Prediction-System.git
cd AQI-Prediction-System
pip install pandas numpy scikit-learn xgboost prophet folium matplotlib seaborn tensorflow
jupyter notebook AQI_Prediction.ipynb
```

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `Facebook Prophet` · `TensorFlow/Keras` · `Folium` · `Matplotlib` · `Seaborn`

---

## 📄 License

MIT License
