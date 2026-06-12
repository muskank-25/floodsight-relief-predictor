# 🌊 FloodSight — Predictive Relief Allocation for Coastal Odisha

> AI-powered flood relief forecasting for coastal Odisha. Predicts ORS, rations, medical kits, tarpaulin & water demand 7–30 days ahead using rainfall, river gauge & population data. Helps NGOs pre-position supplies before disaster strikes. MAPE < 15%.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Challenge](https://img.shields.io/badge/Challenge-1.2%20AI%20%26%20Intelligent%20Systems-purple)]()

---

## 📌 Problem Statement

India's coastal Odisha faces recurring monsoon floods and cyclones every year. NGOs and district disaster authorities still rely on **reactive logistics** — supplies arrive after media coverage peaks, not before communities need them.

Pre-positioning supplies **48 hours before** a flood event:
- Reduces per-beneficiary logistics cost by ~30%
- Expands beneficiary reach from 5,000 → 15,000 people on the same budget
- Saves lives in remote, road-constrained districts

FloodSight solves this by forecasting *where* and *how much* of each critical resource will be needed — up to 30 days in advance.

---

## 🎯 Use Case

**Flood relief resource demand forecasting across 5 coastal Odisha districts:**

| District | Population | Flood Risk | Road Access |
|---|---|---|---|
| Kendrapara | 13,01,761 | Very High | Low |
| Jagatsinghpur | 11,36,604 | High | Medium |
| Puri | 14,98,604 | High | Good |
| Bhadrak | 15,06,337 | High | Medium |
| Balasore | 23,17,419 | Medium-High | Good |

**Resources forecasted:** ORS packets · Dry rations (kg) · Medical kits · Tarpaulin sheets · Drinking water (litres)

---

## 📁 Repository Structure

```
floodsight-relief-predictor/
│
├── Relief_Forecasting_Notebook.ipynb   # Full ML pipeline (10 cells)
├── dashboard.py                        # Streamlit dashboard
├── NGO_Resource_Allocation_Playbook.docx  # Operations manual
│
├── data/
│   └── odisha_relief_data.csv          # Historical dataset (2018–2024)
│
├── models/
│   └── models.pkl                      # Trained GBR models (5 resources)
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

```
Inputs                          Model                    Outputs
──────────────────────          ─────────────────        ──────────────────────
Rainfall (mm) ──────────┐       Gradient Boosting        ORS packets/day
River gauge index ──────┤  →→→  Regressor (sklearn) →→→  Dry rations kg/day
Heatwave index ─────────┤       n_estimators=200          Medical kits/day
Population density ─────┤       max_depth=5               Tarpaulin sheets/day
Road access score ──────┤       learning_rate=0.07        Water litres/day
Historical demand lags ─┘       MAPE < 15% all targets    + Alert level
```

**Feature engineering highlights:**
- Rainfall lag features (1, 3, 7, 14 days)
- 7/14/30-day rolling rainfall averages
- 30-day cumulative rainfall sum
- Cyclical season encoding (sin/cos day-of-year)
- Autoregressive demand lags (1, 7, 14 days)
- Road access buffer multiplier (SPHERE norms)

---

## 📊 Accuracy Results

Evaluated on a held-out 6-month test set (Jul–Dec 2024):

| Resource | MAE | RMSE | MAPE | Status |
|---|---|---|---|---|
| ORS Packets | 13,902 | 20,900 | **8.5%** | ✅ PASS |
| Dry Rations (kg) | 1,194 | 1,776 | **9.0%** | ✅ PASS |
| Medical Kits | 83 | 126 | **13.2%** | ✅ PASS |
| Tarpaulin Sheets | 858 | 1,265 | **10.5%** | ✅ PASS |
| Water (litres) | 44,203 | 65,034 | **8.9%** | ✅ PASS |

> All resources achieve **MAPE < 15%**, beating the challenge benchmark of 20%.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/floodsight-relief-predictor.git
cd floodsight-relief-predictor
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook Relief_Forecasting_Notebook.ipynb
```

The notebook is self-contained — it generates the dataset, trains all models, produces accuracy plots, and runs a 30-day scenario forecast.

### 3. Launch the Dashboard

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser. No configuration needed.

---

## 🖥️ Dashboard Features

| Feature | Description |
|---|---|
| District selector | Switch between all 5 coastal Odisha districts |
| Forecast horizon | Slide between 7 and 30 days |
| Rainfall scenario | Normal / Elevated / Extreme |
| KPI cards | 7-day and total supply requirements per resource |
| Alert banner | 🔴 Critical / 🟡 Elevated / 🟢 Routine with exact action |
| District table | All-district alert overview in one view |
| Historical charts | 2024 actual demand vs forecast overlay |
| CSV export | Download full forecast for procurement team |

---

## 📋 Key Findings

1. **30-day cumulative rainfall** is the single strongest predictor across all resource types
2. **River gauge index with 3-day lag** adds ~12% predictive lift over rainfall alone
3. **Road access score** drives buffer-stock sizing — poor access districts need 40% more pre-positioned stock
4. **Kendrapara and Bhadrak** show highest demand volatility; model confidence is slightly lower here
5. Pre-positioning just **7 days early** covers ~85% of peak event demand

---

## 🌐 Data Sources

| Source | Data | Link |
|---|---|---|
| IMD | Daily rainfall, heatwave alerts | [mausam.imd.gov.in](https://mausam.imd.gov.in) |
| CWC / ISRO Bhuvan | River gauge data | [bhuvan.nrsc.gov.in](https://bhuvan.nrsc.gov.in) |
| NDMA | Incident database, affected population | [ndma.gov.in](https://ndma.gov.in) |
| data.gov.in | District-level demographic data | [data.gov.in](https://data.gov.in) |
| OpenStreetMap | Road network / accessibility | [openstreetmap.org](https://openstreetmap.org) |
| SPHERE Handbook | Humanitarian supply norms | [spherestandards.org](https://spherestandards.org) |

> The repository uses a synthetic dataset generated from real IMD/NDMA distributions. Replace `data/odisha_relief_data.csv` with live data from the above sources for production use.

---

## 📦 Requirements

```
numpy>=1.24
pandas>=1.5
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
streamlit>=1.22
scipy>=1.10
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 📄 Deliverables

| Deliverable | File | Description |
|---|---|---|
| 📓 Notebook | `Relief_Forecasting_Notebook.ipynb` | Reproducible ML pipeline with accuracy report |
| 🖥️ Dashboard | `dashboard.py` | Streamlit app for NGO operations managers |
| 📘 Playbook | `NGO_Resource_Allocation_Playbook.docx` | 9-section operations manual |

---

## 🏗️ Roadmap

- [ ] Integrate live IMD rainfall forecast API for true forward-looking signals
- [ ] Add GeoPandas + Folium choropleth map layer
- [ ] Retrain with LightGBM for ~5% MAPE improvement
- [ ] Add block-level granularity within districts
- [ ] Build NGO feedback portal for post-event actuals upload
- [ ] Docker container for one-click deployment

---

## 📜 License

MIT License — free to use, adapt, and deploy for humanitarian purposes.

---

## 🙏 Acknowledgements

- SPHERE Project for humanitarian supply standards
- NDMA & IMD for open disaster and weather data
- Challenge 1.2 — AI & Intelligent Systems Track

---

*Built for Challenge 1.2 | AI & Intelligent Systems | Coastal Odisha Flood Relief*
