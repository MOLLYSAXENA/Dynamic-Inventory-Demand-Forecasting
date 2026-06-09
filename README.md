# 📊 Walmart Inventory Forecasting Dashboard

### 🔗 Live Project Link: [https://dynamic-inventory-demand-forecasting-rbeovu4dbzmr6qruruzpvt.streamlit.app/]

**Developed by:** Molly Saxena (CSE-AIML, Batch of 2027) | Inderprastha Engineering College | AKTU  

---

## 📌 Project Overview
An end-to-end Machine Learning web application that predicts short-term product demand and automates safety stock recommendations using the **Walmart M5 Dataset**. 

---

## 🧠 The Core Challenge: Fixing Autoregressive Drift

A standard single-model approach fails over a 30-day horizon because it uses its own *predicted* future values as features for the next day, causing errors to compound exponentially. 

* **The Problem:** My initial recursive baseline crashed with a **406.48% MAPE**.
* **The Solution:** I restructured the backend into a **Direct Multi-Step Forecasting Framework** by training an **Ensemble of 30 separate XGBoost Regressors**. Each model is uniquely trained to predict exactly one specific day ahead ($t+1 \dots t+30$) using only true historical data, completely stopping the error loop.

### 🏆 Performance Improvement
| Forecasting Strategy | Mean Absolute Percentage Error (MAPE) | Operational Accuracy Score |
| :--- | :---: | :---: |
| Baseline Recursive XGBoost | 406.48% | -306.48% |
| **Direct Multi-Step XGBoost Ensemble** | **17.51%** | **82.49%** |

---

## 🛠️ Tech Stack & Implementation

* **Machine Learning:** Python, XGBoost Regressor, Scikit-Learn
* **Feature Engineering:** Calendar features (Month, Day of week, Year), Historical Lags (`lag_1`, `lag_2`, `lag_7`), and Rolling Statistical Windows (`rolling_mean_7`).
* **Frontend Dashboard:** Streamlit, Matplotlib (Interactive demand analytics plots).
* **Supply Chain Logic:** Built-in risk buffers (1.2x, 1.5x, 2.0x) to dynamically calculate safety stock and prevent stockouts.

---

## 📂 Project Structure
* `app.py` - Streamlit application UI and execution logic.
* `requirements.txt` - Python dependency manifest.
* `walmart_cleaned_data.csv` - Preprocessed historical dataset.
* `xgboost_walmart_ensemble.pkl` - Packaged dictionary bundle containing the 30 trained models.

---

## 🚀 Local Setup

1. **Clone Repo:**
   ```bash
   git clone [https://github.com/MOLLYSAXENA/Dynamic-Inventory-Demand-Forecasting.git](https://github.com/MOLLYSAXENA/Dynamic-Inventory-Demand-Forecasting.git)
   cd Dynamic-Inventory-Demand-Forecasting
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run App:**
  ```bash
  streamlit run app.py
