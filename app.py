import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 1. Global Page Configuration
st.set_page_config(page_title="Advanced Inventory Dashboard", layout="wide")

st.title("📊 Dynamic Inventory Demand Forecasting Dashboard")
st.markdown("### Developed by Molly Saxena | CSE-AIML Batch of 2027")
st.write("This interactive system leverages an optimized Direct Multi-Step XGBoost Ensemble (82.49% Operational Accuracy) to forecast short-term product demand using the historical Walmart M5 dataset.")

st.markdown("---")

# 2. Sidebar Control Panel Layout
st.sidebar.header("🕹️ Operational Controls")
forecast_horizon = st.sidebar.slider("Select Forecast Horizon (Days):", min_value=1, max_value=30, value=30)
safety_stock_factor = st.sidebar.selectbox("Select Safety Stock Risk Buffer:", ["Low Risk Buffer (1.2x)", "Medium Risk Buffer (1.5x)", "High Risk Buffer (2.0x)"])

# Mapping risk buffer selections to numerical multipliers
buffer_mapping = {"Low Risk Buffer (1.2x)": 1.2, "Medium Risk Buffer (1.5x)": 1.5, "High Risk Buffer (2.0x)": 2.0}
multiplier = buffer_mapping[safety_stock_factor]

# 3. Optimizing Data & Ensemble Loading Framework
@st.cache_resource
def load_production_pipeline():
    """Loads historical sales records and the multi-model production dictionary package."""
    historical_data = pd.read_csv("walmart_cleaned_data.csv")
    historical_data['ds'] = pd.to_datetime(historical_data['ds'])
    
    with open("xgboost_walmart_ensemble.pkl", "rb") as file_pointer:
        package = pickle.load(file_pointer)
        
    return historical_data, package

try:
    # Trigger data payload load
    df_history, ml_package = load_production_pipeline()
    
    st.success("✅ Operational Intelligence System Online: Direct Multi-Step XGBoost Ensemble Package Loaded.")
    
    # Extract structural items from the packaged dictionary
    ensemble_models = ml_package["models"]
    base_features = ml_package["features"]
    seed_features = ml_package["last_known_features"]
    
    # 4. Generate Dynamic Multi-Step Predictions
    predicted_values = []
    future_dates = pd.date_range(start=df_history['ds'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    
    # Query each day-specific model using the last known stable real features data point
    for day in range(1, forecast_horizon + 1):
        model_key = f"day_{day}"
        if model_key in ensemble_models:
            prediction = ensemble_models[model_key].predict(seed_features)[0]
            # Ensure predictions do not fall below zero (impossible in physical sales)
            predicted_values.append(max(0.0, prediction))
        else:
            predicted_values.append(0.0)
            
    # Assemble predictions into a structural dataframe
    future_forecast = pd.DataFrame({
        'Target Date': future_dates,
        'Expected Units': np.round(predicted_values, 2)
    })
    
    # Create simple lower/upper boundary conditions for visualization context
    future_forecast['Minimum Threshold'] = np.maximum(0.0, future_forecast['Expected Units'] * 0.85)
    future_forecast['Maximum Peak Threshold'] = future_forecast['Expected Units'] * 1.15
    
    # 5. Core Supply Chain KPI Scorecards
    total_projected_demand = int(np.ceil(future_forecast['Expected Units'].sum()))
    recommended_safety_stock = int(np.ceil(total_projected_demand * multiplier))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="📊 Target Forecast Horizon Window", value=f"{forecast_horizon} Days")
    with col2:
        st.metric(label="📈 Total Projected Demand (Units)", value=f"{total_projected_demand} Units")
    with col3:
        st.metric(label="🛡️ Recommended Safety Stock", value=f"{recommended_safety_stock} Units", delta=f"+{recommended_safety_stock - total_projected_demand} Buffer units")

    st.markdown("### 📈 Demand Trajectory Optimization Plot")
    
    # 6. Interactive Optimization Plot Layout
    fig, ax = plt.subplots(figsize=(12, 5))
    # Slice the last 60 days of historical sales context to keep chart focused
    ax.plot(df_history['ds'].tail(60), df_history['y'].tail(60), label='Historical Real Sales', color='black', alpha=0.7, marker='o')
    # Plot upcoming forecast projections derived via XGBoost
    ax.plot(future_forecast['Target Date'], future_forecast['Expected Units'], label='XGBoost Expected Demand Pathway', color='#2ca02c', linewidth=2.5, linestyle='--')
    # Fill background error margins
    ax.fill_between(future_forecast['Target Date'], future_forecast['Minimum Threshold'], future_forecast['Maximum Peak Threshold'], color='#2ca02c', alpha=0.12, label='Model Boundary (±15%)')
    
    ax.set_title("Forward-Looking Production Demand Vector Analytics (Ensemble Mode)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Operational Timeline", fontsize=10)
    ax.set_ylabel("Quantity (Units)", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)
    
    # 7. Tabular Raw Data Expansion View
    with st.expander("🔍 View Raw Forecast Pipeline Execution Grid"):
        st.dataframe(future_forecast, use_container_width=True)

except FileNotFoundError:
    st.error("🚨 Critical Error: 'xgboost_walmart_ensemble.pkl' or 'walmart_cleaned_data.csv' files were not discovered in the current environment directory. Please check asset placements.")