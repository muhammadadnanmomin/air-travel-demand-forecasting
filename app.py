import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Air Travel Demand Forecast",
    layout="wide"
)

# Load trained LightGBM model
@st.cache_resource
def load_model():
    return joblib.load("models/lightgbm_model.pkl")

model = load_model()

# Load historical data
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/processed/final_timeseries.csv",
        parse_dates=["Activity Period"]
    )
    df = df.set_index("Activity Period")
    df = df.sort_index()
    return df

df = load_data()

# App UI
st.title("Air Travel Demand Forecasting")
st.write(
    """
    This app forecasts **monthly air passenger demand** using a 
    **LightGBM machine learning model** trained on historical data.
    
    **Model features:**
    - Year
    - Month
    - Passenger count (t-1)
    - Passenger count (t-12)
    """
)

# Sidebar controls
st.sidebar.header("Forecast Settings")

forecast_months = st.sidebar.slider(
    "Select forecast horizon (months)",
    min_value=3,
    max_value=36,
    value=12
)

# --------------------------------------------------
# Prepare future dataframe
# --------------------------------------------------
last_date = df.index.max()

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=forecast_months,
    freq="MS"
)

future_df = pd.DataFrame(index=future_dates)

# Calendar features (SAME as training)
future_df["year"] = future_df.index.year
future_df["month"] = future_df.index.month

# Lag features (RAW scale, SAME as training)
future_df["lag_1"] = df["Passenger Count"].iloc[-1]
future_df["lag_12"] = df["Passenger Count"].iloc[-12]

# --------------------------------------------------
# Feature selection
# --------------------------------------------------
FEATURES = ["year", "month", "lag_1", "lag_12"]
future_X = future_df[FEATURES]

# --------------------------------------------------
# Forecast
# --------------------------------------------------
future_df["forecast"] = model.predict(future_X)

# Safety clamp (passengers cannot be negative)
future_df["forecast"] = np.maximum(future_df["forecast"], 0)


# Plot
st.subheader("Passenger Demand Forecast")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df.index[-60:],
    df["Passenger Count"].iloc[-60:],
    label="Historical",
    linewidth=2
)

ax.plot(
    future_df.index,
    future_df["forecast"],
    label="Forecast",
    color="green",
    linewidth=2
)

ax.set_xlabel("Year")
ax.set_ylabel("Passengers")
ax.legend()
ax.grid(True)

st.pyplot(fig)


# Forecast table
st.subheader("Forecasted Passenger Demand")

forecast_table = future_df[["forecast"]].copy()
forecast_table.rename(
    columns={"forecast": "Predicted Passengers"},
    inplace=True
)

forecast_table.index.name = "Month"

st.dataframe(forecast_table.style.format("{:,.0f}"))


# Footer
st.markdown(
    """
    ---
    **Note:**  
    Forecasts are estimates based on historical patterns.  
    Sudden events (pandemics, policy changes) are not explicitly modeled.
    """
)
