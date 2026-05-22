import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="CryptoCast Dashboard",
    layout="wide"
)

# ==========================================================
# TITLE
# ==========================================================

st.title("🚀 CryptoCast Bitcoin Forecast Dashboard")

st.markdown("""
Deep Learning Bitcoin Forecasting System

Models:
- CNN
- RNN
- LSTM
- Transformer

Forecast Horizons:
- 1 Day
- 3 Day
- 7 Day
""")

# ==========================================================
# LOAD METRICS
# ==========================================================

metrics_df = pd.read_csv(r"C:\Users\Niruban\Documents\Suriya\HCLGUVI\Project_5\metrics_summary.csv")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Dashboard Controls")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["CNN", "RNN", "LSTM", "Transformer"]
)

horizon = st.sidebar.selectbox(
    "Select Horizon",
    ["1d", "3d", "7d"]
)

# ==========================================================
# LOAD PICKLE
# ==========================================================

pickle_file = f"{model_name}_{horizon}.pkl"

if not os.path.exists(pickle_file):
    st.error(f"{pickle_file} not found")
    st.stop()

with open(pickle_file, "rb") as f:
    y_true, y_pred = pickle.load(f)

# ==========================================================
# METRICS SECTION
# ==========================================================

selected_metrics = metrics_df[
    (metrics_df["Model"] == model_name) &
    (metrics_df["Horizon"] == horizon)
]

if len(selected_metrics) > 0:

    row = selected_metrics.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"{row['MAE']:.4f}")
    col2.metric("RMSE", f"{row['RMSE']:.4f}")
    col3.metric("MAPE", f"{row['MAPE']:.2f}%")
    col4.metric("R2", f"{row['R2']:.4f}")

# ==========================================================
# FORECAST CHART
# ==========================================================

st.subheader(f"{model_name} - {horizon} Forecast")

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(y_true, label="Actual Price")
ax.plot(y_pred, label="Predicted Price")

ax.set_title(f"{model_name} {horizon} Prediction")
ax.set_xlabel("Time")
ax.set_ylabel("Scaled Price")

ax.legend()
ax.grid(True)

st.pyplot(fig)

# ==========================================================
# ERROR DISTRIBUTION
# ==========================================================

st.subheader("Prediction Error Distribution")

errors = np.array(y_true) - np.array(y_pred)

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.hist(errors, bins=30)

ax2.set_title("Prediction Errors")
ax2.set_xlabel("Error")
ax2.set_ylabel("Frequency")

ax2.grid(True)

st.pyplot(fig2)

# ==========================================================
# COMPARISON TABLE
# ==========================================================

st.subheader("All Model Comparison")

st.dataframe(metrics_df)

# ==========================================================
# BEST MODEL
# ==========================================================

st.subheader("🏆 Best Models by R2")

best_models = metrics_df.sort_values(
    by="R2",
    ascending=False
)

st.dataframe(best_models)

# ==========================================================
# DOWNLOAD METRICS
# ==========================================================

csv = metrics_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Metrics CSV",
    data=csv,
    file_name="crypto_metrics.csv",
    mime="text/csv"
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    "CryptoCast Deep Learning Forecasting Dashboard | PyTorch + Streamlit"
)