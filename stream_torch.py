import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import os

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="CryptoCast Dashboard",
    layout="wide"
)

# outputs folder written by crypto_trade.py (same folder as this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODELS = ["CNN", "RNN", "LSTM", "Transformer"]
HORIZONS = ["1d", "3d", "7d"]

# ==========================================================
# TITLE
# ==========================================================

st.title("🚀 CryptoCast Bitcoin Forecast Dashboard")

st.markdown("""
Deep Learning Bitcoin Forecasting System: **CNN · RNN · LSTM · Transformer**,
compared against a **naive baseline** ("price in *h* days = today's price").
Forecast horizons: **1 day · 3 days · 7 days**. All prices and errors are in **USD**.
""")

# ==========================================================
# LOAD METRICS
# ==========================================================

metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")

if not os.path.exists(metrics_path):
    st.error("metrics_summary.csv not found. Run `python crypto_trade.py` first.")
    st.stop()

metrics_df = pd.read_csv(metrics_path)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Dashboard Controls")

model_name = st.sidebar.selectbox("Select Model", MODELS, index=2)

horizon = st.sidebar.selectbox("Select Horizon", HORIZONS)

show_naive = st.sidebar.checkbox("Show naive baseline on chart", value=True)

last_n = st.sidebar.slider("Days to show on chart", 30, 400, 400, step=10)

# ==========================================================
# LOAD PICKLE
# ==========================================================

pickle_file = os.path.join(OUTPUT_DIR, f"{model_name}_{horizon}.pkl")

if not os.path.exists(pickle_file):
    st.error(f"{pickle_file} not found")
    st.stop()

with open(pickle_file, "rb") as f:
    result = pickle.load(f)

dates = pd.to_datetime(result["dates"])
y_true = np.array(result["y_true"])
y_pred = np.array(result["y_pred"])
naive = np.array(result["naive"])

# ==========================================================
# METRICS SECTION
# ==========================================================

selected = metrics_df[(metrics_df["Model"] == model_name) & (metrics_df["Horizon"] == horizon)]
naive_row = metrics_df[(metrics_df["Model"] == "Naive") & (metrics_df["Horizon"] == horizon)]

if len(selected) > 0 and len(naive_row) > 0:

    row = selected.iloc[0]
    nrow = naive_row.iloc[0]

    st.subheader(f"{model_name} · {horizon} · test-set metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    # delta = difference vs naive; green means the model is better (lower error)
    col1.metric("MAE", f"${row['MAE']:,.0f}", f"{row['MAE'] - nrow['MAE']:+,.0f} vs naive", delta_color="inverse")
    col2.metric("RMSE", f"${row['RMSE']:,.0f}", f"{row['RMSE'] - nrow['RMSE']:+,.0f} vs naive", delta_color="inverse")
    col3.metric("MAPE", f"{row['MAPE']:.2f}%", f"{row['MAPE'] - nrow['MAPE']:+.2f} pts vs naive", delta_color="inverse")
    col4.metric("R²", f"{row['R2']:.3f}")
    col5.metric("Direction accuracy", f"{row['DirAcc']:.1f}%", help="% of days the model got up/down right")

    skill = row["Skill_vs_Naive_%"]
    if skill > 0:
        st.success(f"{model_name} reduces RMSE by {skill:.1f}% compared with the naive baseline at {horizon}.")
    else:
        st.warning(f"{model_name} does not beat the naive baseline at {horizon} (RMSE {skill:+.1f}%).")

# ==========================================================
# FORECAST CHART
# ==========================================================

st.subheader(f"{model_name} - {horizon} Forecast vs Actual (USD)")

sl = slice(-last_n, None)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(dates[sl], y_true[sl], label="Actual Price", color="black", linewidth=2)
ax.plot(dates[sl], y_pred[sl], label=f"{model_name} Predicted", linewidth=1.5)
if show_naive:
    ax.plot(dates[sl], naive[sl], label="Naive (today's price)", color="gray", linestyle="--", linewidth=1)

ax.set_title(f"{model_name} {horizon} Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

ax.legend()
ax.grid(True, alpha=0.4)

st.pyplot(fig)

st.caption("The prediction lines lag behind the actual price by about the horizon: "
           "the models mostly learn that the future price stays close to today's.")

# ==========================================================
# PRICE COMPARISON TABLE
# ==========================================================

st.subheader("Price comparison: actual vs predicted")

comp = pd.DataFrame({
    "Date": dates.date,
    "Actual ($)": y_true,
    f"{model_name} ($)": y_pred,
    "Naive ($)": naive,
    f"{model_name} error ($)": y_pred - y_true,
    f"{model_name} error (%)": (y_pred - y_true) / y_true * 100,
})

st.dataframe(
    comp.tail(15).iloc[::-1].style.format({
        "Actual ($)": "{:,.0f}", f"{model_name} ($)": "{:,.0f}", "Naive ($)": "{:,.0f}",
        f"{model_name} error ($)": "{:+,.0f}", f"{model_name} error (%)": "{:+.2f}%",
    }),
    width="stretch",
    hide_index=True,
)

# ==========================================================
# ERROR DISTRIBUTION
# ==========================================================

st.subheader("Prediction Error Distribution (%)")

errors_pct = (y_pred - y_true) / y_true * 100
naive_errors_pct = (naive - y_true) / y_true * 100

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.hist(errors_pct, bins=40, alpha=0.6, label=f"{model_name} (std {errors_pct.std():.2f}%)")
ax2.hist(naive_errors_pct, bins=40, alpha=0.4, label=f"Naive (std {naive_errors_pct.std():.2f}%)")
ax2.axvline(0, color="black", linewidth=1)

ax2.set_title("Prediction Errors")
ax2.set_xlabel("Error (% of actual price)")
ax2.set_ylabel("Frequency")
ax2.legend()

ax2.grid(True, alpha=0.4)

st.pyplot(fig2)

# ==========================================================
# COMPARISON TABLES
# ==========================================================

st.subheader("All Model Comparison")

order = ["Naive"] + MODELS

c1, c2 = st.columns(2)

with c1:
    st.markdown("**MAPE % (lower is better)**")
    mape = metrics_df.pivot(index="Model", columns="Horizon", values="MAPE").loc[order, HORIZONS]
    st.dataframe(mape.style.format("{:.2f}").highlight_min(axis=0, color="#c8e6c9"), width="stretch")

with c2:
    st.markdown("**RMSE skill vs naive % (positive = beats naive)**")
    skill_tbl = metrics_df[metrics_df["Model"] != "Naive"].pivot(
        index="Model", columns="Horizon", values="Skill_vs_Naive_%").loc[MODELS, HORIZONS]
    st.dataframe(skill_tbl.style.format("{:+.1f}").highlight_max(axis=0, color="#c8e6c9"), width="stretch")

st.markdown("**Full metrics**")
st.dataframe(metrics_df.round(3), width="stretch", hide_index=True)

# ==========================================================
# BEST MODEL
# ==========================================================

st.subheader("🏆 Best Model per Horizon (by RMSE)")

best = (metrics_df[metrics_df["Model"] != "Naive"]
        .sort_values("RMSE")
        .groupby("Horizon", sort=False).head(1)
        .set_index("Horizon").loc[HORIZONS, ["Model", "MAE", "RMSE", "MAPE", "Skill_vs_Naive_%"]])

st.dataframe(best.style.format({"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "MAPE": "{:.2f}%",
                                "Skill_vs_Naive_%": "{:+.1f}%"}), width="stretch")

# ==========================================================
# NEXT FORECAST
# ==========================================================

next_path = os.path.join(OUTPUT_DIR, "next_forecast.csv")

if os.path.exists(next_path):
    st.subheader("🔮 Next forecast from the latest 60 days")
    nf = pd.read_csv(next_path)
    st.caption(f"Last known close: {nf['Last_Date'].iloc[0]} · ${nf['Last_Close'].iloc[0]:,.2f}")
    table = nf.pivot(index="Model", columns="Horizon", values="Forecast").loc[MODELS, HORIZONS]
    st.dataframe(table.style.format("${:,.0f}"), width="stretch")

# ==========================================================
# DOWNLOAD METRICS
# ==========================================================

d1, d2 = st.columns(2)

d1.download_button(
    label="📥 Download Metrics CSV",
    data=metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="crypto_metrics.csv",
    mime="text/csv"
)

comp_path = os.path.join(OUTPUT_DIR, "price_comparison.csv")
if os.path.exists(comp_path):
    with open(comp_path, "rb") as f:
        d2.download_button(
            label="📥 Download Price Comparison CSV",
            data=f.read(),
            file_name="price_comparison.csv",
            mime="text/csv"
        )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    "CryptoCast Deep Learning Forecasting Dashboard | PyTorch + Streamlit"
)
