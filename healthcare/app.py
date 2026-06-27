"""
Predictive Forecasting of Care Load & Placement Demand
HHS Unaccompanied Alien Children (UAC) Program — Streamlit Dashboard

Run locally:  streamlit run app.py
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import forecasting as fc

st.set_page_config(
    page_title="UAC Care Load Forecasting",
    page_icon="📊",
    layout="wide",
)

TARGET = fc.TARGET
DATA_PATH = os.path.join(APP_DIR, "clean_data.csv")

ALL_MODELS = list(fc.MODEL_REGISTRY.keys())
MODEL_COLORS = {
    "Naive Persistence": "#94a3b8",
    "Moving Average (7d)": "#a78bfa",
    "SARIMA": "#0ea5e9",
    "Exponential Smoothing": "#f59e0b",
    "Random Forest": "#10b981",
    "Gradient Boosting": "#ef4444",
}


# --------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def run_forecast(model_name: str, df: pd.DataFrame, horizon: int):
    result = fc.fit_and_forecast(model_name, df, horizon)
    return result.mean, result.lower, result.upper


@st.cache_data(show_spinner=False)
def run_backtest(model_name: str, df: pd.DataFrame, horizon: int):
    return fc.walk_forward_backtest(df, model_name, horizon=horizon, n_folds=5)


df = load_data()
last_date = df["date"].max()
last_value = df[TARGET].iloc[-1]

# --------------------------------------------------------------------------
# Sidebar — user controls
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Forecast Controls")

horizon = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=60, value=14, step=1)

selected_models = st.sidebar.multiselect(
    "Models to compare",
    options=ALL_MODELS,
    default=["SARIMA", "Random Forest", "Gradient Boosting"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Capacity Scenario")
default_capacity = int(round(df[TARGET].max() * 1.05 / 10) * 10)
capacity_threshold = st.sidebar.number_input(
    "Shelter / system capacity threshold",
    min_value=0, value=default_capacity, step=50,
    help="Used for breach-probability and surge-lead-time KPIs",
)

st.sidebar.markdown("---")
show_observed_only = st.sidebar.checkbox(
    "Show only originally-reported days in history chart", value=False,
    help="Source data skips most Fridays/Saturdays; unchecked shows the interpolated continuous series."
)

if not selected_models:
    st.warning("Select at least one model from the sidebar to see forecasts.")
    st.stop()

# --------------------------------------------------------------------------
# Header / KPI strip
# --------------------------------------------------------------------------
st.title("🏥 Predictive Forecasting of Care Load & Placement Demand")
st.caption("HHS Unaccompanied Alien Children (UAC) Program — early-warning forecasting dashboard")

with st.spinner("Running models..."):
    forecasts = {}
    for m in selected_models:
        mean, lower, upper = run_forecast(m, df, horizon)
        forecasts[m] = fc.ForecastResult(mean, lower, upper)

primary_model = selected_models[0]
primary_result = forecasts[primary_model]
primary_backtest = run_backtest(primary_model, df, horizon=min(horizon, 14))

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Current HHS Care Load", f"{int(last_value):,}", help=f"As of {last_date.date()}")
k2.metric(
    "Forecast Accuracy",
    f"{fc.forecast_accuracy_pct(primary_backtest['MAPE']):.1f}%",
    help=f"Based on {primary_model} walk-forward backtest (100 − MAPE)",
)
breach_prob = fc.capacity_breach_probability(primary_result, capacity_threshold)
k3.metric("Capacity Breach Probability", f"{breach_prob:.1f}%", help=f"Chance of exceeding {capacity_threshold:,} within the forecast horizon")
lead_time = fc.surge_lead_time(primary_result, capacity_threshold)
k4.metric("Surge Lead Time", f"{lead_time} days" if lead_time else "No breach", help="Days until mean forecast first crosses capacity threshold")
stability = fc.forecast_stability_index(primary_result)
k5.metric("Forecast Stability Index", f"{stability:.1f}/100", help="100 = very stable day-to-day forecast")

st.markdown("---")

# --------------------------------------------------------------------------
# Tabs — core modules
# --------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Care Load Forecast", "🚪 Discharge & Net Pressure", "🔍 Model Comparison", "📋 Data & Methodology",
])

# ---- Tab 1: Future Care Load Forecast Chart ----
with tab1:
    st.subheader("Future Care Load Forecast")

    hist_window = st.slider("History window to display (days)", 30, 365, 90, key="hist_window_1")
    hist_df = df.tail(hist_window)
    if show_observed_only:
        hist_df = hist_df[hist_df["is_observed"]]

    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df[TARGET], mode="lines", name="Historical",
        line=dict(color="#1e293b", width=2),
    ))

    for m in selected_models:
        res = forecasts[m]
        color = MODEL_COLORS.get(m, "#888")
        fig.add_trace(go.Scatter(
            x=future_dates, y=res.mean, mode="lines+markers", name=m,
            line=dict(color=color, width=2, dash="solid"),
        ))
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(res.upper) + list(res.lower[::-1]),
            fill="toself", fillcolor=color, opacity=0.12,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            name=f"{m} CI",
        ))

    fig.add_hline(
        y=capacity_threshold, line_dash="dot", line_color="#dc2626",
        annotation_text="Capacity threshold", annotation_position="top left",
    )
    fig.add_vline(x=last_date, line_dash="dash", line_color="#94a3b8")

    fig.update_layout(
        height=480, hovermode="x unified",
        margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="Children in HHS Care",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Shaded bands show ~95% confidence intervals. Dashed vertical line marks the forecast origin "
        f"({last_date.date()}). Dotted red line is the user-set capacity threshold ({capacity_threshold:,})."
    )

# ---- Tab 2: Discharge demand & net pressure panel ----
with tab2:
    st.subheader("Net Pressure & Discharge Context")
    st.caption(
        "Net pressure = Transfers into HHS care − Discharges out. Positive values signal rising care load; "
        "negative values signal the system is releasing children faster than it receives them."
    )

    pressure_window = st.slider("History window (days)", 30, 365, 120, key="hist_window_2")
    pdf = df.tail(pressure_window)

    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                          vertical_spacing=0.08,
                          subplot_titles=("Transfers In vs. Discharges Out", "Net Pressure (Transfers − Discharges)"))

    fig2.add_trace(go.Scatter(x=pdf["date"], y=pdf["transfers_to_hhs"], name="Transfers to HHS",
                               line=dict(color="#0ea5e9")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=pdf["date"], y=pdf["discharges"], name="Discharges",
                               line=dict(color="#10b981")), row=1, col=1)

    colors = np.where(pdf["net_pressure"] >= 0, "#ef4444", "#10b981")
    fig2.add_trace(go.Bar(x=pdf["date"], y=pdf["net_pressure"], name="Net Pressure",
                          marker_color=colors), row=2, col=1)
    fig2.add_hline(y=0, line_color="#64748b", line_width=1, row=2, col=1)

    fig2.update_layout(height=560, margin=dict(t=40, b=20, l=10, r=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0))
    st.plotly_chart(fig2, use_container_width=True)

    avg_net_pressure_7d = df["net_pressure"].tail(7).mean()
    c1, c2 = st.columns(2)
    c1.metric("Avg Net Pressure (last 7d)", f"{avg_net_pressure_7d:+.1f} children/day")
    c2.metric("Avg Discharges (last 7d)", f"{df['discharges'].tail(7).mean():.1f} children/day")

# ---- Tab 3: Model Selection & Comparison ----
with tab3:
    st.subheader("Model Comparison — Walk-Forward Backtest")
    st.caption(
        "Each model is retrained on a rolling origin and scored on its next "
        f"{min(horizon, 14)}-day forecast across 5 historical folds (strict time-based validation, no random sampling)."
    )

    compare_models = st.multiselect(
        "Models to backtest", options=ALL_MODELS, default=selected_models, key="compare_models"
    )

    if compare_models:
        rows = []
        horizon_curve = {}
        with st.spinner("Backtesting..."):
            for m in compare_models:
                bt = run_backtest(m, df, horizon=min(horizon, 14))
                rows.append({
                    "Model": m,
                    "MAE": round(bt["MAE"], 2),
                    "RMSE": round(bt["RMSE"], 2),
                    "MAPE (%)": round(bt["MAPE"], 2),
                    "Accuracy (%)": round(fc.forecast_accuracy_pct(bt["MAPE"]), 2),
                })
                horizon_curve[m] = bt.get("horizon_errors", {})

        result_df = pd.DataFrame(rows).sort_values("RMSE")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        best_model = result_df.iloc[0]["Model"]
        st.success(f"✅ Best performing model on this backtest: **{best_model}** (lowest RMSE)")

        st.markdown("##### Short vs. Medium-Term Reliability (Horizon Error)")
        fig3 = go.Figure()
        for m, herr in horizon_curve.items():
            if herr:
                fig3.add_trace(go.Scatter(
                    x=list(herr.keys()), y=list(herr.values()), mode="lines+markers",
                    name=m, line=dict(color=MODEL_COLORS.get(m, "#888")),
                ))
        fig3.update_layout(
            height=380, xaxis_title="Forecast day ahead", yaxis_title="Mean Absolute Error",
            margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Rising error with horizon day is expected; a flatter curve indicates a more reliable medium-term model.")
    else:
        st.info("Select at least one model above to run the comparison.")

# ---- Tab 4: Data & methodology ----
with tab4:
    st.subheader("Data & Methodology")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Dataset coverage**")
        st.write(f"- Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        st.write(f"- Total days (continuous): {len(df):,}")
        st.write(f"- Originally reported days: {int(df['is_observed'].sum()):,}")
        st.write(f"- Interpolated (filled) days: {int((~df['is_observed']).sum()):,}")
    with c2:
        st.markdown("**Models implemented**")
        st.write("- Baseline: Naive Persistence, Moving Average")
        st.write("- Statistical: SARIMA, Exponential Smoothing")
        st.write("- Machine Learning: Random Forest, Gradient Boosting")

    st.markdown("**Feature engineering**")
    st.write(
        "Lag features (t-1, t-7, t-14), 7/14-day rolling mean & std, "
        "net pressure (transfers − discharges) signals, and calendar effects "
        "(day-of-week, month, month-start/end flags)."
    )

    st.markdown("**Validation strategy**")
    st.write(
        "Strict time-based train/test split with walk-forward (rolling-origin) backtesting — "
        "models are never trained on future data relative to what they're scored against."
    )

    st.markdown("##### Sample of cleaned data")
    st.dataframe(df.tail(15), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download cleaned dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="uac_clean_data.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Built for the Unified Mentor / HHS UAC Program predictive forecasting project. "
    "Forecasts are statistical estimates for planning purposes only and do not constitute official HHS guidance."
)
