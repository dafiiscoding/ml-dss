import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pizza_dss.config import FEATURE_COLUMNS, METRICS_DIR
from pizza_dss.business_analysis import (
    customer_preference_tables,
    forecast_metrics,
    forecast_monthly_demand,
    hourly_staffing_plan,
    hypothesis_tests,
    recommendation_rules,
    synthetic_data_audit,
)
from pizza_dss.dashboard_data import load_dashboard_data, make_single_order_frame
from pizza_dss.data_loader import load_dataset
from pizza_dss.decision_rules import get_dss_decision
from pizza_dss.modeling import load_best_model, predict_delay_probability

st.set_page_config(page_title="Pizza Delivery DSS", layout="wide")
st.title("Pizza Delivery Decision Support System")

tabs = st.tabs(
    [
        "Overview",
        "EDA",
        "Customer Behavior",
        "Forecast & Staffing",
        "Model Evaluation",
        "Single Order Demo",
        "Delay Queue",
        "Data Quality",
    ]
)

with tabs[0]:
    df = load_dataset()
    queue = load_dashboard_data()
    cols = st.columns(4)
    cols[0].metric("Orders", f"{len(df):,}")
    cols[1].metric("Delayed rate", f"{df['is_delayed'].mean() * 100:.1f}%")
    cols[2].metric("Restaurants", df["restaurant_name"].nunique())
    cols[3].metric("High priority test orders", int((queue["priority"] == "High").sum()))

    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x="delivery_duration_min", color="is_delayed", nbins=12)
        st.plotly_chart(fig, width="stretch")
    with right:
        traffic = (
            df.groupby("traffic_level")["is_delayed"]
            .mean()
            .reset_index(name="delay_rate")
        )
        fig = px.bar(traffic, x="traffic_level", y="delay_rate")
        st.plotly_chart(fig, width="stretch")

with tabs[1]:
    df = load_dataset()
    col = st.selectbox(
        "Group by",
        ["traffic_level", "pizza_size", "pizza_type", "restaurant_name", "order_hour"],
    )
    rates = (
        df.groupby(col)["is_delayed"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"sum": "delayed", "mean": "delay_rate"})
        .sort_values("delay_rate", ascending=False)
        .reset_index()
    )
    st.dataframe(rates, hide_index=True, width="stretch")
    st.plotly_chart(px.bar(rates.head(20), x=col, y="delay_rate"), width="stretch")

with tabs[2]:
    df = load_dataset()
    prefs = customer_preference_tables(df)
    left, right = st.columns(2)
    with left:
        st.subheader("Size preference")
        st.dataframe(prefs["size_mix"], hide_index=True, width="stretch")
        st.plotly_chart(
            px.bar(prefs["size_mix"], x="pizza_size_label", y="order_share"),
            width="stretch",
        )
    with right:
        st.subheader("Type preference")
        type_mix = prefs["type_mix"].head(12)
        st.dataframe(type_mix, hide_index=True, width="stretch")
        st.plotly_chart(px.bar(type_mix, x="pizza_type", y="order_share"), width="stretch")

    st.subheader("Same pizza type: preferred restaurant")
    st.dataframe(prefs["top_restaurant_by_type"], hide_index=True, width="stretch")

    st.subheader("Recommendation rules")
    st.dataframe(recommendation_rules(df).head(30), hide_index=True, width="stretch")

with tabs[3]:
    df = load_dataset()
    forecast = forecast_monthly_demand(df)
    staffing = hourly_staffing_plan(df)
    metrics = forecast_metrics(forecast)
    cols = st.columns(3)
    cols[0].metric("Forecast backtest rows", metrics["backtest_rows"])
    cols[1].metric("MAE", metrics["mae"])
    cols[2].metric("MAPE", metrics["mape"])

    fig = px.line(
        forecast,
        x="order_period",
        y=["actual_orders", "forecast_orders"],
        markers=True,
        title="Monthly demand and seasonal-naive forecast",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Staffing scenario for 100 orders/day")
    st.dataframe(staffing.query("orders > 0"), hide_index=True, width="stretch")
    st.plotly_chart(
        px.bar(staffing, x="order_hour", y="scenario_orders_per_day"),
        width="stretch",
    )

with tabs[4]:
    dev_path = METRICS_DIR / "model_dev_comparison.csv"
    test_path = METRICS_DIR / "model_test_metrics.csv"
    baseline_path = METRICS_DIR / "baseline_test_metrics.csv"
    if not dev_path.exists() or not test_path.exists():
        st.warning("Run `python -m scripts.run_all` first.")
    else:
        st.subheader("Development set comparison")
        dev = pd.read_csv(dev_path)
        st.dataframe(dev, hide_index=True, width="stretch")
        st.plotly_chart(px.bar(dev, x="model", y="f2"), width="stretch")

        st.subheader("Locked test result")
        st.dataframe(pd.read_csv(test_path), hide_index=True, width="stretch")
        st.subheader("Test baselines")
        st.dataframe(pd.read_csv(baseline_path), hide_index=True, width="stretch")

with tabs[5]:
    model = load_best_model()
    df = load_dataset()
    values = {}
    left, right = st.columns(2)
    with left:
        values["restaurant_name"] = st.selectbox("Restaurant", sorted(df["restaurant_name"].unique()))
        values["location"] = st.selectbox("Location", sorted(df["location"].unique()))
        values["pizza_size"] = st.selectbox("Pizza size", sorted(df["pizza_size"].unique()))
        values["pizza_type"] = st.selectbox("Pizza type", sorted(df["pizza_type"].unique()))
        values["traffic_level"] = st.selectbox("Traffic", ["Low", "Medium", "High"], index=1)
    with right:
        values["payment_method"] = st.selectbox("Payment method", sorted(df["payment_method"].unique()))
        values["payment_category"] = st.selectbox("Payment category", sorted(df["payment_category"].unique()))
        values["is_peak_hour"] = st.checkbox("Peak hour", value=True)
        values["is_weekend"] = st.checkbox("Weekend", value=False)
        values["order_month"] = st.selectbox("Order month", sorted(df["order_month"].unique()))
        values["order_hour"] = st.slider("Order hour", 0, 23, 19)
        values["distance_km"] = st.slider("Distance km", 0.5, 10.0, 5.0, 0.1)
        values["toppings_count"] = st.slider("Toppings", 1, 5, 3)
        values["estimated_duration_min"] = values["distance_km"] * 2.4
        values["traffic_impact"] = {"Low": 1, "Medium": 2, "High": 3}[values["traffic_level"]]
        values["pizza_size_score"] = {"Small": 1, "Medium": 2, "Large": 3, "XL": 4}.get(values["pizza_size"], 2)
        values["pizza_complexity"] = values["toppings_count"] * values["pizza_size_score"]
        values["topping_density"] = values["toppings_count"] / max(values["distance_km"], 0.1)

    order = make_single_order_frame(values)
    prob = predict_delay_probability(model, order)[0]
    decision = get_dss_decision(order.iloc[0], prob)
    cols = st.columns(3)
    cols[0].metric("Delayed probability", f"{decision['delayed_probability']:.1%}")
    cols[1].metric("Risk score", decision["delay_risk_score"])
    cols[2].metric("Priority", decision["priority"])
    st.success(decision["recommended_action"])

with tabs[6]:
    queue = load_dashboard_data()
    with st.sidebar:
        priority = st.selectbox("Priority", ["All", "High", "Medium", "Low"])
        traffic = st.selectbox("Traffic", ["All"] + sorted(queue["traffic_level"].unique()))
    filtered = queue.copy()
    if priority != "All":
        filtered = filtered[filtered["priority"] == priority]
    if traffic != "All":
        filtered = filtered[filtered["traffic_level"] == traffic]
    st.dataframe(filtered, hide_index=True, width="stretch", height=520)
    st.download_button(
        "Download queue",
        filtered.to_csv(index=False).encode("utf-8"),
        "pizza_delay_priority_queue.csv",
        "text/csv",
    )

with tabs[7]:
    df = load_dataset()
    st.subheader("Synthetic/data realism audit")
    synthetic = synthetic_data_audit(df)
    st.dataframe(synthetic, hide_index=True, width="stretch")
    st.plotly_chart(
        px.bar(
            synthetic["severity"].value_counts().reset_index(name="checks"),
            x="severity",
            y="checks",
        ),
        width="stretch",
    )

    st.subheader("Hypothesis tests")
    st.dataframe(hypothesis_tests(df), hide_index=True, width="stretch")
