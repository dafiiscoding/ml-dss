"""Phân tích nghiệp vụ: audit dữ liệu synthetic, kiểm định giả thiết, customer
behavior, dự báo nhu cầu/staffing, recommendation và kiểm định xu hướng.

Phục vụ notebook 05. Lưu ý: dữ liệu là synthetic nên kết quả forecast/trend là
**minh hoạ phương pháp**, không phải kết luận kinh doanh. Thuật ngữ (MAPE,
seasonal-naive, Mann-Kendall, Cramér's V…) xem `docs/GLOSSARY.md`.
"""
import json
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pizza_dss.config import ANALYSIS_DATE, FIGURES_DIR, METRICS_DIR, TARGET_COLUMN
from pizza_dss.data_loader import SIZE_SCORE, load_dataset

try:
    from scipy.stats import chi2_contingency, linregress
except Exception:  # pragma: no cover - scipy is listed in requirements.
    chi2_contingency = None
    linregress = None


def _cramers_v(chi2, n, shape):
    r, k = shape
    denom = n * max(min(k - 1, r - 1), 1)
    return math.sqrt(max(chi2, 0) / denom) if denom else 0.0


def _safe_share(series):
    total = series.sum()
    return series / total if total else series * 0


def synthetic_data_audit(df=None):
    df = load_dataset() if df is None else df.copy()
    analysis_date = pd.Timestamp(ANALYSIS_DATE)
    size_formula = df["toppings_count"] * df["pizza_size"].map(SIZE_SCORE)
    ordered = df.sort_values("order_time")
    date_gaps = ordered["order_time"].diff().dropna()
    order_id_numbers = df["order_id"].str.extract(r"ORD(\d+)")[0].dropna().astype(int)
    expected_ids = set(range(order_id_numbers.min(), order_id_numbers.max() + 1))
    observed_ids = set(order_id_numbers.tolist())

    checks = [
        {
            "check": "No missing values",
            "severity": "warning",
            "evidence": f"{int(df.isna().sum().sum())} missing cells across {df.shape[1]} columns",
            "interpretation": "Clean data is useful, but a zero-missing operational file is a synthetic-data signal.",
        },
        {
            "check": "Future timestamp relative to analysis date",
            "severity": "warning" if df["order_time"].max() > analysis_date else "ok",
            "evidence": f"max order_time={df['order_time'].max()}, analysis_date={analysis_date.date()}",
            "interpretation": "The file title says 2024-25, but rows reach July 2026.",
        },
        {
            "check": "Label is deterministic from delivery duration",
            "severity": "critical",
            "evidence": str(bool((df[TARGET_COLUMN] == (df["delivery_duration_min"] > 30)).all())),
            "interpretation": "delivery_duration_min, delay_min and delivery_time must be blocked from predictive features.",
        },
        {
            "check": "Estimated duration formula",
            "severity": "warning",
            "evidence": f"max |estimated_duration_min - 2.4 * distance_km| = {float((df['estimated_duration_min'] - 2.4 * df['distance_km']).abs().max()):.10f}",
            "interpretation": "estimated_duration_min is a deterministic transform of distance_km.",
        },
        {
            "check": "Topping density formula",
            "severity": "warning",
            "evidence": f"max |topping_density - toppings_count / distance_km| = {float((df['topping_density'] - df['toppings_count'] / df['distance_km']).abs().max()):.10f}",
            "interpretation": "topping_density duplicates information already in toppings_count and distance_km.",
        },
        {
            "check": "Pizza complexity formula",
            "severity": "warning",
            "evidence": f"max |pizza_complexity - toppings_count * size_score| = {float((df['pizza_complexity'] - size_formula).abs().max()):.10f}",
            "interpretation": "pizza_complexity is engineered from toppings_count and encoded size.",
        },
        {
            "check": "Traffic impact is only encoded traffic level",
            "severity": "warning",
            "evidence": f"unique traffic_impact values={sorted(df['traffic_impact'].unique().tolist())}",
            "interpretation": "traffic_impact is redundant with traffic_level.",
        },
        {
            "check": "Delivery duration has few discrete values",
            "severity": "warning",
            "evidence": f"{df['delivery_duration_min'].nunique()} unique values: {sorted(df['delivery_duration_min'].unique().tolist())}",
            "interpretation": "Real delivery duration usually has richer minute-level variation.",
        },
        {
            "check": "Order hour support is narrow",
            "severity": "warning",
            "evidence": f"{df['order_hour'].nunique()} unique hours; {(df['order_hour'].between(18, 20).mean() * 100):.1f}% are 18-20h",
            "interpretation": "This is useful for dinner-peak staffing but too clean for full-day demand modeling.",
        },
        {
            "check": "Distance support is discretized",
            "severity": "warning",
            "evidence": f"{df['distance_km'].nunique()} unique distances across {len(df)} rows",
            "interpretation": "Distances look sampled from a short menu of values rather than GPS-derived trips.",
        },
        {
            "check": "Toppings count support is narrow",
            "severity": "warning",
            "evidence": ", ".join(f"{k}:{v}" for k, v in df["toppings_count"].value_counts().sort_index().items()),
            "interpretation": "Toppings are limited to 1-5, with most rows between 2 and 5.",
        },
        {
            "check": "Restaurant distribution is near uniform",
            "severity": "info",
            "evidence": f"min={df['restaurant_name'].value_counts().min()}, max={df['restaurant_name'].value_counts().max()} across 5 brands",
            "interpretation": "Uniform brand counts often appear in generated examples.",
        },
        {
            "check": "Sequential order ids",
            "severity": "info",
            "evidence": f"missing ids inside ORD range={len(expected_ids - observed_ids)}",
            "interpretation": "Sequential ids are acceptable, but they reinforce the generated-data reading.",
        },
        {
            "check": "Repeated time gaps",
            "severity": "warning",
            "evidence": f"top gap={date_gaps.value_counts().index[0]} occurs {int(date_gaps.value_counts().iloc[0])} times",
            "interpretation": "Order timestamps progress with repeated gaps rather than a natural arrival process.",
        },
        {
            "check": "Starts after New Year",
            "severity": "info",
            "evidence": f"min order_time={df['order_time'].min()}",
            "interpretation": "The 2024 series starts on Jan 5; without business context, do not infer a New Year effect.",
        },
    ]
    return pd.DataFrame(checks)


def redundant_feature_audit(df=None):
    df = load_dataset() if df is None else df.copy()
    size_formula = df["toppings_count"] * df["pizza_size_score"]
    rows = [
        {
            "column": "estimated_duration_min",
            "source_or_formula": "2.4 * distance_km",
            "max_abs_error": float((df["estimated_duration_min"] - 2.4 * df["distance_km"]).abs().max()),
            "model_action": "Drop in compact feature set; keep distance_km.",
        },
        {
            "column": "topping_density",
            "source_or_formula": "toppings_count / distance_km",
            "max_abs_error": float((df["topping_density"] - df["toppings_count"] / df["distance_km"]).abs().max()),
            "model_action": "Drop in compact feature set; keep base variables.",
        },
        {
            "column": "pizza_complexity",
            "source_or_formula": "toppings_count * pizza_size_score",
            "max_abs_error": float((df["pizza_complexity"] - size_formula).abs().max()),
            "model_action": "Use for interpretation; compact model uses size_score and toppings_count.",
        },
        {
            "column": "traffic_impact",
            "source_or_formula": "Low=1, Medium=2, High=3",
            "max_abs_error": 0.0,
            "model_action": "Drop in compact feature set; keep traffic_level.",
        },
        {
            "column": "delay_min",
            "source_or_formula": "delivery_duration_min - estimated_duration_min",
            "max_abs_error": float((df["delay_min"] - (df["delivery_duration_min"] - df["estimated_duration_min"])).abs().max()),
            "model_action": "Block as leakage.",
        },
        {
            "column": "delivery_efficiency_min_per_km",
            "source_or_formula": "delivery_duration_min / distance_km",
            "max_abs_error": float((df["delivery_efficiency_min_per_km"] - df["delivery_duration_min"] / df["distance_km"]).abs().max()),
            "model_action": "Block as leakage.",
        },
        {
            "column": "restaurant_avg_time",
            "source_or_formula": "group statistic from delivery duration",
            "max_abs_error": np.nan,
            "model_action": "Block as post-hoc aggregate leakage.",
        },
    ]
    return pd.DataFrame(rows)


def dtype_audit(df=None):
    df = load_dataset() if df is None else df.copy()
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing": [int(df[col].isna().sum()) for col in df.columns],
            "nunique": [int(df[col].nunique(dropna=True)) for col in df.columns],
            "sample_values": [
                ", ".join(map(str, df[col].dropna().drop_duplicates().head(5).tolist()))
                for col in df.columns
            ],
        }
    )


def hypothesis_tests(df=None):
    df = load_dataset() if df is None else df.copy()
    if chi2_contingency is None:
        return pd.DataFrame(
            [
                {
                    "variable": "scipy_missing",
                    "test": "chi-square",
                    "p_value": np.nan,
                    "interpretation": "Install scipy to run hypothesis tests.",
                }
            ]
        )

    work = df.copy()
    top_locations = work["location"].value_counts().head(12).index
    work["location_top12"] = np.where(work["location"].isin(top_locations), work["location"], "Other")
    variables = [
        "payment_method",
        "payment_category",
        "pizza_size",
        "pizza_type",
        "traffic_level",
        "restaurant_name",
        "location_top12",
        "time_segment",
        "complexity_band",
    ]
    rows = []
    for variable in variables:
        table = pd.crosstab(work[variable], work[TARGET_COLUMN])
        chi2, p_value, dof, expected = chi2_contingency(table)
        rows.append(
            {
                "variable": variable,
                "test": "chi-square independence vs is_delayed",
                "rows": int(table.shape[0]),
                "chi2": float(chi2),
                "dof": int(dof),
                "p_value": float(p_value),
                "cramers_v": _cramers_v(chi2, table.to_numpy().sum(), table.shape),
                "min_expected": float(expected.min()),
                "significant_at_0_05": bool(p_value < 0.05),
                "interpretation": (
                    "Evidence of association with delay"
                    if p_value < 0.05
                    else "No strong evidence of association with delay"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["significant_at_0_05", "cramers_v"], ascending=False)


def customer_preference_tables(df=None):
    df = load_dataset() if df is None else df.copy()
    order_total = len(df)

    size_mix = (
        df.groupby(["pizza_size_score", "pizza_size_label", "pizza_size"])[TARGET_COLUMN]
        .agg(["count", "sum", "mean"])
        .rename(columns={"sum": "delayed", "mean": "delay_rate"})
        .reset_index()
        .sort_values("pizza_size_score")
    )
    size_mix["order_share"] = size_mix["count"] / order_total

    type_mix = (
        df.groupby("pizza_type")[TARGET_COLUMN]
        .agg(["count", "sum", "mean"])
        .rename(columns={"sum": "delayed", "mean": "delay_rate"})
        .reset_index()
        .sort_values("count", ascending=False)
    )
    type_mix["order_share"] = type_mix["count"] / order_total

    restaurant_type = (
        df.groupby(["pizza_type", "restaurant_name"])
        .size()
        .reset_index(name="orders")
        .sort_values(["pizza_type", "orders"], ascending=[True, False])
    )
    restaurant_type["share_within_type"] = restaurant_type.groupby("pizza_type")["orders"].transform(
        lambda values: values / values.sum()
    )
    top_restaurant_by_type = restaurant_type.loc[
        restaurant_type.groupby("pizza_type")["orders"].idxmax()
    ].sort_values("orders", ascending=False)

    restaurant_size = (
        df.groupby(["restaurant_name", "pizza_size_label", "pizza_size_score"])
        .size()
        .reset_index(name="orders")
        .sort_values(["restaurant_name", "pizza_size_score"])
    )
    restaurant_size["share_within_restaurant"] = restaurant_size.groupby("restaurant_name")[
        "orders"
    ].transform(lambda values: values / values.sum())

    type_size_matrix = pd.crosstab(df["pizza_type"], df["pizza_size_label"])

    location_summary = (
        df.groupby("location")
        .agg(
            orders=("order_id", "count"),
            delayed=(TARGET_COLUMN, "sum"),
            delay_rate=(TARGET_COLUMN, "mean"),
            avg_distance_km=("distance_km", "mean"),
            top_type=("pizza_type", lambda value: value.value_counts().index[0]),
            top_size=("pizza_size", lambda value: value.value_counts().index[0]),
        )
        .reset_index()
        .sort_values(["orders", "delay_rate"], ascending=False)
    )

    chain_aggregated = (
        df.groupby(["location", "time_segment"])
        .agg(
            orders=("order_id", "count"),
            delayed=(TARGET_COLUMN, "sum"),
            delay_rate=(TARGET_COLUMN, "mean"),
            avg_distance_km=("distance_km", "mean"),
        )
        .reset_index()
        .sort_values(["orders", "delay_rate"], ascending=False)
    )

    return {
        "size_mix": size_mix,
        "type_mix": type_mix,
        "restaurant_type_preference": restaurant_type,
        "top_restaurant_by_type": top_restaurant_by_type,
        "restaurant_size_preference": restaurant_size,
        "type_size_matrix": type_size_matrix.reset_index(),
        "location_summary": location_summary,
        "chain_aggregated_location_time": chain_aggregated,
    }


def monthly_demand(df=None):
    df = load_dataset() if df is None else df.copy()
    monthly = (
        df.groupby("order_period")
        .agg(
            orders=("order_id", "count"),
            delayed=(TARGET_COLUMN, "sum"),
            delay_rate=(TARGET_COLUMN, "mean"),
            avg_distance_km=("distance_km", "mean"),
        )
        .reset_index()
        .sort_values("order_period")
    )
    monthly["month_start"] = pd.to_datetime(monthly["order_period"] + "-01")
    monthly["is_partial_month"] = monthly["month_start"].dt.to_period("M") == df["order_time"].max().to_period("M")
    return monthly


def forecast_monthly_demand(df=None, horizon=6):
    monthly = monthly_demand(df)
    monthly = monthly.sort_values("month_start").reset_index(drop=True)
    lookup = monthly.set_index(monthly["month_start"].dt.to_period("M"))["orders"].to_dict()
    rows = []
    for _, row in monthly.iterrows():
        period = row["month_start"].to_period("M")
        prior_year = period - 12
        forecast = lookup.get(prior_year, np.nan)
        rows.append(
            {
                "order_period": str(period),
                "month_start": row["month_start"].date().isoformat(),
                "actual_orders": int(row["orders"]),
                "forecast_orders": forecast,
                "split": "backtest" if not pd.isna(forecast) else "history_no_prior_year",
                "method": "seasonal_naive_prior_year",
                "is_partial_month": bool(row["is_partial_month"]),
            }
        )

    last_period = monthly["month_start"].max().to_period("M")
    for step in range(1, horizon + 1):
        period = last_period + step
        prior_year = period - 12
        forecast = lookup.get(prior_year, float(monthly["orders"].tail(12).median()))
        rows.append(
            {
                "order_period": str(period),
                "month_start": period.to_timestamp().date().isoformat(),
                "actual_orders": np.nan,
                "forecast_orders": float(forecast),
                "split": "future",
                "method": "seasonal_naive_prior_year",
                "is_partial_month": False,
            }
        )
    forecast_df = pd.DataFrame(rows)
    backtest = forecast_df.dropna(subset=["forecast_orders"])
    backtest = backtest[backtest["split"] == "backtest"].copy()
    if not backtest.empty:
        backtest["abs_error"] = (backtest["actual_orders"] - backtest["forecast_orders"]).abs()
        backtest["ape"] = backtest["abs_error"] / backtest["actual_orders"].replace(0, np.nan)
    return forecast_df


def forecast_metrics(forecast_df):
    backtest = forecast_df[forecast_df["split"] == "backtest"].dropna(subset=["forecast_orders"]).copy()
    if backtest.empty:
        return {"backtest_rows": 0, "mae": None, "mape": None}
    abs_error = (backtest["actual_orders"] - backtest["forecast_orders"]).abs()
    ape = abs_error / backtest["actual_orders"].replace(0, np.nan)
    return {
        "backtest_rows": int(len(backtest)),
        "mae": round(float(abs_error.mean()), 3),
        "mape": round(float(ape.mean()), 3),
        "method": "seasonal_naive_prior_year",
        "warning": "Synthetic and partial-month data make this a planning demo, not a production forecast.",
    }


def hourly_staffing_plan(df=None, daily_order_scenario=100, orders_per_staff_hour=12):
    df = load_dataset() if df is None else df.copy()
    hourly = df.groupby("order_hour").size().reset_index(name="orders")
    all_hours = pd.DataFrame({"order_hour": range(24)})
    hourly = all_hours.merge(hourly, on="order_hour", how="left").fillna({"orders": 0})
    hourly["order_share"] = _safe_share(hourly["orders"])
    hourly["scenario_orders_per_day"] = hourly["order_share"] * daily_order_scenario
    hourly["recommended_staff"] = np.ceil(
        hourly["scenario_orders_per_day"] / orders_per_staff_hour
    ).astype(int)
    hourly["recommended_staff"] = hourly["recommended_staff"].clip(lower=1)
    hourly["time_segment"] = pd.cut(
        hourly["order_hour"],
        bins=[0, 11, 15, 17, 21, 24],
        labels=["Other", "Lunch", "Afternoon", "Dinner", "Late"],
        right=False,
        include_lowest=True,
    ).astype(str)
    return hourly


def recommendation_rules(df=None):
    df = load_dataset() if df is None else df.copy()
    rows = []

    def add_top_rule(group_cols, recommend_col, rule_name, min_context_orders=8):
        grouped = df.groupby(group_cols + [recommend_col]).size().reset_index(name="orders")
        grouped["context_orders"] = grouped.groupby(group_cols)["orders"].transform("sum")
        grouped["confidence"] = grouped["orders"] / grouped["context_orders"]
        top = grouped.loc[grouped.groupby(group_cols)["orders"].idxmax()].copy()
        top = top[top["context_orders"] >= min_context_orders]
        for _, row in top.iterrows():
            context = ", ".join(f"{col}={row[col]}" for col in group_cols)
            rows.append(
                {
                    "rule_type": rule_name,
                    "context": context,
                    "recommendation": f"{recommend_col}={row[recommend_col]}",
                    "context_orders": int(row["context_orders"]),
                    "matched_orders": int(row["orders"]),
                    "confidence": float(row["confidence"]),
                }
            )

    add_top_rule(["restaurant_name"], "pizza_type", "Top pizza type by restaurant")
    add_top_rule(["restaurant_name", "pizza_size"], "pizza_type", "Top pizza type by restaurant and size")
    add_top_rule(["location"], "restaurant_name", "Preferred restaurant by location")
    add_top_rule(["pizza_type"], "restaurant_name", "Most chosen restaurant for same pizza type")
    add_top_rule(["time_segment"], "pizza_size", "Top size by time segment")

    return pd.DataFrame(rows).sort_values(["confidence", "context_orders"], ascending=False)


def monthly_product_trends(df=None):
    df = load_dataset() if df is None else df.copy()
    type_trend = (
        df.groupby(["order_period", "pizza_type"])
        .size()
        .reset_index(name="orders")
        .sort_values(["order_period", "orders"], ascending=[True, False])
    )
    type_trend["share_within_month"] = type_trend.groupby("order_period")["orders"].transform(
        lambda values: values / values.sum()
    )
    size_trend = (
        df.groupby(["order_period", "pizza_size_label"])
        .size()
        .reset_index(name="orders")
        .sort_values(["order_period", "pizza_size_label"])
    )
    size_trend["share_within_month"] = size_trend.groupby("order_period")["orders"].transform(
        lambda values: values / values.sum()
    )
    return type_trend, size_trend


def _monthly_share_history(df, dimension, column):
    month_totals = df.groupby("order_period").size().sort_index()
    categories = sorted(df[column].dropna().unique().tolist())
    index = pd.MultiIndex.from_product(
        [month_totals.index.tolist(), categories], names=["order_period", "category"]
    )
    counts = df.groupby(["order_period", column]).size()
    history = counts.reindex(index, fill_value=0).reset_index(name="orders")
    history["dimension"] = dimension
    history["month_orders"] = history["order_period"].map(month_totals).astype(int)
    history["observed_share"] = history["orders"] / history["month_orders"].replace(0, np.nan)
    history["month_start"] = pd.to_datetime(history["order_period"] + "-01")
    last_period = df["order_time"].max().to_period("M")
    history["is_partial_month"] = history["month_start"].dt.to_period("M") == last_period
    return history


def _trend_stats(history):
    rows = []
    for (dimension, category), group in history.groupby(["dimension", "category"]):
        group = group.sort_values("month_start")
        x = np.arange(len(group), dtype=float)
        y = group["observed_share"].astype(float).to_numpy()
        if len(group) < 3 or np.isclose(y.std(), 0):
            slope = 0.0
            p_value = 1.0
        elif linregress is None:
            slope = float(np.polyfit(x, y, deg=1)[0])
            p_value = np.nan
        else:
            result = linregress(x, y)
            slope = float(result.slope)
            p_value = float(result.pvalue)
        rows.append(
            {
                "dimension": dimension,
                "category": category,
                "trend_slope_per_month": slope,
                "trend_p_value": p_value,
                "trend_is_significant_0_05": bool(False if pd.isna(p_value) else p_value < 0.05),
                "trend_direction": "up" if slope > 0 else "down" if slope < 0 else "flat",
            }
        )
    return pd.DataFrame(rows)


def preference_trend_forecast(df=None, horizon=6):
    """Forecast future size/type shares.

    The method is intentionally simple and transparent: estimate a linear trend
    on monthly shares, then use seasonal-naive category shares for future months.
    In this synthetic file the expected business conclusion is usually "no
    stable preference trend", which is still a useful coursework result.
    """
    df = load_dataset() if df is None else df.copy()
    histories = pd.concat(
        [
            _monthly_share_history(df, "pizza_type", "pizza_type"),
            _monthly_share_history(df, "pizza_size", "pizza_size_label"),
        ],
        ignore_index=True,
    )
    stats = _trend_stats(histories)
    histories = histories.merge(stats, on=["dimension", "category"], how="left")
    histories["split"] = "history"
    histories["forecast_step"] = 0
    histories["forecast_share"] = np.nan
    histories["method"] = "history"
    histories["caveat"] = "Observed share from the generated Kaggle file."

    future_rows = []
    for dimension, group in histories.groupby("dimension"):
        group = group.sort_values("month_start")
        pivot = group.pivot_table(
            index=group["month_start"].dt.to_period("M"),
            columns="category",
            values="observed_share",
            aggfunc="first",
        ).sort_index()
        last_period = pivot.index.max()
        last_shares = pivot.iloc[-1].fillna(0.0)
        for step in range(1, horizon + 1):
            period = last_period + step
            prior_year = period - 12
            if prior_year in pivot.index:
                shares = pivot.loc[prior_year].fillna(0.0)
                method = "seasonal_naive_prior_year_share"
            else:
                shares = last_shares.copy()
                method = "last_observed_share_fallback"
            share_sum = shares.sum()
            if share_sum > 0:
                shares = shares / share_sum
            for category, forecast_share in shares.items():
                stat = stats[(stats["dimension"] == dimension) & (stats["category"] == category)].iloc[0]
                future_rows.append(
                    {
                        "dimension": dimension,
                        "category": category,
                        "order_period": str(period),
                        "month_start": period.to_timestamp(),
                        "orders": np.nan,
                        "month_orders": np.nan,
                        "observed_share": np.nan,
                        "is_partial_month": False,
                        "trend_slope_per_month": stat["trend_slope_per_month"],
                        "trend_p_value": stat["trend_p_value"],
                        "trend_is_significant_0_05": stat["trend_is_significant_0_05"],
                        "trend_direction": stat["trend_direction"],
                        "split": "future",
                        "forecast_step": step,
                        "forecast_share": float(forecast_share),
                        "method": method,
                        "caveat": "Forecast is a method demo; generated random shares do not support a strong business trend claim.",
                    }
                )

    forecast = pd.concat([histories, pd.DataFrame(future_rows)], ignore_index=True)
    forecast["month_start"] = pd.to_datetime(forecast["month_start"]).dt.date.astype(str)
    return forecast.sort_values(["dimension", "category", "split", "order_period"]).reset_index(drop=True)


def mann_kendall_trend(values):
    """Non-parametric Mann-Kendall trend test.

    Returns Kendall's tau, the normal-approx p-value and a verdict. Unlike a plain
    slope, it gives a significance level so we can honestly say "no trend".
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 3:
        return {"n": int(n), "s": 0.0, "tau": float("nan"), "z": float("nan"),
                "p_value": float("nan"), "trend": "insufficient_data"}
    s = 0.0
    for i in range(n - 1):
        s += np.sign(x[i + 1:] - x[i]).sum()
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    tau = s / (0.5 * n * (n - 1))
    if p_value < 0.05 and s > 0:
        trend = "increasing"
    elif p_value < 0.05 and s < 0:
        trend = "decreasing"
    else:
        trend = "no_trend"
    return {"n": int(n), "s": float(s), "tau": float(tau), "z": float(z),
            "p_value": float(p_value), "trend": trend}


def preference_trend_tests(df=None):
    """Mann-Kendall trend test on the monthly share of each size/type category."""
    df = load_dataset() if df is None else df.copy()
    type_trend, size_trend = monthly_product_trends(df)
    rows = []
    for dimension, frame, key in [("pizza_type", type_trend, "pizza_type"),
                                  ("pizza_size", size_trend, "pizza_size_label")]:
        for category, group in frame.sort_values("order_period").groupby(key):
            result = mann_kendall_trend(group["share_within_month"].tolist())
            rows.append({"dimension": dimension, "category": category,
                         "months": result["n"], "tau": round(result["tau"], 4) if result["tau"] == result["tau"] else None,
                         "p_value": round(result["p_value"], 4) if result["p_value"] == result["p_value"] else None,
                         "trend": result["trend"]})
    return pd.DataFrame(rows).sort_values(["dimension", "category"]).reset_index(drop=True)


def forecast_method_comparison(df=None, ma_window=3):
    """Backtest seasonal-naive vs a moving-average forecast for monthly demand."""
    monthly = monthly_demand(df).sort_values("month_start").reset_index(drop=True)
    series = monthly.set_index(monthly["month_start"].dt.to_period("M"))["orders"]
    sn, ma = [], []
    for i, period in enumerate(series.index):
        actual = float(series.iloc[i])
        prior_year = period - 12
        if prior_year in series.index:
            forecast = float(series.loc[prior_year])
            sn.append((abs(actual - forecast), abs(actual - forecast) / actual if actual else np.nan))
        if i >= ma_window:
            forecast = float(series.iloc[i - ma_window:i].mean())
            ma.append((abs(actual - forecast), abs(actual - forecast) / actual if actual else np.nan))

    def _agg(errors, name):
        if not errors:
            return {"method": name, "backtest_points": 0, "mae": None, "mape": None}
        ae = np.array([e[0] for e in errors])
        ape = np.array([e[1] for e in errors])
        return {"method": name, "backtest_points": len(errors),
                "mae": round(float(ae.mean()), 3), "mape": round(float(np.nanmean(ape)), 3)}

    return pd.DataFrame([
        _agg(sn, "seasonal_naive_prior_year"),
        _agg(ma, f"moving_average_{ma_window}"),
    ])


def build_business_artifacts():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    synthetic = synthetic_data_audit(df)
    synthetic.to_csv(METRICS_DIR / "synthetic_data_audit.csv", index=False)
    redundant = redundant_feature_audit(df)
    redundant.to_csv(METRICS_DIR / "redundant_feature_audit.csv", index=False)
    dtype_audit(df).to_csv(METRICS_DIR / "dtype_audit.csv", index=False)
    tests = hypothesis_tests(df)
    tests.round(6).to_csv(METRICS_DIR / "hypothesis_tests.csv", index=False)

    preferences = customer_preference_tables(df)
    for name, table in preferences.items():
        table.round(6).to_csv(METRICS_DIR / f"{name}.csv", index=False)

    monthly = monthly_demand(df)
    monthly.to_csv(METRICS_DIR / "monthly_demand.csv", index=False)
    forecast = forecast_monthly_demand(df)
    forecast.round(6).to_csv(METRICS_DIR / "monthly_demand_forecast.csv", index=False)
    forecast_summary = forecast_metrics(forecast)
    (METRICS_DIR / "forecast_summary.json").write_text(
        json.dumps(forecast_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    staffing = hourly_staffing_plan(df)
    staffing.round(6).to_csv(METRICS_DIR / "hourly_staffing_plan.csv", index=False)

    rules = recommendation_rules(df)
    rules.round(6).to_csv(METRICS_DIR / "recommendation_rules.csv", index=False)

    type_trend, size_trend = monthly_product_trends(df)
    type_trend.round(6).to_csv(METRICS_DIR / "monthly_pizza_type_trend.csv", index=False)
    size_trend.round(6).to_csv(METRICS_DIR / "monthly_pizza_size_trend.csv", index=False)
    preference_forecast = preference_trend_forecast(df)
    preference_forecast.round(6).to_csv(METRICS_DIR / "preference_trend_forecast.csv", index=False)
    preference_trend_tests(df).to_csv(METRICS_DIR / "preference_trend_tests.csv", index=False)
    forecast_method_comparison(df).to_csv(METRICS_DIR / "forecast_method_comparison.csv", index=False)

    summary = {
        "synthetic_warning_count": int((synthetic["severity"].isin(["warning", "critical"])).sum()),
        "hypothesis_tests_significant": int(tests["significant_at_0_05"].sum()) if "significant_at_0_05" in tests else 0,
        "top_size": preferences["size_mix"].sort_values("count", ascending=False).iloc[0]["pizza_size"],
        "top_type": preferences["type_mix"].iloc[0]["pizza_type"],
        "forecast": forecast_summary,
        "staffing_peak_hour": int(staffing.sort_values("scenario_orders_per_day", ascending=False).iloc[0]["order_hour"]),
        "recommendation_rules": int(len(rules)),
        "preference_trend_significant_count": int(
            preference_forecast.loc[
                preference_forecast["split"] == "history",
                ["dimension", "category", "trend_is_significant_0_05"],
            ]
            .drop_duplicates()["trend_is_significant_0_05"]
            .sum()
        ),
        "preference_trend_caveat": "Generated shares are mostly flat; use the forecast as a method demo, not a demand signal.",
    }
    (METRICS_DIR / "business_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _write_figures(df, monthly, forecast, staffing, preferences, synthetic)
    _write_preference_trend_figures(preference_forecast)
    return summary


def _write_figures(df, monthly, forecast, staffing, preferences, synthetic):
    fig, ax = plt.subplots(figsize=(7, 4))
    df["pizza_complexity"].value_counts().sort_index().plot.bar(ax=ax)
    ax.set_title("Pizza complexity distribution")
    ax.set_xlabel("Pizza complexity")
    ax.set_ylabel("Orders")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "complexity_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pd.to_datetime(monthly["month_start"]), monthly["orders"], marker="o", label="Actual")
    future = forecast[forecast["split"] == "future"].copy()
    if not future.empty:
        ax.plot(pd.to_datetime(future["month_start"]), future["forecast_orders"], marker="o", linestyle="--", label="Forecast")
    ax.set_title("Monthly demand and seasonal-naive forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Orders")
    ax.legend()
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monthly_demand_forecast.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(staffing["order_hour"], staffing["scenario_orders_per_day"])
    ax.set_title("Order-hour mix for staffing scenario")
    ax.set_xlabel("Order hour")
    ax.set_ylabel("Scenario orders per day")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "hourly_staffing_plan.png", dpi=160)
    plt.close(fig)

    size_mix = preferences["size_mix"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(size_mix["pizza_size_label"], size_mix["order_share"])
    ax.set_title("Customer preference by pizza size")
    ax.set_xlabel("Pizza size")
    ax.set_ylabel("Order share")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "size_preference.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    severity_order = ["critical", "warning", "info", "ok"]
    counts = synthetic["severity"].value_counts().reindex(severity_order).dropna()
    ax.bar(counts.index, counts.values)
    ax.set_title("Data realism audit flags")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Checks")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "synthetic_data_flags.png", dpi=160)
    plt.close(fig)


def _write_preference_trend_figures(preference_forecast):
    for dimension, filename, title in [
        ("pizza_type", "preference_type_share_forecast.png", "Pizza type share trend and forecast"),
        ("pizza_size", "preference_size_share_forecast.png", "Pizza size share trend and forecast"),
    ]:
        subset = preference_forecast[preference_forecast["dimension"] == dimension].copy()
        history = subset[subset["split"] == "history"].copy()
        future = subset[subset["split"] == "future"].copy()
        if history.empty or future.empty:
            continue

        history["month_start"] = pd.to_datetime(history["month_start"])
        future["month_start"] = pd.to_datetime(future["month_start"])
        history_pivot = history.pivot_table(
            index="month_start", columns="category", values="observed_share", aggfunc="first"
        ).sort_index()
        future_pivot = future.pivot_table(
            index="month_start", columns="category", values="forecast_share", aggfunc="first"
        ).sort_index()

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.stackplot(
            history_pivot.index,
            [history_pivot[column].fillna(0) for column in history_pivot.columns],
            labels=history_pivot.columns,
            alpha=0.72,
        )
        for column in future_pivot.columns:
            ax.plot(
                future_pivot.index,
                future_pivot[column],
                linestyle="--",
                marker="o",
                linewidth=1.4,
                label=f"{column} forecast",
            )
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel("Share within month")
        ax.set_ylim(0, 1)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    print(build_business_artifacts())
