"""Sinh hàng đợi ưu tiên cho dashboard: chạy mô hình khoá trên test, gắn Risk
Score/Priority/Action cho từng đơn rồi sắp theo rủi ro giảm dần.

Là cầu nối giữa mô hình (modeling) và DSS (decision_rules) cho Streamlit/Power BI.
Thuật ngữ xem `docs/GLOSSARY.md`.
"""
import pandas as pd
import matplotlib.pyplot as plt

from pizza_dss.config import (
    FIGURES_DIR,
    METRICS_DIR,
    PRIORITY_LOW_MAX,
    PRIORITY_MEDIUM_MAX,
    PROCESSED_DATA_DIR,
)
from pizza_dss.data_loader import load_processed_splits
from pizza_dss.decision_rules import (
    RISK_COMPONENT_WEIGHTS,
    calculate_delay_risk_components,
    risk_component_policy_spec,
    get_dss_decision,
)
from pizza_dss.modeling import load_best_model, predict_delay_probability

DASHBOARD_QUEUE_PATH = PROCESSED_DATA_DIR / "delay_priority_queue.csv"
PRIORITY_ORDER = ["Low", "Medium", "High"]


def build_dashboard_data():
    _, _, test_df = load_processed_splits()
    model = load_best_model()
    probabilities = predict_delay_probability(model, test_df)
    rows = []
    for idx, (_, order) in enumerate(test_df.iterrows()):
        probability = float(probabilities[idx])
        decision = get_dss_decision(order, probability)
        components = calculate_delay_risk_components(order, probability)
        component_values = {
            f"{name}_score": round(float(value), 4)
            for name, value in components.items()
        }
        component_contributions = {
            f"{name}_contribution": round(float(value * RISK_COMPONENT_WEIGHTS[name]), 4)
            for name, value in components.items()
        }
        rows.append(
            {
                "order_id": order["order_id"],
                "restaurant_name": order["restaurant_name"],
                "location": order["location"],
                "pizza_size": order["pizza_size"],
                "pizza_type": order["pizza_type"],
                "traffic_level": order["traffic_level"],
                "distance_km": order["distance_km"],
                "order_hour": order["order_hour"],
                "true_is_delayed": bool(order["is_delayed"]),
                **decision,
                **component_values,
                **component_contributions,
            }
        )
    queue = pd.DataFrame(rows).sort_values(
        "delay_risk_score", ascending=False
    ).reset_index(drop=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    queue.to_csv(DASHBOARD_QUEUE_PATH, index=False)
    write_dss_artifacts(queue)
    return queue


def load_dashboard_data():
    if not DASHBOARD_QUEUE_PATH.exists():
        return build_dashboard_data()
    return pd.read_csv(DASHBOARD_QUEUE_PATH)


def risk_calibration_table(queue):
    work = queue.copy()
    work["risk_band"] = pd.Categorical(work["priority"], categories=PRIORITY_ORDER, ordered=True)
    grouped = (
        work.groupby("risk_band", observed=False)
        .agg(
            orders=("order_id", "count"),
            actual_delayed=("true_is_delayed", "sum"),
            actual_delay_rate=("true_is_delayed", "mean"),
            mean_risk_score=("delay_risk_score", "mean"),
            mean_predicted_probability=("delayed_probability", "mean"),
        )
        .reset_index()
    )
    grouped["actual_delay_rate"] = grouped["actual_delay_rate"].fillna(0.0)
    grouped["mean_risk_score"] = grouped["mean_risk_score"].fillna(0.0)
    grouped["mean_predicted_probability"] = grouped["mean_predicted_probability"].fillna(0.0)
    grouped["monotonic_from_previous"] = (
        grouped["actual_delay_rate"].diff().fillna(0.0) >= -1e-12
    )
    return grouped


def priority_threshold_sensitivity(queue, scenarios=None):
    scenarios = scenarios or [
        (30, 60, "tighter_capacity"),
        (PRIORITY_LOW_MAX, PRIORITY_MEDIUM_MAX, "locked_policy"),
        (40, 70, "looser_capacity"),
    ]
    rows = []
    for low_cut, high_cut, scenario in scenarios:
        band = pd.cut(
            queue["delay_risk_score"],
            [-1, low_cut, high_cut, 1000],
            labels=PRIORITY_ORDER,
        )
        work = queue.assign(priority_scenario=band)
        counts = work["priority_scenario"].value_counts().reindex(PRIORITY_ORDER).fillna(0).astype(int)
        high = work[work["priority_scenario"] == "High"]
        rows.append(
            {
                "scenario": scenario,
                "low_cut": int(low_cut),
                "high_cut": int(high_cut),
                "low_orders": int(counts["Low"]),
                "medium_orders": int(counts["Medium"]),
                "high_orders": int(counts["High"]),
                "high_actual_delayed": int(high["true_is_delayed"].sum()),
                "high_delay_rate": float(high["true_is_delayed"].mean()) if len(high) else 0.0,
                "captured_delayed_share": (
                    float(high["true_is_delayed"].sum() / queue["true_is_delayed"].sum())
                    if queue["true_is_delayed"].sum()
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def risk_component_breakdown(queue, top_n=12):
    contribution_columns = [f"{name}_contribution" for name in RISK_COMPONENT_WEIGHTS]
    rows = []
    for column in contribution_columns:
        component = column.removesuffix("_contribution")
        rows.append(
            {
                "section": "overall_mean_contribution",
                "order_id": "ALL",
                "component": component,
                "weight": RISK_COMPONENT_WEIGHTS[component],
                "mean_contribution": float(queue[column].mean()),
                "mean_component_score": float(queue[f"{component}_score"].mean()),
            }
        )
    top = queue.sort_values("delay_risk_score", ascending=False).head(top_n)
    for _, row in top.iterrows():
        for column in contribution_columns:
            component = column.removesuffix("_contribution")
            rows.append(
                {
                    "section": "top_priority_orders",
                    "order_id": row["order_id"],
                    "component": component,
                    "weight": RISK_COMPONENT_WEIGHTS[component],
                    "mean_contribution": float(row[column]),
                    "mean_component_score": float(row[f"{component}_score"]),
                }
            )
    return pd.DataFrame(rows)


def write_dss_artifacts(queue):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    calibration = risk_calibration_table(queue)
    sensitivity = priority_threshold_sensitivity(queue)
    breakdown = risk_component_breakdown(queue)
    policy = pd.DataFrame(risk_component_policy_spec())
    calibration.round(6).to_csv(METRICS_DIR / "risk_calibration.csv", index=False)
    sensitivity.round(6).to_csv(METRICS_DIR / "priority_threshold_sensitivity.csv", index=False)
    breakdown.round(6).to_csv(METRICS_DIR / "risk_component_breakdown.csv", index=False)
    policy.to_csv(METRICS_DIR / "risk_component_policy_spec.csv", index=False)
    _write_dss_figures(calibration, sensitivity, breakdown)
    return {
        "risk_calibration_rows": int(len(calibration)),
        "priority_threshold_sensitivity_rows": int(len(sensitivity)),
        "risk_component_breakdown_rows": int(len(breakdown)),
    }


def _write_dss_figures(calibration, sensitivity, breakdown):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(calibration["risk_band"].astype(str), calibration["actual_delay_rate"])
    ax.plot(
        calibration["risk_band"].astype(str),
        calibration["mean_predicted_probability"],
        marker="o",
        color="red",
        label="Mean predicted P(delay)",
    )
    ax.set_ylim(0, 1)
    ax.set_title("Risk calibration by priority band")
    ax.set_xlabel("Priority band")
    ax.set_ylabel("Delay rate / probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "risk_calibration.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sensitivity["scenario"], sensitivity["high_orders"])
    ax.set_title("High-priority volume by threshold scenario")
    ax.set_xlabel("Threshold scenario")
    ax.set_ylabel("High priority orders")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "priority_threshold_sensitivity.png", dpi=160)
    plt.close(fig)

    overall = breakdown[breakdown["section"] == "overall_mean_contribution"].copy()
    overall = overall.sort_values("mean_contribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(overall["component"], overall["mean_contribution"])
    ax.set_title("Mean Risk Score contribution by component")
    ax.set_xlabel("Score contribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "risk_component_breakdown.png", dpi=160)
    plt.close(fig)


def make_single_order_frame(values):
    return pd.DataFrame([values])


if __name__ == "__main__":
    print(build_dashboard_data().head().to_string(index=False))
