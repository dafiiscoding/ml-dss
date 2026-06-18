"""Bài toán phân công/vận tải: gán nhóm đơn High-priority cho tài xế sao cho tổng
chi phí nhỏ nhất (Hungarian; greedy fallback nếu thiếu scipy).

Lưu ý: đơn hàng là thật nhưng **đội tài xế/capacity là giả lập** (dataset không
có bảng tài xế). Thuật ngữ (assignment, Hungarian, greedy…) xem `docs/GLOSSARY.md`.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pizza_dss.config import FIGURES_DIR, METRICS_DIR, RANDOM_STATE
from pizza_dss.dashboard_data import load_dashboard_data

TRANSPORT_COST_POLICY = [
    {
        "term": "travel_minutes",
        "formula": "distance_km * 3.2 / driver_speed_factor",
        "source": "distance_km từ đơn thật; speed_factor từ fleet giả lập",
        "effect": "Đơn xa hoặc tài xế chậm hơn có cost cao hơn.",
    },
    {
        "term": "priority_penalty",
        "formula": "0.35 * (100 - delay_risk_score)",
        "source": "delay_risk_score từ queue DSS",
        "effect": "Đơn risk càng cao thì penalty càng thấp để được ưu tiên gán.",
    },
    {
        "term": "high_traffic_penalty",
        "formula": "6 nếu traffic_level == 'High', ngược lại 0",
        "source": "traffic_level từ đơn thật",
        "effect": "Traffic cao làm tăng cost vì cần thêm buffer vận hành.",
    },
    {
        "term": "same_location_bonus",
        "formula": "-8 nếu order.location == driver.base_location, ngược lại 0",
        "source": "location từ đơn thật; base_location từ fleet giả lập",
        "effect": "Ưu tiên tài xế cùng khu vực để giảm chi phí proxy.",
    },
    {
        "term": "cost_floor",
        "formula": "max(travel + priority + traffic - same_location_bonus, 0)",
        "source": "quy tắc an toàn của mô hình cost",
        "effect": "Đảm bảo cost không âm khi có bonus cùng khu vực.",
    },
]


def transport_cost_policy_spec():
    """Return the transparent cost formula used by the assignment scenario."""
    return [item.copy() for item in TRANSPORT_COST_POLICY]


def build_driver_fleet():
    """Return a small deterministic fleet used for the coursework scenario.

    The Kaggle dataset does not contain drivers, capacity, wages, or depot
    locations. This fleet is therefore a transparent simulation layer used only
    to demonstrate a prescriptive DSS/transportation-problem step.
    """
    return pd.DataFrame(
        [
            {"driver_id": "D01", "base_location": "New York, NY", "capacity": 2, "speed_factor": 1.00},
            {"driver_id": "D02", "base_location": "Los Angeles, CA", "capacity": 2, "speed_factor": 0.95},
            {"driver_id": "D03", "base_location": "Chicago, IL", "capacity": 2, "speed_factor": 1.05},
            {"driver_id": "D04", "base_location": "Miami, FL", "capacity": 2, "speed_factor": 1.00},
            {"driver_id": "D05", "base_location": "Dallas, TX", "capacity": 2, "speed_factor": 0.98},
            {"driver_id": "D06", "base_location": "Boston, MA", "capacity": 2, "speed_factor": 1.02},
        ]
    )


def _expanded_driver_slots(drivers):
    rows = []
    for _, driver in drivers.iterrows():
        for slot in range(int(driver["capacity"])):
            item = driver.to_dict()
            item["slot"] = slot + 1
            item["driver_slot"] = f"{driver['driver_id']}-{slot + 1}"
            rows.append(item)
    return pd.DataFrame(rows)


def _assignment_cost(order, driver_slot):
    same_location_bonus = 8 if order["location"] == driver_slot["base_location"] else 0
    travel_minutes = float(order["distance_km"]) * 3.2 / float(driver_slot["speed_factor"])
    priority_penalty = 100 - float(order["delay_risk_score"])
    high_traffic_penalty = 6 if order["traffic_level"] == "High" else 0
    return max(travel_minutes + 0.35 * priority_penalty + high_traffic_penalty - same_location_bonus, 0)


def solve_transport_assignment(queue=None, drivers=None, top_n=12):
    """Assign top-risk orders to available driver slots by minimum cost."""
    queue = load_dashboard_data() if queue is None else queue.copy()
    drivers = build_driver_fleet() if drivers is None else drivers.copy()
    orders = queue.sort_values("delay_risk_score", ascending=False).head(top_n).reset_index(drop=True)
    slots = _expanded_driver_slots(drivers)
    if len(orders) > len(slots):
        orders = orders.head(len(slots)).copy()

    cost_matrix = np.zeros((len(orders), len(slots)))
    for i, (_, order) in enumerate(orders.iterrows()):
        for j, (_, slot) in enumerate(slots.iterrows()):
            cost_matrix[i, j] = _assignment_cost(order, slot)

    try:
        from scipy.optimize import linear_sum_assignment

        order_idx, slot_idx = linear_sum_assignment(cost_matrix)
    except Exception:
        remaining_orders = set(range(len(orders)))
        remaining_slots = set(range(len(slots)))
        pairs = []
        while remaining_orders and remaining_slots:
            i, j = min(
                ((i, j) for i in remaining_orders for j in remaining_slots),
                key=lambda pair: cost_matrix[pair[0], pair[1]],
            )
            pairs.append((i, j))
            remaining_orders.remove(i)
            remaining_slots.remove(j)
        order_idx, slot_idx = zip(*pairs)

    rows = []
    for i, j in zip(order_idx, slot_idx):
        order = orders.iloc[i]
        slot = slots.iloc[j]
        rows.append(
            {
                "order_id": order["order_id"],
                "restaurant_name": order["restaurant_name"],
                "location": order["location"],
                "traffic_level": order["traffic_level"],
                "distance_km": order["distance_km"],
                "delay_risk_score": order["delay_risk_score"],
                "priority": order["priority"],
                "driver_id": slot["driver_id"],
                "driver_slot": slot["driver_slot"],
                "driver_base_location": slot["base_location"],
                "estimated_assignment_cost": round(float(cost_matrix[i, j]), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("delay_risk_score", ascending=False).reset_index(drop=True)


def build_dss_figures(queue=None, assignments=None):
    """Save DSS-layer charts: priority mix, risk distribution and assignment cost."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    queue = load_dashboard_data() if queue is None else queue
    assignments = solve_transport_assignment() if assignments is None else assignments
    paths = []

    order = ["Low", "Medium", "High"]
    counts = queue["priority"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=["#27ae60", "#f39c12", "#c0392b"])
    ax.set_title("Priority distribution in the delay queue")
    ax.set_ylabel("Orders")
    for i, value in enumerate(counts.values):
        ax.text(i, value, int(value), ha="center", va="bottom")
    fig.tight_layout()
    path = FIGURES_DIR / "priority_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(queue["delay_risk_score"], bins=20, color="#2980b9", edgecolor="white")
    for boundary, label in [(35, "Low/Med"), (65, "Med/High")]:
        ax.axvline(boundary, color="red", linestyle="--", linewidth=1)
        ax.text(boundary, ax.get_ylim()[1] * 0.9, label, rotation=90, va="top", fontsize=8)
    ax.set_title("Delay risk score distribution")
    ax.set_xlabel("Delay risk score")
    ax.set_ylabel("Orders")
    fig.tight_layout()
    path = FIGURES_DIR / "risk_score_histogram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = assignments.sort_values("estimated_assignment_cost")
    ax.barh(plot_df["order_id"], plot_df["estimated_assignment_cost"], color="#16a085")
    ax.set_title("Estimated assignment cost per high-priority order")
    ax.set_xlabel("Assignment cost")
    fig.tight_layout()
    path = FIGURES_DIR / "transport_assignment_cost.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def build_transport_artifacts():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    assignments = solve_transport_assignment()
    drivers = build_driver_fleet()
    assignments.to_csv(METRICS_DIR / "transport_assignment.csv", index=False)
    drivers.to_csv(METRICS_DIR / "transport_driver_scenario.csv", index=False)
    pd.DataFrame(transport_cost_policy_spec()).to_csv(
        METRICS_DIR / "transport_cost_policy_spec.csv", index=False
    )
    summary = {
        "scenario_type": "simulated_driver_fleet_on_real_orders",
        "assigned_orders": int(len(assignments)),
        "drivers": int(len(drivers)),
        "total_capacity": int(drivers["capacity"].sum()),
        "mean_assignment_cost": round(float(assignments["estimated_assignment_cost"].mean()), 2),
        "high_priority_assigned": int((assignments["priority"] == "High").sum()),
        "algorithm": "Hungarian linear sum assignment; greedy fallback if scipy is unavailable",
        "cost_formula": "distance_km*3.2/speed_factor + 0.35*(100-risk_score) + high_traffic_penalty - same_location_bonus, floored at 0",
        "note": "Drivers/capacity are simulated because the Kaggle dataset has no driver table.",
    }
    with open(METRICS_DIR / "transport_assignment_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    build_dss_figures(assignments=assignments)
    return summary


if __name__ == "__main__":
    print(build_transport_artifacts())
