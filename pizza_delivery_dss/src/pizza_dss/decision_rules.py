"""Tầng quyết định DSS: biến xác suất trễ thành Delay Risk Score (0-100),
Priority (Low/Medium/High) và Recommended Action.

Đây là **chính sách minh bạch** (trọng số do nhóm chọn), KHÔNG phải nhãn học từ
dữ liệu — bóc tách từng thành phần trong `calculate_delay_risk_components`. Thuật
ngữ xem `docs/GLOSSARY.md`.
"""
from pizza_dss.config import PRIORITY_LOW_MAX, PRIORITY_MEDIUM_MAX

TRAFFIC_SCORE = {"Low": 20, "Medium": 60, "High": 100}
RISK_COMPONENT_WEIGHTS = {
    "model_probability": 0.55,
    "traffic_pressure": 0.15,
    "distance_pressure": 0.12,
    "peak_pressure": 0.08,
    "complexity_pressure": 0.06,
    "weekend_pressure": 0.04,
}


def calculate_delay_risk_components(order, delayed_probability):
    distance = float(order["distance_km"])
    complexity = float(order["pizza_complexity"])
    traffic = TRAFFIC_SCORE.get(str(order["traffic_level"]), 50)
    peak = 100 if bool(order["is_peak_hour"]) else 20
    weekend = 70 if bool(order["is_weekend"]) else 40
    return {
        "model_probability": delayed_probability * 100,
        "traffic_pressure": traffic,
        "distance_pressure": min(distance / 10 * 100, 100),
        "peak_pressure": peak,
        "complexity_pressure": min(complexity / 20 * 100, 100),
        "weekend_pressure": weekend,
    }


def calculate_delay_risk_score(order, delayed_probability):
    components = calculate_delay_risk_components(order, delayed_probability)
    score = sum(RISK_COMPONENT_WEIGHTS[name] * components[name] for name in RISK_COMPONENT_WEIGHTS)
    return round(float(score), 1)


def get_priority_level(risk_score):
    if risk_score <= PRIORITY_LOW_MAX:
        return "Low"
    if risk_score <= PRIORITY_MEDIUM_MAX:
        return "Medium"
    return "High"


def recommend_action(order, priority):
    if priority == "High":
        return (
            "Escalate to dispatch lead, prepare backup driver, and notify "
            "customer support before the promised delivery window is missed."
        )
    if priority == "Medium":
        return (
            "Monitor driver assignment and traffic conditions; send proactive "
            "customer update if ETA worsens."
        )
    if str(order["traffic_level"]) == "High":
        return "Keep in normal queue but watch traffic because conditions are high."
    return "Keep in normal delivery queue."


def get_dss_decision(order, delayed_probability):
    risk_score = calculate_delay_risk_score(order, delayed_probability)
    priority = get_priority_level(risk_score)
    return {
        "delayed_probability": round(float(delayed_probability), 4),
        "delay_risk_score": risk_score,
        "priority": priority,
        "recommended_action": recommend_action(order, priority),
    }
