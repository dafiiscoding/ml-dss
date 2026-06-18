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

RISK_COMPONENT_POLICY = [
    {
        "component": "model_probability",
        "score_formula": "delayed_probability * 100",
        "normalization": "Xác suất trễ từ model được đổi từ 0-1 sang 0-100.",
        "weight": RISK_COMPONENT_WEIGHTS["model_probability"],
        "rationale": "Tín hiệu học máy là bằng chứng chính nên chiếm hơn một nửa Risk Score.",
    },
    {
        "component": "traffic_pressure",
        "score_formula": "Low=20, Medium=60, High=100",
        "normalization": "Traffic ordinal được ánh xạ thành áp lực vận hành trên thang 0-100.",
        "weight": RISK_COMPONENT_WEIGHTS["traffic_pressure"],
        "rationale": "Traffic là driver rủi ro rõ trong EDA và tác động trực tiếp tới giao hàng.",
    },
    {
        "component": "distance_pressure",
        "score_formula": "min(distance_km / 10 * 100, 100)",
        "normalization": "10 km là mức xa nhất trong dữ liệu, nên distance được scale tuyến tính 0-100.",
        "weight": RISK_COMPONENT_WEIGHTS["distance_pressure"],
        "rationale": "Đơn càng xa càng ít buffer, nhưng vẫn thấp hơn trọng số model/traffic.",
    },
    {
        "component": "peak_pressure",
        "score_formula": "100 nếu peak hour, ngược lại 20",
        "normalization": "Peak hour là cờ nhị phân, đổi thành mức áp lực cao/thấp.",
        "weight": RISK_COMPONENT_WEIGHTS["peak_pressure"],
        "rationale": "Giờ cao điểm ảnh hưởng năng lực vận hành nhưng không mạnh bằng traffic/distance.",
    },
    {
        "component": "complexity_pressure",
        "score_formula": "min(pizza_complexity / 20 * 100, 100)",
        "normalization": "Pizza complexity tối đa quan sát là 20, nên scale tuyến tính 0-100.",
        "weight": RISK_COMPONENT_WEIGHTS["complexity_pressure"],
        "rationale": "Đơn phức tạp có thể làm chuẩn bị lâu hơn, dùng như tín hiệu phụ.",
    },
    {
        "component": "weekend_pressure",
        "score_formula": "70 nếu weekend, ngược lại 40",
        "normalization": "Weekend là cờ nhị phân, đổi thành áp lực vừa/cao trên thang 0-100.",
        "weight": RISK_COMPONENT_WEIGHTS["weekend_pressure"],
        "rationale": "Weekend là yếu tố phụ, giữ trọng số nhỏ vì audit cho thấy tín hiệu yếu.",
    },
]


def risk_component_policy_spec():
    return [item.copy() for item in RISK_COMPONENT_POLICY]


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


def explain_delay_risk_score(order, delayed_probability):
    components = calculate_delay_risk_components(order, delayed_probability)
    rows = []
    for spec in risk_component_policy_spec():
        component = spec["component"]
        component_score = float(components[component])
        weight = float(spec["weight"])
        rows.append(
            {
                **spec,
                "component_score": round(component_score, 4),
                "weighted_contribution": round(component_score * weight, 4),
            }
        )
    return rows


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
