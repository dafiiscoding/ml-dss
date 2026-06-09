"""Create transparent validation artifacts for the rule-based DSS layer."""
import json
import os

import pandas as pd

from src.config import CATEGORY_WEIGHTS, REPORTS_DIR, ROUTING_RULES
from src.dashboard_data import load_dashboard_data
from src.decision_rules import get_dss_decision

METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")


def _scenario(name, probability, category, confidence, text, conflict=0.0):
    result = {
        "fused_informative_prob": probability,
        "fused_category": category,
        "fused_category_confidence": confidence,
        "conflict_score": conflict,
        "manual_review_threshold": 0.54,
    }
    decision = get_dss_decision(result, text)
    return {
        "scenario": name,
        "informative_probability": probability,
        "category": category,
        "category_confidence": confidence,
        "conflict_score": conflict,
        **decision,
    }


def build():
    os.makedirs(METRICS_DIR, exist_ok=True)
    if set(CATEGORY_WEIGHTS) != set(ROUTING_RULES):
        raise ValueError("Every risk category must have exactly one routing rule.")

    scenarios = pd.DataFrame(
        [
            _scenario(
                "critical_injury",
                0.99,
                "injured_or_dead_people",
                0.95,
                "urgent trapped injured rescue help",
            ),
            _scenario(
                "infrastructure_damage",
                0.82,
                "infrastructure_and_utility_damage",
                0.75,
                "bridge collapsed road blocked",
            ),
            _scenario(
                "ordinary_post",
                0.10,
                "not_humanitarian",
                0.99,
                "ordinary social update",
            ),
            _scenario(
                "critical_conflict",
                0.99,
                "injured_or_dead_people",
                0.95,
                "urgent trapped injured rescue help",
                conflict=0.90,
            ),
        ]
    )
    scenarios.to_csv(
        os.path.join(METRICS_DIR, "dss_scenarios.csv"), index=False
    )

    batch = load_dashboard_data()
    safe = batch[batch["evaluation_eligible"].astype(bool)].copy()
    sensitivity_rows = []
    for low_max, medium_max, policy in (
        (29, 59, "sensitive"),
        (39, 69, "current"),
        (49, 79, "strict"),
    ):
        priority = pd.cut(
            safe["risk_score"],
            bins=[-1, low_max, medium_max, 100],
            labels=["Low", "Medium", "High"],
        )
        counts = priority.value_counts()
        sensitivity_rows.append(
            {
                "policy": policy,
                "low_max": low_max,
                "medium_max": medium_max,
                "low_cases": int(counts.get("Low", 0)),
                "medium_cases": int(counts.get("Medium", 0)),
                "high_cases": int(counts.get("High", 0)),
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(
        os.path.join(METRICS_DIR, "dss_threshold_sensitivity.csv"),
        index=False,
    )

    summary = {
        "priority_ground_truth_available": False,
        "interpretation": (
            "Risk weights and priority thresholds are transparent policy "
            "assumptions, not learned or claimed as statistically optimal."
        ),
        "routing_categories_covered": len(ROUTING_RULES),
        "evaluation_rows": len(safe),
        "priority_counts": safe["priority"].value_counts().to_dict(),
        "manual_review_cases": int(safe["manual_review"].sum()),
        "manual_review_rate": round(float(safe["manual_review"].mean()), 4),
        "high_priority_review_cases": int(
            ((safe["priority"] == "High") & safe["manual_review"]).sum()
        ),
    }
    with open(
        os.path.join(METRICS_DIR, "dss_operational_summary.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return scenarios, summary


if __name__ == "__main__":
    build()
