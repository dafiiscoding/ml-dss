from src.config import PRIORITY_LOW_MAX, PRIORITY_MEDIUM_MAX, ROUTING_RULES
from src.risk_scoring import calculate_risk_score

def get_priority_level(risk_score):
    """
    Maps 0-100 risk score to Low, Medium, High priority.
    """
    if risk_score <= PRIORITY_LOW_MAX:
        return "Low"
    elif risk_score <= PRIORITY_MEDIUM_MAX:
        return "Medium"
    else:
        return "High"

def get_dss_decision(fusion_results, raw_text):
    """
    Takes prediction results and text, applies decision layer rules,
    and determines Priority, Response Team, Recommended Action, and Human Review necessity.
    """
    fusion_prob = fusion_results["fused_informative_prob"]
    category = fusion_results["fused_category"]
    cat_conf = fusion_results["fused_category_confidence"]
    conflict_score = fusion_results["conflict_score"]

    # 1. Compute Risk Score
    risk_score = calculate_risk_score(fusion_prob, category, raw_text, cat_conf)

    # 2. Determine base priority
    priority = get_priority_level(risk_score)

    # 3. Determine Routing and Action based on Category
    routing = ROUTING_RULES.get(category, {
        "team": "Coordination Team",
        "action": "Route to general logistics for evaluation."
    })

    base_team = routing["team"]
    base_action = routing["action"]
    assigned_team = base_team
    recommended_action = base_action
    manual_review = False
    review_reason = ""

    # 4. Decision Rule Overrides

    # Rule 4.1: High Conflict between Text and Image
    conflict_threshold = fusion_results.get("manual_review_threshold", 0.5)
    if conflict_score >= conflict_threshold:
        assigned_team = (
            "Supervisor"
            if base_team == "No Action"
            else f"Supervisor + {base_team}"
        )
        recommended_action = (
            "Verify the text-image conflict immediately. In parallel: "
            f"{base_action}"
        )
        # Human review can escalate Low to Medium, but must never downgrade an
        # already High-priority operational case.
        if priority == "Low":
            priority = "Medium"
        manual_review = True
        review_reason = (
            f"High multimodal conflict ({conflict_score:.2f} >= "
            f"{conflict_threshold:.2f}). Text and image require human review."
        )

    # Rule 4.2: Low Confidence but high risk keyword matches
    elif category == "not_humanitarian" and risk_score > 40:
        # Classifier thinks it's not humanitarian, but keywords are very active
        assigned_team = "Supervisor"
        recommended_action = "Review content for potential sarcasm, misclassifications, or emerging crisis indicators."
        priority = "Medium"
        manual_review = True
        review_reason = f"Low model confidence with high-risk keyword count (Risk Score: {risk_score})."

    # Rule 4.3: High priority cases mapped to No Action
    elif priority == "High" and assigned_team == "No Action":
        assigned_team = "Coordination Team"
        recommended_action = "Review for potential threat. Escalate low priority category marked as high risk."

    return {
        "risk_score": risk_score,
        "priority": priority,
        "base_priority": get_priority_level(risk_score),
        "assigned_team": assigned_team,
        "base_assigned_team": base_team,
        "recommended_action": recommended_action,
        "manual_review": manual_review,
        "review_reason": review_reason
    }

if __name__ == "__main__":
    # Test decision logic
    mock_fusion = {
        "fused_informative_prob": 0.85,
        "fused_category": "injured_or_dead_people",
        "fused_category_confidence": 0.90,
        "conflict_score": 0.10,
        "text_informative_prob": 0.80,
        "image_informative_prob": 0.90
    }
    tweet = "3 people injured in natural disaster!"

    decision = get_dss_decision(mock_fusion, tweet)
    print("Normal High Priority Case:")
    for k, v in decision.items():
        print(f"  {k}: {v}")

    # Test conflict override
    mock_conflict = {
        "fused_informative_prob": 0.50,
        "fused_category": "injured_or_dead_people",
        "fused_category_confidence": 0.60,
        "conflict_score": 0.80, # high conflict
        "text_informative_prob": 0.90,
        "image_informative_prob": 0.10
    }

    conflict_decision = get_dss_decision(mock_conflict, tweet)
    print("\nConflict Override Case:")
    for k, v in conflict_decision.items():
        print(f"  {k}: {v}")
