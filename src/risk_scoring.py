import re
from src.config import (
    CATEGORY_WEIGHTS, DEFAULT_CATEGORY_WEIGHT, CRISIS_KEYWORDS,
    WEIGHT_RISK_FUSION, WEIGHT_RISK_CATEGORY, WEIGHT_RISK_KEYWORD, WEIGHT_RISK_CONFIDENCE
)
from src.text_preprocessing import clean_tweet_text

def calculate_keyword_score(text):
    """
    Calculates a keyword score between 0 and 100 based on the presence of crisis-related keywords.
    Each unique keyword matches adds 20 points, capped at 100.
    """
    cleaned = clean_tweet_text(text)
    words = set(cleaned.split())

    matches = 0
    for keyword in CRISIS_KEYWORDS:
        if keyword in words:
            matches += 1

    # Scale: 1 match = 20 pts, 2 = 40, etc. Capped at 100.
    score = min(matches * 20, 100)
    return score, matches

def calculate_risk_components(fusion_prob, category, text, category_confidence):
    """
    Return transparent 0-100 risk components.

    Category confidence is multiplied by category severity. A confident
    ``not_humanitarian`` prediction must not increase operational risk.
    """
    s_fusion = fusion_prob * 100
    cat_weight = CATEGORY_WEIGHTS.get(category, DEFAULT_CATEGORY_WEIGHT)
    s_category = cat_weight * 100
    s_keyword, matches = calculate_keyword_score(text)
    s_confidence = category_confidence * s_category
    return {
        "informative": s_fusion,
        "category_severity": s_category,
        "keyword_urgency": s_keyword,
        "severity_confidence": s_confidence,
        "keyword_matches": matches,
    }


def calculate_risk_score(fusion_prob, category, text, category_confidence):
    """Calculate the policy-based 0-100 risk score."""
    components = calculate_risk_components(
        fusion_prob, category, text, category_confidence
    )
    total_weight = (WEIGHT_RISK_FUSION + WEIGHT_RISK_CATEGORY +
                    WEIGHT_RISK_KEYWORD + WEIGHT_RISK_CONFIDENCE)
    risk_score = (
        WEIGHT_RISK_FUSION * components["informative"] +
        WEIGHT_RISK_CATEGORY * components["category_severity"] +
        WEIGHT_RISK_KEYWORD * components["keyword_urgency"] +
        WEIGHT_RISK_CONFIDENCE * components["severity_confidence"]
    ) / total_weight
    return round(risk_score, 1)

if __name__ == "__main__":
    # Test risk score
    tweet = "URGENT: Flood on Main Street! 3 people are injured under the collapsed bridge!"
    keyword_score, matches = calculate_keyword_score(tweet)
    print(f"Text: '{tweet}'")
    print(f"Keyword score: {keyword_score} (Matches: {matches})")

    score = calculate_risk_score(
        fusion_prob=0.95,
        category="injured_or_dead_people",
        text=tweet,
        category_confidence=0.88
    )
    print(f"Calculated Risk Score: {score}")
