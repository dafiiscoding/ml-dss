"""
End-to-end integration test: model predictions -> Late Fusion -> DSS decision.

Uses a real CrisisMMD test-set example and fails if required artifacts are
missing. A skipped integration test must not be mistaken for a passing stage.
"""
import os
import sys

from src.data_loader import load_dataset
from src.config import MODELS_DIR
from src.decision_rules import get_dss_decision


def _models_ready():
    need = ["text_inf_clf.pkl", "text_cat_clf.pkl", "image_inf_clf.pkl", "image_cat_clf.pkl"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in need)


def run_integration_test():
    print("Starting End-to-End Integration Test...")
    if not _models_ready():
        raise FileNotFoundError(
            "Text and image models are required. Run `python -m scripts.run_all`."
        )

    from src.fusion import MultimodalFusionPredictor
    predictor = MultimodalFusionPredictor()
    print("Success: MultimodalFusionPredictor loaded.")

    # Pick a real informative test example.
    _, _, test_df, img_base = load_dataset(use_sample_if_missing=False)
    sample = test_df[test_df["label"] == "informative"].iloc[0]
    text, image = sample["tweet_text"], sample["image"]
    print(f"\nTest case:\n  Text:  {text[:120]}\n  Image: {image}")

    res = predictor.predict(text, image, img_base)
    decision = get_dss_decision(res, text)

    for k in ["risk_score", "priority", "assigned_team", "recommended_action"]:
        assert k in decision, f"Missing key: {k}"
    assert 0 <= decision["risk_score"] <= 100
    assert decision["priority"] in ["Low", "Medium", "High"]

    print(f"\n  Risk Score: {decision['risk_score']}")
    print(f"  Priority:   {decision['priority']}")
    print(f"  Team:       {decision['assigned_team']}")
    print(f"  Action:     {decision['recommended_action']}")
    print(f"  True label: {sample['label_top']} | Fused: {res['fused_category']}")
    print("\n=== INTEGRATION TEST PASSED ===")


if __name__ == "__main__":
    run_integration_test()
