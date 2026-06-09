"""Build a cached operational test-set view for the Streamlit dashboard."""
import os

import numpy as np
import pandas as pd

from src.config import CATEGORY_WEIGHTS, MODELS_DIR, PROCESSED_DATA_DIR
from src.data_loader import load_dataset
from src.decision_rules import get_dss_decision
from src.fusion import load_all_models, load_fusion_config
from src.split_integrity import evaluation_mask

CLASSES = np.array(list(CATEGORY_WEIGHTS))
DASHBOARD_DATA_PATH = os.path.join(
    PROCESSED_DATA_DIR, "dashboard_test_predictions.csv"
)


def _align_matrix(probabilities, model_classes):
    positions = {label: idx for idx, label in enumerate(model_classes)}
    aligned = np.zeros((len(probabilities), len(CLASSES)))
    for idx, label in enumerate(CLASSES):
        if label in positions:
            aligned[:, idx] = probabilities[:, positions[label]]
    return aligned


def _positive_probability(model, features):
    classes = list(model.classes_)
    return model.predict_proba(features)[:, classes.index("informative")]


def build_dashboard_data():
    _, _, test_df, _ = load_dataset(use_sample_if_missing=False)
    eligible = evaluation_mask("test", test_df)
    models = load_all_models()
    config = load_fusion_config()

    text_features = models["vectorizer"].transform(test_df["tweet_text"])
    image_features = np.load(os.path.join(MODELS_DIR, "X_test_img_emb.npy"))

    text_inf = _positive_probability(models["text_inf_clf"], text_features)
    image_inf = _positive_probability(models["image_inf_clf"], image_features)
    text_cat = _align_matrix(
        models["text_cat_clf"].predict_proba(text_features),
        models["text_cat_clf"].classes_,
    )
    image_cat = _align_matrix(
        models["image_cat_clf"].predict_proba(image_features),
        models["image_cat_clf"].classes_,
    )

    inf_cfg = config["informative"]
    cat_cfg = config["category"]
    fused_inf = (
        inf_cfg["text_weight"] * text_inf
        + inf_cfg["image_weight"] * image_inf
    )
    fused_cat = (
        cat_cfg["text_weight"] * text_cat
        + cat_cfg["image_weight"] * image_cat
    )
    text_idx = np.argmax(text_cat, axis=1)
    image_idx = np.argmax(image_cat, axis=1)
    fused_idx = np.argmax(fused_cat, axis=1)
    binary_conflict = np.abs(text_inf - image_inf)
    category_conflict = 0.5 * np.abs(text_cat - image_cat).sum(axis=1)
    conflict = np.maximum(binary_conflict, category_conflict)

    rows = []
    for i, (_, source) in enumerate(test_df.iterrows()):
        result = {
            "text_informative_prob": float(text_inf[i]),
            "image_informative_prob": float(image_inf[i]),
            "fused_informative_prob": float(fused_inf[i]),
            "informative_threshold": float(inf_cfg["threshold"]),
            "is_informative": bool(fused_inf[i] >= inf_cfg["threshold"]),
            "text_category": str(CLASSES[text_idx[i]]),
            "text_category_confidence": float(text_cat[i, text_idx[i]]),
            "image_category": str(CLASSES[image_idx[i]]),
            "image_category_confidence": float(image_cat[i, image_idx[i]]),
            "fused_category": str(CLASSES[fused_idx[i]]),
            "fused_category_confidence": float(fused_cat[i, fused_idx[i]]),
            "binary_conflict_score": float(binary_conflict[i]),
            "category_conflict_score": float(category_conflict[i]),
            "conflict_score": float(conflict[i]),
            "manual_review_threshold": float(
                config["manual_review"]["conflict_threshold"]
            ),
            "image_present": True,
        }
        decision = get_dss_decision(result, source["tweet_text"])
        rows.append({
            "tweet_id": source["tweet_id"],
            "image_id": source["image_id"],
            "event_name": source["event_name"],
            "tweet_text": source["tweet_text"],
            "image": source["image"],
            "true_label": source["label"],
            "true_category": source["label_top"],
            "multimodal_agree": source["multimodal_agree"],
            "evaluation_eligible": bool(eligible[i]),
            "fused_informative_prob": result["fused_informative_prob"],
            "fused_category": result["fused_category"],
            "conflict_score": result["conflict_score"],
            **decision,
        })

    output = pd.DataFrame(rows).sort_values(
        "risk_score", ascending=False
    ).reset_index(drop=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output.to_csv(DASHBOARD_DATA_PATH, index=False)
    print(f"Saved {len(output)} dashboard rows -> {DASHBOARD_DATA_PATH}")
    return output


def load_dashboard_data():
    if not os.path.exists(DASHBOARD_DATA_PATH):
        return build_dashboard_data()
    return pd.read_csv(DASHBOARD_DATA_PATH)


if __name__ == "__main__":
    build_dashboard_data()
