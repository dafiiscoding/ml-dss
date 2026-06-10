import json
import os
import pickle

import numpy as np

from src.config import (
    CATEGORY_WEIGHTS,
    FUSION_CONFIG_PATH,
    INFORMATIVE_THRESHOLD,
    MANUAL_REVIEW_CONFLICT_THRESHOLD,
    MODELS_DIR,
    WEIGHT_IMAGE_PROB,
    WEIGHT_TEXT_PROB,
)
from src.image_preprocessing import extract_image_embeddings
from src.text_preprocessing import TextVectorizerWrapper

CANONICAL_CATEGORIES = np.array(list(CATEGORY_WEIGHTS))
_MODELS = {}


def load_fusion_config():
    defaults = {
        "informative": {
            "text_weight": WEIGHT_TEXT_PROB,
            "image_weight": WEIGHT_IMAGE_PROB,
            "threshold": INFORMATIVE_THRESHOLD,
        },
        "category": {
            "text_weight": WEIGHT_TEXT_PROB,
            "image_weight": WEIGHT_IMAGE_PROB,
        },
        "manual_review": {
            "conflict_threshold": MANUAL_REVIEW_CONFLICT_THRESHOLD,
        },
    }
    if not os.path.exists(FUSION_CONFIG_PATH):
        return defaults
    with open(FUSION_CONFIG_PATH, encoding="utf-8") as f:
        saved = json.load(f)
    for section, values in defaults.items():
        saved.setdefault(section, {})
        for key, value in values.items():
            saved[section].setdefault(key, value)
    return saved


def align_probabilities(probabilities, model_classes):
    """Align one probability vector to the canonical eight-class order."""
    positions = {label: idx for idx, label in enumerate(model_classes)}
    return np.array([
        probabilities[positions[label]] if label in positions else 0.0
        for label in CANONICAL_CATEGORIES
    ])


def positive_probability(model, features):
    classes = list(model.classes_)
    return float(
        model.predict_proba(features)[0, classes.index("informative")]
    )


def load_all_models(include_image=True):
    global _MODELS
    if "vectorizer" not in _MODELS:
        _MODELS["vectorizer"] = TextVectorizerWrapper.load()
    for key, filename in (
            ("text_inf_clf", "text_inf_clf.pkl"),
            ("text_cat_clf", "text_cat_clf.pkl"),
        ):
        if key not in _MODELS:
            with open(os.path.join(MODELS_DIR, filename), "rb") as f:
                _MODELS[key] = pickle.load(f)
    if include_image:
        for key, filename in (
            ("image_inf_clf", "image_inf_clf.pkl"),
            ("image_cat_clf", "image_cat_clf.pkl"),
        ):
            if key not in _MODELS:
                with open(os.path.join(MODELS_DIR, filename), "rb") as f:
                    _MODELS[key] = pickle.load(f)
    return _MODELS


class MultimodalFusionPredictor:
    def __init__(self):
        self.models = load_all_models(include_image=False)
        self.config = load_fusion_config()
        self.categories_list = CANONICAL_CATEGORIES

    def predict(self, text, image_path=None, base_dir=None):
        text_features = self.models["vectorizer"].transform([text])
        text_inf_prob = positive_probability(
            self.models["text_inf_clf"], text_features
        )
        text_cat_probs = align_probabilities(
            self.models["text_cat_clf"].predict_proba(text_features)[0],
            self.models["text_cat_clf"].classes_,
        )

        image_present = bool(image_path)
        if image_present:
            self.models = load_all_models(include_image=True)
            image_features = extract_image_embeddings(
                [image_path], base_dir or "", batch_size=1
            )
            image_inf_prob = positive_probability(
                self.models["image_inf_clf"], image_features
            )
            image_cat_probs = align_probabilities(
                self.models["image_cat_clf"].predict_proba(image_features)[0],
                self.models["image_cat_clf"].classes_,
            )
            inf_cfg = self.config["informative"]
            cat_cfg = self.config["category"]
            fused_inf_prob = (
                inf_cfg["text_weight"] * text_inf_prob
                + inf_cfg["image_weight"] * image_inf_prob
            )
            fused_cat_probs = (
                cat_cfg["text_weight"] * text_cat_probs
                + cat_cfg["image_weight"] * image_cat_probs
            )
            binary_conflict = abs(text_inf_prob - image_inf_prob)
            category_conflict = 0.5 * np.abs(
                text_cat_probs - image_cat_probs
            ).sum()
            conflict_score = max(binary_conflict, category_conflict)
        else:
            image_inf_prob = None
            image_cat_probs = None
            fused_inf_prob = text_inf_prob
            fused_cat_probs = text_cat_probs
            binary_conflict = 0.0
            category_conflict = 0.0
            conflict_score = 0.0

        text_idx = int(np.argmax(text_cat_probs))
        fused_idx = int(np.argmax(fused_cat_probs))
        if image_present:
            image_idx = int(np.argmax(image_cat_probs))
        else:
            image_idx = None

        return {
            "text_informative_prob": text_inf_prob,
            "image_informative_prob": image_inf_prob,
            "fused_informative_prob": float(fused_inf_prob),
            "informative_threshold": float(
                self.config["informative"]["threshold"]
            ),
            "is_informative": bool(
                fused_inf_prob >= self.config["informative"]["threshold"]
            ),
            "text_category": str(self.categories_list[text_idx]),
            "text_category_confidence": float(text_cat_probs[text_idx]),
            "image_category": (
                str(self.categories_list[image_idx]) if image_present else None
            ),
            "image_category_confidence": (
                float(image_cat_probs[image_idx]) if image_present else None
            ),
            "fused_category": str(self.categories_list[fused_idx]),
            "fused_category_confidence": float(fused_cat_probs[fused_idx]),
            "binary_conflict_score": float(binary_conflict),
            "category_conflict_score": float(category_conflict),
            "conflict_score": float(conflict_score),
            "manual_review_threshold": float(
                self.config["manual_review"]["conflict_threshold"]
            ),
            "image_present": image_present,
        }
