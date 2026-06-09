"""Tune multimodal fusion on dev and report final results on test.

All three systems are evaluated against the same official multimodal targets
(`label` and `label_top`). Modality-specific labels remain diagnostic fields;
they are not mixed into the operational comparison.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.config import (
    CATEGORY_WEIGHTS,
    FUSION_CONFIG_PATH,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.data_loader import load_dataset
from src.split_integrity import filter_evaluation_split
from src.text_preprocessing import TextVectorizerWrapper

METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")
CLASSES = np.array(list(CATEGORY_WEIGHTS))


def _load(name):
    with open(os.path.join(MODELS_DIR, name), "rb") as f:
        return pickle.load(f)


def _positive_probability(model, X):
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index("informative")]


def _align_matrix(probabilities, model_classes):
    positions = {label: idx for idx, label in enumerate(model_classes)}
    aligned = np.zeros((len(probabilities), len(CLASSES)), dtype=float)
    for out_idx, label in enumerate(CLASSES):
        if label in positions:
            aligned[:, out_idx] = probabilities[:, positions[label]]
    return aligned


def _split_predictions(df, embeddings, models, vectorizer):
    text_features = vectorizer.transform(df["tweet_text"])
    return {
        "text_inf": _positive_probability(models["text_inf"], text_features),
        "image_inf": _positive_probability(models["image_inf"], embeddings),
        "text_cat": _align_matrix(
            models["text_cat"].predict_proba(text_features),
            models["text_cat"].classes_,
        ),
        "image_cat": _align_matrix(
            models["image_cat"].predict_proba(embeddings),
            models["image_cat"].classes_,
        ),
    }


def _binary_metrics(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, predicted),
        "Balanced Accuracy": balanced_accuracy_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "F2": fbeta_score(y_true, predicted, beta=2, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predicted),
        "Average Precision": average_precision_score(y_true, probabilities),
    }


def _category_metrics(y_true, probabilities):
    predicted = CLASSES[np.argmax(probabilities, axis=1)]
    return {
        "Accuracy": accuracy_score(y_true, predicted),
        "Macro F1": f1_score(y_true, predicted, average="macro", zero_division=0),
        "Weighted F1": f1_score(
            y_true, predicted, average="weighted", zero_division=0
        ),
    }


def _best_threshold(y_true, probabilities):
    best = None
    for threshold in np.linspace(0.20, 0.80, 31):
        metrics = _binary_metrics(y_true, probabilities, threshold)
        rank = (metrics["F2"], metrics["F1"], metrics["Accuracy"])
        if best is None or rank > best[0]:
            best = (rank, float(threshold), metrics)
    return best[1], best[2]


def _tune_binary_fusion(y_true, text_prob, image_prob):
    best = None
    rows = []
    for text_weight in np.linspace(0, 1, 21):
        fused = text_weight * text_prob + (1 - text_weight) * image_prob
        threshold, metrics = _best_threshold(y_true, fused)
        rows.append({
            "Text Weight": text_weight,
            "Image Weight": 1 - text_weight,
            "Threshold": threshold,
            **metrics,
        })
        rank = (metrics["F2"], metrics["F1"], metrics["Accuracy"])
        if best is None or rank > best[0]:
            best = (rank, float(text_weight), threshold)
    return best[1], best[2], pd.DataFrame(rows)


def _tune_category_fusion(y_true, text_prob, image_prob):
    best = None
    rows = []
    for text_weight in np.linspace(0, 1, 21):
        fused = text_weight * text_prob + (1 - text_weight) * image_prob
        metrics = _category_metrics(y_true, fused)
        rows.append({
            "Text Weight": text_weight,
            "Image Weight": 1 - text_weight,
            **metrics,
        })
        rank = (metrics["Macro F1"], metrics["Weighted F1"], metrics["Accuracy"])
        if best is None or rank > best[0]:
            best = (rank, float(text_weight))
    return best[1], pd.DataFrame(rows)


def _conflict_scores(predictions):
    binary_gap = np.abs(predictions["text_inf"] - predictions["image_inf"])
    category_tv = 0.5 * np.abs(
        predictions["text_cat"] - predictions["image_cat"]
    ).sum(axis=1)
    return np.maximum(binary_gap, category_tv)


def _tune_conflict_threshold(labels, scores, max_review_rate=0.25):
    """Select a high-conflict cutoff under a fixed human-review capacity."""
    target = (labels == "Negative").astype(int)
    best = None
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = scores >= threshold
        review_rate = float(predicted.mean())
        if review_rate > max_review_rate:
            continue
        metrics = _binary_metrics(target, scores, threshold)
        rank = (metrics["F1"], metrics["Precision"], metrics["Recall"])
        if best is None or rank > best[0]:
            best = (rank, float(threshold), review_rate)
    if best is None:
        return float(np.quantile(scores, 1 - max_review_rate))
    return best[1]


def evaluate():
    _, val_df, test_df, _ = load_dataset(use_sample_if_missing=False)
    val_embeddings = np.load(os.path.join(MODELS_DIR, "X_val_img_emb.npy"))
    test_embeddings = np.load(os.path.join(MODELS_DIR, "X_test_img_emb.npy"))
    val_df, val_embeddings = filter_evaluation_split(
        "val", val_df, val_embeddings
    )
    test_df, test_embeddings = filter_evaluation_split(
        "test", test_df, test_embeddings
    )
    vectorizer = TextVectorizerWrapper.load()
    models = {
        "text_inf": _load("text_inf_clf.pkl"),
        "text_cat": _load("text_cat_clf.pkl"),
        "image_inf": _load("image_inf_clf.pkl"),
        "image_cat": _load("image_cat_clf.pkl"),
    }
    val_pred = _split_predictions(
        val_df, val_embeddings, models, vectorizer,
    )
    test_pred = _split_predictions(
        test_df, test_embeddings, models, vectorizer,
    )

    y_val_inf = (val_df["label"] == "informative").astype(int).to_numpy()
    y_test_inf = (test_df["label"] == "informative").astype(int).to_numpy()

    text_threshold, _ = _best_threshold(y_val_inf, val_pred["text_inf"])
    image_threshold, _ = _best_threshold(y_val_inf, val_pred["image_inf"])
    inf_text_weight, inf_threshold, binary_tuning = _tune_binary_fusion(
        y_val_inf, val_pred["text_inf"], val_pred["image_inf"]
    )
    cat_text_weight, category_tuning = _tune_category_fusion(
        val_df["label_top"].to_numpy(),
        val_pred["text_cat"],
        val_pred["image_cat"],
    )

    conflict_threshold = _tune_conflict_threshold(
        val_df["multimodal_agree"].to_numpy(), _conflict_scores(val_pred)
    )

    fused_test_inf = (
        inf_text_weight * test_pred["text_inf"]
        + (1 - inf_text_weight) * test_pred["image_inf"]
    )
    binary_rows = []
    for name, probabilities, threshold in (
        ("Text-only", test_pred["text_inf"], text_threshold),
        ("Image-only", test_pred["image_inf"], image_threshold),
        ("Late Fusion", fused_test_inf, inf_threshold),
    ):
        binary_rows.append({
            "Model": name,
            "Threshold": threshold,
            **_binary_metrics(y_test_inf, probabilities, threshold),
        })
    binary_df = pd.DataFrame(binary_rows).set_index("Model").round(4)

    fused_test_cat = (
        cat_text_weight * test_pred["text_cat"]
        + (1 - cat_text_weight) * test_pred["image_cat"]
    )
    category_rows = []
    for name, probabilities in (
        ("Text-only", test_pred["text_cat"]),
        ("Image-only", test_pred["image_cat"]),
        ("Late Fusion", fused_test_cat),
    ):
        category_rows.append({
            "Model": name,
            **_category_metrics(test_df["label_top"], probabilities),
        })
    category_df = pd.DataFrame(category_rows).set_index("Model").round(4)
    fused_category_predictions = CLASSES[np.argmax(fused_test_cat, axis=1)]
    category_report = pd.DataFrame(
        classification_report(
            test_df["label_top"],
            fused_category_predictions,
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        )
    ).T.round(4)
    category_confusion = pd.DataFrame(
        confusion_matrix(
            test_df["label_top"],
            fused_category_predictions,
            labels=CLASSES,
        ),
        index=CLASSES,
        columns=CLASSES,
    )

    conflict_test = _conflict_scores(test_pred)
    conflict_target = (test_df["multimodal_agree"] == "Negative").astype(int)
    conflict_metrics = pd.Series(
        _binary_metrics(conflict_target, conflict_test, conflict_threshold),
        name="Manual Review",
    ).round(4)
    conflict_metrics["Review Rate"] = round(
        float((conflict_test >= conflict_threshold).mean()), 4
    )

    config = {
        "informative": {
            "text_weight": inf_text_weight,
            "image_weight": 1 - inf_text_weight,
            "threshold": inf_threshold,
        },
        "category": {
            "text_weight": cat_text_weight,
            "image_weight": 1 - cat_text_weight,
        },
        "manual_review": {
            "conflict_threshold": conflict_threshold,
            "max_review_rate_on_dev": 0.25,
        },
        "selection_split": "leakage-safe dev",
        "evaluation_split": "leakage-safe test",
        "evaluation_rows": {
            "validation": len(val_df),
            "test": len(test_df),
        },
        "target": "official multimodal label",
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    binary_df.to_csv(os.path.join(METRICS_DIR, "fusion_informative_test.csv"))
    category_df.to_csv(os.path.join(METRICS_DIR, "fusion_humanitarian_test.csv"))
    category_report.to_csv(
        os.path.join(
            METRICS_DIR, "fusion_humanitarian_classification_report.csv"
        )
    )
    category_confusion.to_csv(
        os.path.join(METRICS_DIR, "fusion_humanitarian_confusion.csv")
    )
    conflict_metrics.to_frame("Test").to_csv(
        os.path.join(METRICS_DIR, "manual_review_test.csv")
    )
    binary_tuning.round(4).to_csv(
        os.path.join(METRICS_DIR, "fusion_informative_dev_tuning.csv"),
        index=False,
    )
    category_tuning.round(4).to_csv(
        os.path.join(METRICS_DIR, "fusion_humanitarian_dev_tuning.csv"),
        index=False,
    )
    with open(FUSION_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("=== Informative test ===")
    print(binary_df.to_string())
    print("\n=== Humanitarian test ===")
    print(category_df.to_string())
    print("\n=== Manual-review test ===")
    print(conflict_metrics.to_string())
    print(f"\nFusion config saved to {FUSION_CONFIG_PATH}")
    return binary_df, category_df, conflict_metrics, config


if __name__ == "__main__":
    evaluate()
