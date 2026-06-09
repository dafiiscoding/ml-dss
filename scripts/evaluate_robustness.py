"""Measure metric sensitivity after verified near-duplicate exclusions.

This script applies the already locked fusion weights and thresholds to both
the canonical and robust test masks. It never tunes or overwrites the canonical
fusion configuration.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.baselines import (
    evaluate_humanitarian_baseline,
    evaluate_informative_baselines,
)
from src.config import FUSION_CONFIG_PATH, MODELS_DIR, REPORTS_DIR
from src.data_loader import load_dataset
from src.evaluate_fusion import (
    CLASSES,
    _binary_metrics,
    _category_metrics,
    _conflict_scores,
    _load,
    _split_predictions,
)
from src.split_integrity import evaluation_mask, robust_evaluation_mask
from src.text_preprocessing import TextVectorizerWrapper

METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")


def _slice_predictions(predictions, mask):
    return {name: values[mask] for name, values in predictions.items()}


def _evaluate_subset(name, frame, predictions, mask, config, train_df):
    subset = frame.loc[mask].reset_index(drop=True)
    pred = _slice_predictions(predictions, mask)

    informative_config = config["informative"]
    fused_informative = (
        informative_config["text_weight"] * pred["text_inf"]
        + informative_config["image_weight"] * pred["image_inf"]
    )
    informative_metrics = _binary_metrics(
        (subset["label"] == "informative").astype(int).to_numpy(),
        fused_informative,
        informative_config["threshold"],
    )

    category_config = config["category"]
    fused_category = (
        category_config["text_weight"] * pred["text_cat"]
        + category_config["image_weight"] * pred["image_cat"]
    )
    category_metrics = _category_metrics(subset["label_top"], fused_category)
    category_predictions = CLASSES[np.argmax(fused_category, axis=1)]
    category_report = pd.DataFrame(
        classification_report(
            subset["label_top"],
            category_predictions,
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        )
    ).T
    category_confusion = pd.DataFrame(
        confusion_matrix(
            subset["label_top"],
            category_predictions,
            labels=CLASSES,
        ),
        index=CLASSES,
        columns=CLASSES,
    )

    conflict_scores = _conflict_scores(pred)
    conflict_threshold = config["manual_review"]["conflict_threshold"]
    manual_review_metrics = _binary_metrics(
        (subset["multimodal_agree"] == "Negative").astype(int).to_numpy(),
        conflict_scores,
        conflict_threshold,
    )
    manual_review_metrics["Review Rate"] = float(
        (conflict_scores >= conflict_threshold).mean()
    )

    informative_baselines = evaluate_informative_baselines(
        train_df["label"], subset["label"]
    )
    humanitarian_baseline = evaluate_humanitarian_baseline(
        train_df["label_top"], subset["label_top"]
    )
    return {
        "name": name,
        "rows": len(subset),
        "informative_positive_rate": float(
            (subset["label"] == "informative").mean()
        ),
        "informative": informative_metrics,
        "humanitarian": category_metrics,
        "manual_review": manual_review_metrics,
        "informative_baselines": informative_baselines,
        "humanitarian_baseline": humanitarian_baseline,
        "category_report": category_report,
        "category_confusion": category_confusion,
    }


def _comparison_rows(canonical, robust):
    rows = []
    groups = (
        ("Informative Fusion", "informative"),
        ("Humanitarian Fusion", "humanitarian"),
        ("Manual Review", "manual_review"),
    )
    for task, key in groups:
        for metric, canonical_value in canonical[key].items():
            robust_value = robust[key][metric]
            rows.append(
                {
                    "Task": task,
                    "Metric": metric,
                    "Canonical": canonical_value,
                    "Robust": robust_value,
                    "Delta": robust_value - canonical_value,
                }
            )

    for model in canonical["informative_baselines"].index:
        for metric in canonical["informative_baselines"].columns:
            canonical_value = float(
                canonical["informative_baselines"].loc[model, metric]
            )
            robust_value = float(
                robust["informative_baselines"].loc[model, metric]
            )
            rows.append(
                {
                    "Task": f"Informative Baseline - {model}",
                    "Metric": metric,
                    "Canonical": canonical_value,
                    "Robust": robust_value,
                    "Delta": robust_value - canonical_value,
                }
            )

    canonical_humanitarian = canonical["humanitarian_baseline"].iloc[0]
    robust_humanitarian = robust["humanitarian_baseline"].iloc[0]
    for metric in (
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "Weighted F1",
        "MCC",
    ):
        canonical_value = float(canonical_humanitarian[metric])
        robust_value = float(robust_humanitarian[metric])
        rows.append(
            {
                "Task": "Humanitarian Baseline - Train majority",
                "Metric": metric,
                "Canonical": canonical_value,
                "Robust": robust_value,
                "Delta": robust_value - canonical_value,
            }
        )
    return pd.DataFrame(rows)


def load_locked_test_context():
    """Load full test predictions without fitting or changing any artifact."""
    train_df, _, test_df, _ = load_dataset(use_sample_if_missing=False)
    embeddings = np.load(os.path.join(MODELS_DIR, "X_test_img_emb.npy"))
    with open(FUSION_CONFIG_PATH, encoding="utf-8") as stream:
        config = json.load(stream)

    models = {
        "text_inf": _load("text_inf_clf.pkl"),
        "text_cat": _load("text_cat_clf.pkl"),
        "image_inf": _load("image_inf_clf.pkl"),
        "image_cat": _load("image_cat_clf.pkl"),
    }
    predictions = _split_predictions(
        test_df,
        embeddings,
        models,
        TextVectorizerWrapper.load(),
    )
    return train_df, test_df, predictions, config


def evaluate():
    train_df, test_df, predictions, config = load_locked_test_context()
    canonical = _evaluate_subset(
        "canonical",
        test_df,
        predictions,
        evaluation_mask("test", test_df),
        config,
        train_df,
    )
    robust = _evaluate_subset(
        "robust",
        test_df,
        predictions,
        robust_evaluation_mask("test", test_df),
        config,
        train_df,
    )

    os.makedirs(METRICS_DIR, exist_ok=True)
    comparison = _comparison_rows(canonical, robust)
    comparison.round(6).to_csv(
        os.path.join(METRICS_DIR, "robustness_metric_comparison.csv"),
        index=False,
    )
    pd.DataFrame([robust["informative"]], index=["Late Fusion"]).round(6).to_csv(
        os.path.join(METRICS_DIR, "robust_informative_test.csv")
    )
    pd.DataFrame([robust["humanitarian"]], index=["Late Fusion"]).round(6).to_csv(
        os.path.join(METRICS_DIR, "robust_humanitarian_test.csv")
    )
    pd.DataFrame(
        [robust["manual_review"]], index=["Manual Review"]
    ).round(6).to_csv(
        os.path.join(METRICS_DIR, "robust_manual_review_test.csv")
    )
    robust["informative_baselines"].to_csv(
        os.path.join(METRICS_DIR, "robust_baseline_informative_test.csv")
    )
    robust["humanitarian_baseline"].to_csv(
        os.path.join(METRICS_DIR, "robust_baseline_humanitarian_test.csv")
    )
    robust["category_report"].round(6).to_csv(
        os.path.join(
            METRICS_DIR,
            "robust_humanitarian_classification_report.csv",
        )
    )
    robust["category_confusion"].to_csv(
        os.path.join(METRICS_DIR, "robust_humanitarian_confusion.csv")
    )

    always_canonical = float(
        canonical["informative_baselines"].loc["Always informative", "F2"]
    )
    always_robust = float(
        robust["informative_baselines"].loc["Always informative", "F2"]
    )
    summary = {
        "method": (
            "Sensitivity analysis with canonical fusion weights and thresholds "
            "locked; no model fitting, selection, or retuning on robust rows."
        ),
        "canonical_rows": canonical["rows"],
        "robust_rows": robust["rows"],
        "additional_rows_excluded": canonical["rows"] - robust["rows"],
        "canonical_positive_rate": canonical["informative_positive_rate"],
        "robust_positive_rate": robust["informative_positive_rate"],
        "canonical_metrics": {
            "informative": canonical["informative"],
            "humanitarian": canonical["humanitarian"],
            "manual_review": canonical["manual_review"],
        },
        "robust_metrics": {
            "informative": robust["informative"],
            "humanitarian": robust["humanitarian"],
            "manual_review": robust["manual_review"],
        },
        "informative_f2_gain_over_always_positive": {
            "canonical": canonical["informative"]["F2"] - always_canonical,
            "robust": robust["informative"]["F2"] - always_robust,
        },
    }
    with open(
        os.path.join(METRICS_DIR, "robustness_summary.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    print("=== Locked-config robustness sensitivity ===")
    print(
        comparison[
            comparison["Task"].isin(
                ("Informative Fusion", "Humanitarian Fusion", "Manual Review")
            )
        ].round(4).to_string(index=False)
    )
    print(
        "\nRows: "
        f"canonical={canonical['rows']}, robust={robust['rows']}; "
        "no retuning performed."
    )
    return summary, comparison


if __name__ == "__main__":
    evaluate()
