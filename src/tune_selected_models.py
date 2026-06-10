"""Tune the four selected classifier families on the leakage-safe dev split.

The broad six-model comparison identifies one family per modality/task. This
module tunes only those selected families, persists the winning estimators, and
evaluates them once on the canonical test split. CLIP remains frozen.
"""

import json
import os
import pickle
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.data_loader import load_dataset
from src.split_integrity import evaluation_mask
from src.text_preprocessing import TextVectorizerWrapper

METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)


@dataclass(frozen=True)
class TuningTask:
    name: str
    model_path: str
    task_type: str
    target_column: str
    feature_source: str
    candidates: tuple


def _linear_svm(c):
    return CalibratedClassifierCV(
        LinearSVC(
            C=c,
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        cv=3,
        method="sigmoid",
        n_jobs=-1,
    )


def _logistic(c):
    return LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )


TASKS = (
    TuningTask(
        name="text_informative",
        model_path="text_inf_clf.pkl",
        task_type="binary",
        target_column="label",
        feature_source="text",
        candidates=tuple(
            ({"C": c}, _linear_svm(c)) for c in (0.05, 0.1, 0.3, 1.0, 3.0)
        ),
    ),
    TuningTask(
        name="text_humanitarian",
        model_path="text_cat_clf.pkl",
        task_type="multiclass",
        target_column="label_top",
        feature_source="text",
        candidates=tuple(
            ({"C": c}, _logistic(c)) for c in (0.05, 0.1, 0.3, 1.0, 3.0, 10.0)
        ),
    ),
    TuningTask(
        name="image_informative",
        model_path="image_inf_clf.pkl",
        task_type="binary",
        target_column="label",
        feature_source="image",
        candidates=tuple(
            (
                {"n_neighbors": k, "weights": weights},
                KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weights,
                    metric="euclidean",
                    n_jobs=-1,
                ),
            )
            for k in (5, 9, 15, 25, 41)
            for weights in ("uniform", "distance")
        ),
    ),
    TuningTask(
        name="image_humanitarian",
        model_path="image_cat_clf.pkl",
        task_type="multiclass",
        target_column="label_top",
        feature_source="image",
        candidates=tuple(
            ({"C": c}, _logistic(c)) for c in (0.05, 0.1, 0.3, 1.0, 3.0, 10.0)
        ),
    ),
)


def _metrics(y_true, y_pred, task_type):
    if task_type == "binary":
        kwargs = {"pos_label": "informative", "zero_division": 0}
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, **kwargs),
            "Recall": recall_score(y_true, y_pred, **kwargs),
            "F1": f1_score(y_true, y_pred, **kwargs),
            "F2": fbeta_score(y_true, y_pred, beta=2, **kwargs),
        }
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Weighted F1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def _rank(metrics, task_type):
    if task_type == "binary":
        return metrics["F2"], metrics["F1"], metrics["Accuracy"]
    return metrics["Macro F1"], metrics["Weighted F1"], metrics["Accuracy"]


def _is_better(candidate, incumbent, task_type, tolerance=0.001):
    """Compare dev results while avoiding immaterial multiclass wins.

    Macro-F1 on very rare classes can move from one or two predictions. When
    two multiclass configurations differ by at most ``tolerance``, prefer
    Weighted-F1 and Accuracy instead of treating the tiny Macro-F1 change as a
    decisive improvement.
    """
    if incumbent is None:
        return True
    if task_type == "binary":
        return _rank(candidate, task_type) > _rank(incumbent, task_type)
    primary_delta = candidate["Macro F1"] - incumbent["Macro F1"]
    if primary_delta > tolerance:
        return True
    if primary_delta < -tolerance:
        return False
    return (
        candidate["Weighted F1"],
        candidate["Accuracy"],
    ) > (
        incumbent["Weighted F1"],
        incumbent["Accuracy"],
    )


def _load_features():
    train_df, val_df, test_df, _ = load_dataset(use_sample_if_missing=False)
    vectorizer = TextVectorizerWrapper.load()
    text_features = {
        "train": vectorizer.transform(train_df["tweet_text"]),
        "val": vectorizer.transform(val_df["tweet_text"]),
        "test": vectorizer.transform(test_df["tweet_text"]),
    }
    image_features = {
        split: np.load(os.path.join(MODELS_DIR, f"X_{split}_img_emb.npy"))
        for split in ("train", "val", "test")
    }
    for split, frame in (("val", val_df), ("test", test_df)):
        mask = evaluation_mask(split, frame)
        text_features[split] = text_features[split][mask]
        image_features[split] = image_features[split][mask]
        if split == "val":
            val_df = frame.loc[mask].reset_index(drop=True)
        else:
            test_df = frame.loc[mask].reset_index(drop=True)
    return (
        {"train": train_df, "val": val_df, "test": test_df},
        {"text": text_features, "image": image_features},
    )


def tune_selected_models(task_names=None):
    frames, features = _load_features()
    output = os.path.join(METRICS_DIR, "tuned_model_summary.json")
    if task_names and os.path.exists(output):
        with open(output, encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        summary = {}
    summary.update({
        "protocol": {
            "family_selection": "six-model comparison on canonical dev",
            "hyperparameter_selection": "canonical dev only",
            "final_evaluation": "canonical test once after selection",
            "multiclass_tie_tolerance": 0.001,
            "clip": "frozen feature extractor; no CLIP fine-tuning",
        },
        "evaluation_rows": {
            split: len(frame) for split, frame in frames.items()
        },
    })
    summary.setdefault("tasks", {})
    tasks_to_run = [
        task for task in TASKS
        if task_names is None or task.name in set(task_names)
    ]
    if task_names and len(tasks_to_run) != len(set(task_names)):
        known = {task.name for task in TASKS}
        unknown = sorted(set(task_names) - known)
        raise ValueError(f"Unknown tuning task(s): {unknown}")

    for task in tasks_to_run:
        print(f"\n=== TUNING {task.name} ===", flush=True)
        x = features[task.feature_source]
        y_train = frames["train"][task.target_column]
        y_val = frames["val"][task.target_column]
        y_test = frames["test"][task.target_column]
        rows = []
        best = None

        for params, model in task.candidates:
            model.fit(x["train"], y_train)
            dev_metrics = _metrics(y_val, model.predict(x["val"]), task.task_type)
            row = {
                "Parameters": json.dumps(params, sort_keys=True),
                **dev_metrics,
            }
            rows.append(row)
            rank = _rank(dev_metrics, task.task_type)
            print(
                f"{task.name} {params}: "
                f"{'F2' if task.task_type == 'binary' else 'Macro F1'}="
                f"{rank[0]:.4f}",
                flush=True,
            )
            if best is None or _is_better(
                dev_metrics, best["dev_metrics"], task.task_type
            ):
                best = {
                    "rank": rank,
                    "params": params,
                    "model": model,
                    "dev_metrics": dev_metrics,
                }

        results = pd.DataFrame(rows)
        primary = "F2" if task.task_type == "binary" else "Macro F1"
        results["Selected"] = results["Parameters"].eq(
            json.dumps(best["params"], sort_keys=True)
        )
        results = results.sort_values(
            [primary, "F1" if task.task_type == "binary" else "Weighted F1", "Accuracy"],
            ascending=False,
        )
        results.round(4).to_csv(
            os.path.join(METRICS_DIR, f"{task.name}_tuning.csv"),
            index=False,
        )

        test_metrics = _metrics(
            y_test,
            best["model"].predict(x["test"]),
            task.task_type,
        )
        with open(os.path.join(MODELS_DIR, task.model_path), "wb") as handle:
            pickle.dump(best["model"], handle)
        summary["tasks"][task.name] = {
            "selected_parameters": best["params"],
            "selection_metric": primary,
            "dev_metrics": {
                key: round(float(value), 6)
                for key, value in best["dev_metrics"].items()
            },
            "test_metrics": {
                key: round(float(value), 6)
                for key, value in test_metrics.items()
            },
        }
        print(
            f"Selected {task.name}: {best['params']} "
            f"dev {primary}={best['dev_metrics'][primary]:.4f}",
            flush=True,
        )

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"\nSaved tuned models and summary to {output}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        action="append",
        choices=[task.name for task in TASKS],
        help="Tune only this task; may be provided multiple times.",
    )
    arguments = parser.parse_args()
    tune_selected_models(arguments.task)
