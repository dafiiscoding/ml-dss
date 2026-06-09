"""
Model training & comparison for the Disaster Response DSS.

Implements the course classification algorithms (Decision Tree, Naive Bayes,
k-NN, SVM, Ensemble) alongside Logistic Regression and compares them on the
two CrisisMMD tasks (informativeness, humanitarian category). Reusable for both
the text branch (TF-IDF features) and the image branch (CLIP embeddings).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, classification_report,
)
from src.config import RANDOM_STATE


def build_models(feature_type="text"):
    """
    Return the dict of named classifiers to compare.

    feature_type:
      - "text"  : features are non-negative TF-IDF -> MultinomialNB is valid.
      - "dense" : features are dense embeddings (CLIP) -> GaussianNB instead.

    LinearSVC is wrapped in CalibratedClassifierCV so the SVM also exposes
    predict_proba (needed by the late-fusion layer).
    """
    nb = MultinomialNB() if feature_type == "text" else GaussianNB()
    svm = CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000), cv=3)
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=20, random_state=RANDOM_STATE),
        "Naive Bayes": nb,
        "k-NN": KNeighborsClassifier(n_neighbors=15),
        "SVM (Linear)": svm,
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE),
    }


def _metrics(y_true, y_pred, task, pos_label=None):
    if task == "binary":
        kw = {"pos_label": pos_label, "zero_division": 0}
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, **kw),
            "Recall": recall_score(y_true, y_pred, **kw),
            "F1": f1_score(y_true, y_pred, **kw),
            "F2": fbeta_score(y_true, y_pred, beta=2, **kw),
        }
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def run_comparison(X_train, y_train, X_test, y_test, task="binary",
                   feature_type="text", pos_label=None):
    """
    Fit every model and return (metrics_df, fitted_models, predictions).

    task: "binary" (informativeness) or "multiclass" (humanitarian).
    pos_label: positive class for binary metrics (e.g. "informative").
    """
    models = build_models(feature_type)
    rows, fitted, preds = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        m = {"Model": name}
        m.update(_metrics(y_test, y_pred, task, pos_label))
        rows.append(m)
        fitted[name] = model
        preds[name] = y_pred
    metrics_df = pd.DataFrame(rows).set_index("Model").round(4)
    return metrics_df, fitted, preds


def pick_best(metrics_df, task="binary"):
    """Choose the best model. For this DSS we prioritise Recall-oriented F2
    (binary) / Macro F1 (multiclass) because missing a true emergency is costly."""
    key = "F2" if task == "binary" else "Macro F1"
    best = metrics_df[key].idxmax()
    return best, key


def evaluate_fitted_model(model, X, y, task="binary", pos_label=None):
    """Evaluate one already-fitted model with the project's standard metrics."""
    return pd.Series(_metrics(y, model.predict(X), task, pos_label)).round(4)


if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.split_integrity import evaluation_mask
    from src.text_preprocessing import TextVectorizerWrapper

    train_df, val_df, _, _ = load_dataset(use_sample_if_missing=False)
    val_mask = evaluation_mask("val", val_df)
    safe_val = val_df.loc[val_mask].reset_index(drop=True)
    vec = TextVectorizerWrapper(max_features=2000)
    Xtr = vec.fit_transform(train_df["tweet_text"])
    Xdev = vec.transform(safe_val["tweet_text"])

    print("=== DEV MODEL SELECTION: Informativeness (binary) ===")
    mdf, _, _ = run_comparison(
        Xtr, train_df["label"], Xdev, safe_val["label"],
        task="binary", feature_type="text", pos_label="informative")
    print(mdf.to_string())
    b, k = pick_best(mdf, "binary"); print(f"-> Best by {k}: {b}\n")

    print("=== DEV MODEL SELECTION: Humanitarian category (8-class) ===")
    mdf2, _, _ = run_comparison(
        Xtr, train_df["label_top"], Xdev, safe_val["label_top"],
        task="multiclass", feature_type="text")
    print(mdf2.to_string())
    b2, k2 = pick_best(mdf2, "multiclass"); print(f"-> Best by {k2}: {b2}")
