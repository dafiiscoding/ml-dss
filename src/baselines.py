"""Deterministic dummy baselines for leakage-safe model evaluation."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def _informative_metrics(y_true, predictions, probabilities):
    y_true_binary = (np.asarray(y_true) == "informative").astype(int)
    predicted_binary = (np.asarray(predictions) == "informative").astype(int)
    return {
        "Accuracy": accuracy_score(y_true_binary, predicted_binary),
        "Balanced Accuracy": balanced_accuracy_score(
            y_true_binary, predicted_binary
        ),
        "Precision": precision_score(
            y_true_binary, predicted_binary, zero_division=0
        ),
        "Recall": recall_score(
            y_true_binary, predicted_binary, zero_division=0
        ),
        "F1": f1_score(y_true_binary, predicted_binary, zero_division=0),
        "F2": fbeta_score(
            y_true_binary, predicted_binary, beta=2, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true_binary, predicted_binary),
        "Average Precision": average_precision_score(
            y_true_binary, probabilities
        ),
    }


def _positive_probability(model, features):
    classes = list(model.classes_)
    probabilities = model.predict_proba(features)
    return probabilities[:, classes.index("informative")]


def evaluate_informative_baselines(train_labels, evaluation_labels):
    """Evaluate deterministic binary baselines fitted on train labels."""
    train_labels = np.asarray(train_labels)
    evaluation_labels = np.asarray(evaluation_labels)
    train_features = np.zeros((len(train_labels), 1))
    evaluation_features = np.zeros((len(evaluation_labels), 1))
    strategies = {
        "Prior / majority": DummyClassifier(strategy="prior"),
        "Always informative": DummyClassifier(
            strategy="constant", constant="informative"
        ),
        "Always not informative": DummyClassifier(
            strategy="constant", constant="not_informative"
        ),
    }
    rows = []
    for name, model in strategies.items():
        model.fit(train_features, train_labels)
        predictions = model.predict(evaluation_features)
        metrics = _informative_metrics(
            evaluation_labels,
            predictions,
            _positive_probability(model, evaluation_features),
        )
        rows.append({"Model": name, **metrics})
    return pd.DataFrame(rows).set_index("Model").round(4)


def evaluate_humanitarian_baseline(train_labels, evaluation_labels):
    """Evaluate the train-majority baseline for the eight-class task."""
    train_labels = np.asarray(train_labels)
    evaluation_labels = np.asarray(evaluation_labels)
    train_features = np.zeros((len(train_labels), 1))
    evaluation_features = np.zeros((len(evaluation_labels), 1))
    model = DummyClassifier(strategy="most_frequent")
    model.fit(train_features, train_labels)
    predictions = model.predict(evaluation_features)
    return pd.DataFrame(
        [
            {
                "Model": "Train majority",
                "Predicted Class": str(predictions[0]),
                "Accuracy": accuracy_score(evaluation_labels, predictions),
                "Balanced Accuracy": balanced_accuracy_score(
                    evaluation_labels, predictions
                ),
                "Macro F1": f1_score(
                    evaluation_labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "Weighted F1": f1_score(
                    evaluation_labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "MCC": matthews_corrcoef(evaluation_labels, predictions),
            }
        ]
    ).set_index("Model").round(4)
