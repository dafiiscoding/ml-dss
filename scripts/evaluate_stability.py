"""Bootstrap, event, and class stability for the locked robust evaluation."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from scripts.evaluate_robustness import load_locked_test_context
from src.config import FIGURES_DIR, REPORTS_DIR
from src.evaluate_fusion import CLASSES, _conflict_scores
from src.split_integrity import evaluation_mask, robust_evaluation_mask
from src.stability_analysis import (
    percentile_interval,
    stratified_bootstrap_indices,
)

METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")
N_BOOTSTRAP = 2000
RANDOM_STATE = 42


def _locked_outputs(predictions, config):
    informative = (
        config["informative"]["text_weight"] * predictions["text_inf"]
        + config["informative"]["image_weight"] * predictions["image_inf"]
    )
    category = (
        config["category"]["text_weight"] * predictions["text_cat"]
        + config["category"]["image_weight"] * predictions["image_cat"]
    )
    return {
        "informative_probability": informative,
        "informative_prediction": (
            informative >= config["informative"]["threshold"]
        ).astype(int),
        "category_prediction": CLASSES[np.argmax(category, axis=1)],
        "conflict_score": _conflict_scores(predictions),
        "review_prediction": (
            _conflict_scores(predictions)
            >= config["manual_review"]["conflict_threshold"]
        ).astype(int),
    }


def _bootstrap_intervals(frame, outputs, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    y_inf = (frame["label"] == "informative").astype(int).to_numpy()
    inf_pred = outputs["informative_prediction"]
    inf_metrics = {
        "Accuracy": lambda idx: accuracy_score(y_inf[idx], inf_pred[idx]),
        "F1": lambda idx: f1_score(
            y_inf[idx], inf_pred[idx], zero_division=0
        ),
        "F2": lambda idx: fbeta_score(
            y_inf[idx], inf_pred[idx], beta=2, zero_division=0
        ),
        "MCC": lambda idx: matthews_corrcoef(y_inf[idx], inf_pred[idx]),
    }
    samples = {name: [] for name in inf_metrics}
    informative_gain = []
    for _ in range(n_bootstrap):
        idx = stratified_bootstrap_indices(y_inf, rng)
        for name, metric in inf_metrics.items():
            samples[name].append(metric(idx))
        informative_gain.append(
            inf_metrics["F2"](idx)
            - fbeta_score(
                y_inf[idx],
                np.ones(len(idx), dtype=int),
                beta=2,
                zero_division=0,
            )
        )
    for name, values in samples.items():
        estimate = inf_metrics[name](np.arange(len(frame)))
        rows.append(
            {
                "Task": "Informative Fusion",
                "Metric": name,
                "Estimate": estimate,
                **percentile_interval(values),
                "Resamples": n_bootstrap,
                "Method": "stratified by informative target",
            }
        )
    informative_baseline_f2 = fbeta_score(
        y_inf,
        np.ones(len(y_inf), dtype=int),
        beta=2,
        zero_division=0,
    )
    rows.append(
        {
            "Task": "Informative Fusion vs Dummy",
            "Metric": "F2 Gain",
            "Estimate": inf_metrics["F2"](np.arange(len(frame)))
            - informative_baseline_f2,
            **percentile_interval(informative_gain),
            "Resamples": n_bootstrap,
            "Method": "paired, stratified by informative target",
        }
    )

    y_category = frame["label_top"].to_numpy()
    category_pred = outputs["category_prediction"]
    category_metrics = {
        "Accuracy": lambda idx: accuracy_score(
            y_category[idx], category_pred[idx]
        ),
        "Macro F1": lambda idx: f1_score(
            y_category[idx],
            category_pred[idx],
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
        "Weighted F1": lambda idx: f1_score(
            y_category[idx],
            category_pred[idx],
            labels=CLASSES,
            average="weighted",
            zero_division=0,
        ),
    }
    samples = {name: [] for name in category_metrics}
    humanitarian_gain = []
    for _ in range(n_bootstrap):
        idx = stratified_bootstrap_indices(y_category, rng)
        for name, metric in category_metrics.items():
            samples[name].append(metric(idx))
        humanitarian_gain.append(
            category_metrics["Macro F1"](idx)
            - f1_score(
                y_category[idx],
                np.full(len(idx), "not_humanitarian", dtype=object),
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )
        )
    for name, values in samples.items():
        estimate = category_metrics[name](np.arange(len(frame)))
        rows.append(
            {
                "Task": "Humanitarian Fusion",
                "Metric": name,
                "Estimate": estimate,
                **percentile_interval(values),
                "Resamples": n_bootstrap,
                "Method": "stratified by humanitarian target",
            }
        )
    humanitarian_baseline_macro_f1 = f1_score(
        y_category,
        np.full(len(y_category), "not_humanitarian", dtype=object),
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )
    rows.append(
        {
            "Task": "Humanitarian Fusion vs Dummy",
            "Metric": "Macro F1 Gain",
            "Estimate": category_metrics["Macro F1"](
                np.arange(len(frame))
            )
            - humanitarian_baseline_macro_f1,
            **percentile_interval(humanitarian_gain),
            "Resamples": n_bootstrap,
            "Method": "paired, stratified by humanitarian target",
        }
    )

    y_review = (
        frame["multimodal_agree"] == "Negative"
    ).astype(int).to_numpy()
    review_pred = outputs["review_prediction"]
    review_metrics = {
        "Precision": lambda idx: precision_score(
            y_review[idx], review_pred[idx], zero_division=0
        ),
        "Recall": lambda idx: recall_score(
            y_review[idx], review_pred[idx], zero_division=0
        ),
        "F1": lambda idx: f1_score(
            y_review[idx], review_pred[idx], zero_division=0
        ),
        "Review Rate": lambda idx: float(review_pred[idx].mean()),
    }
    samples = {name: [] for name in review_metrics}
    for _ in range(n_bootstrap):
        idx = stratified_bootstrap_indices(y_review, rng)
        for name, metric in review_metrics.items():
            samples[name].append(metric(idx))
    for name, values in samples.items():
        estimate = review_metrics[name](np.arange(len(frame)))
        rows.append(
            {
                "Task": "Manual Review",
                "Metric": name,
                "Estimate": estimate,
                **percentile_interval(values),
                "Resamples": n_bootstrap,
                "Method": "stratified by disagreement target",
            }
        )
    return pd.DataFrame(rows)


def _event_stability(frame, outputs):
    rows = []
    for event, group in frame.groupby("event_name", sort=False):
        idx = group.index.to_numpy()
        y_inf = (group["label"] == "informative").astype(int).to_numpy()
        inf_pred = outputs["informative_prediction"][idx]
        y_category = group["label_top"].to_numpy()
        category_pred = outputs["category_prediction"][idx]
        present_classes = np.unique(y_category)
        y_review = (
            group["multimodal_agree"] == "Negative"
        ).astype(int).to_numpy()
        review_pred = outputs["review_prediction"][idx]
        rows.append(
            {
                "Event": event,
                "Rows": len(group),
                "Informative Rate": float(y_inf.mean()),
                "Informative Accuracy": accuracy_score(y_inf, inf_pred),
                "Informative F1": f1_score(
                    y_inf, inf_pred, zero_division=0
                ),
                "Informative F2": fbeta_score(
                    y_inf, inf_pred, beta=2, zero_division=0
                ),
                "Informative MCC": matthews_corrcoef(y_inf, inf_pred),
                "Supported Humanitarian Classes": len(present_classes),
                "Humanitarian Accuracy": accuracy_score(
                    y_category, category_pred
                ),
                "Humanitarian Macro F1 (present classes)": f1_score(
                    y_category,
                    category_pred,
                    labels=present_classes,
                    average="macro",
                    zero_division=0,
                ),
                "Humanitarian Weighted F1": f1_score(
                    y_category,
                    category_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "Disagreement Rate": float(y_review.mean()),
                "Review Precision": precision_score(
                    y_review, review_pred, zero_division=0
                ),
                "Review Recall": recall_score(
                    y_review, review_pred, zero_division=0
                ),
                "Review F1": f1_score(
                    y_review, review_pred, zero_division=0
                ),
                "Review Rate": float(review_pred.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("Rows", ascending=False)


def _class_stability(test_df, outputs, canonical_mask, robust_mask):
    rows = []
    for label in CLASSES:
        values = {"Class": label}
        for name, mask in (
            ("Canonical", canonical_mask),
            ("Robust", robust_mask),
        ):
            truth = test_df.loc[mask, "label_top"].to_numpy()
            predicted = outputs["category_prediction"][mask]
            values[f"{name} Precision"] = precision_score(
                truth,
                predicted,
                labels=[label],
                average="macro",
                zero_division=0,
            )
            values[f"{name} Recall"] = recall_score(
                truth,
                predicted,
                labels=[label],
                average="macro",
                zero_division=0,
            )
            values[f"{name} F1"] = f1_score(
                truth,
                predicted,
                labels=[label],
                average="macro",
                zero_division=0,
            )
            values[f"{name} Support"] = int((truth == label).sum())
        values["F1 Delta"] = values["Robust F1"] - values["Canonical F1"]
        values["Support Delta"] = (
            values["Robust Support"] - values["Canonical Support"]
        )
        values["Stability Flag"] = (
            "low support"
            if values["Robust Support"] < 30
            else (
                "material F1 shift"
                if abs(values["F1 Delta"]) >= 0.05
                else "stable"
            )
        )
        rows.append(values)
    return pd.DataFrame(rows)


def _build_figures(intervals, events):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    selected_keys = {
        ("Informative Fusion", "F2"),
        ("Humanitarian Fusion", "Macro F1"),
        ("Manual Review", "F1"),
    }
    selected = intervals[
        intervals.apply(
            lambda row: (row["Task"], row["Metric"]) in selected_keys,
            axis=1,
        )
    ].copy()
    selected["Label"] = selected["Task"] + " - " + selected["Metric"]
    xerr = np.vstack(
        [
            selected["Estimate"] - selected["CI Low"],
            selected["CI High"] - selected["Estimate"],
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(
        selected["Estimate"],
        selected["Label"],
        xerr=xerr,
        fmt="o",
        color="#22577a",
        capsize=4,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Metric value with 95% bootstrap CI")
    ax.set_title("Locked robust-test uncertainty")
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGURES_DIR, "robust_bootstrap_intervals.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    plot = events.sort_values("Humanitarian Macro F1 (present classes)")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        plot["Event"],
        plot["Humanitarian Macro F1 (present classes)"],
        color="#c0392b",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 over classes present in each event")
    ax.set_title("Robust-test performance varies by disaster event")
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGURES_DIR, "robust_event_stability.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def evaluate(n_bootstrap=N_BOOTSTRAP):
    _, test_df, predictions, config = load_locked_test_context()
    canonical_mask = evaluation_mask("test", test_df)
    robust_mask = robust_evaluation_mask("test", test_df)
    outputs = _locked_outputs(predictions, config)

    robust_frame = test_df.loc[robust_mask].copy()
    robust_frame.index = np.flatnonzero(robust_mask)
    intervals = _bootstrap_intervals(
        robust_frame.reset_index(drop=True),
        {key: values[robust_mask] for key, values in outputs.items()},
        n_bootstrap=n_bootstrap,
    )
    events = _event_stability(robust_frame, outputs)
    classes = _class_stability(
        test_df, outputs, canonical_mask, robust_mask
    )

    os.makedirs(METRICS_DIR, exist_ok=True)
    intervals.round(6).to_csv(
        os.path.join(METRICS_DIR, "robust_bootstrap_intervals.csv"),
        index=False,
    )
    events.round(6).to_csv(
        os.path.join(METRICS_DIR, "robust_event_stability.csv"),
        index=False,
    )
    classes.round(6).to_csv(
        os.path.join(METRICS_DIR, "robust_class_stability.csv"),
        index=False,
    )
    _build_figures(intervals, events)

    macro_row = intervals[
        (intervals["Task"] == "Humanitarian Fusion")
        & (intervals["Metric"] == "Macro F1")
    ].iloc[0]
    f2_row = intervals[
        (intervals["Task"] == "Informative Fusion")
        & (intervals["Metric"] == "F2")
    ].iloc[0]
    f2_gain_row = intervals[
        (intervals["Task"] == "Informative Fusion vs Dummy")
        & (intervals["Metric"] == "F2 Gain")
    ].iloc[0]
    macro_gain_row = intervals[
        (intervals["Task"] == "Humanitarian Fusion vs Dummy")
        & (intervals["Metric"] == "Macro F1 Gain")
    ].iloc[0]
    summary = {
        "method": (
            f"{n_bootstrap} deterministic stratified bootstrap resamples on "
            "the locked robust test set; no fitting or retuning."
        ),
        "informative_f2_95_ci": [
            float(f2_row["CI Low"]),
            float(f2_row["CI High"]),
        ],
        "humanitarian_macro_f1_95_ci": [
            float(macro_row["CI Low"]),
            float(macro_row["CI High"]),
        ],
        "informative_f2_gain_vs_dummy_95_ci": [
            float(f2_gain_row["CI Low"]),
            float(f2_gain_row["CI High"]),
        ],
        "humanitarian_macro_f1_gain_vs_dummy_95_ci": [
            float(macro_gain_row["CI Low"]),
            float(macro_gain_row["CI High"]),
        ],
        "event_rows": int(len(events)),
        "lowest_event_informative_f2": events.loc[
            events["Informative F2"].idxmin(), "Event"
        ],
        "lowest_event_humanitarian_macro_f1": events.loc[
            events["Humanitarian Macro F1 (present classes)"].idxmin(),
            "Event",
        ],
        "low_support_classes": classes.loc[
            classes["Stability Flag"] == "low support", "Class"
        ].tolist(),
        "material_f1_shift_classes": classes.loc[
            classes["Stability Flag"] == "material F1 shift", "Class"
        ].tolist(),
    }
    with open(
        os.path.join(METRICS_DIR, "robust_stability_summary.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    print("=== Robust bootstrap intervals ===")
    print(intervals.round(4).to_string(index=False))
    print("\n=== Event stability ===")
    print(
        events[
            [
                "Event",
                "Rows",
                "Informative F2",
                "Humanitarian Macro F1 (present classes)",
                "Review F1",
            ]
        ].round(4).to_string(index=False)
    )
    print("\n=== Class stability ===")
    print(
        classes[
            [
                "Class",
                "Canonical F1",
                "Robust F1",
                "F1 Delta",
                "Robust Support",
                "Stability Flag",
            ]
        ].round(4).to_string(index=False)
    )
    return summary, intervals, events, classes


if __name__ == "__main__":
    evaluate()
