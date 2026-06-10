import os

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.config import REPORTS_DIR
from src.dashboard_data import load_dashboard_data

st.set_page_config(page_title="Model Evaluation - Disaster Response DSS", layout="wide")
st.title("Model Evaluation")
st.caption(
    "Models are selected on 2,189 leakage-safe dev rows. Final reporting uses "
    "2,169 leakage-safe test rows against one common multimodal ground truth."
)

METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")

validation_tab, fusion_tab, robustness_tab, review_tab = st.tabs(
    [
        "Six-model validation",
        "Final multimodal test",
        "Robustness",
        "Manual-review rule",
    ]
)

with validation_tab:
    st.subheader("Text branch")
    left, right = st.columns(2)
    with left:
        st.markdown("**Informative classification (selection metric: F2)**")
        st.dataframe(
            pd.read_csv(
                os.path.join(METRICS_DIR, "text_informative_validation.csv")
            ),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.markdown("**Humanitarian classification (selection metric: Macro F1)**")
        st.dataframe(
            pd.read_csv(
                os.path.join(METRICS_DIR, "text_humanitarian_validation.csv")
            ),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Image branch using CLIP embeddings")
    left, right = st.columns(2)
    with left:
        st.dataframe(
            pd.read_csv(
                os.path.join(METRICS_DIR, "image_informative_validation.csv")
            ),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.dataframe(
            pd.read_csv(
                os.path.join(METRICS_DIR, "image_humanitarian_validation.csv")
            ),
            hide_index=True,
            width="stretch",
        )

with fusion_tab:
    binary = pd.read_csv(
        os.path.join(METRICS_DIR, "fusion_informative_test.csv")
    )
    category = pd.read_csv(
        os.path.join(METRICS_DIR, "fusion_humanitarian_test.csv")
    )
    dummy_binary = pd.read_csv(
        os.path.join(METRICS_DIR, "baseline_informative_test.csv")
    )
    dummy_category = pd.read_csv(
        os.path.join(METRICS_DIR, "baseline_humanitarian_test.csv")
    )
    st.subheader("Dummy baselines")
    left, right = st.columns(2)
    with left:
        st.dataframe(dummy_binary, hide_index=True, width="stretch")
    with right:
        st.dataframe(dummy_category, hide_index=True, width="stretch")
    st.warning(
        "Always predicting informative already reaches F2 = 0.8939 because "
        "62.75% of the safe test set is positive. Fusion reaches 0.9045, a "
        "gain of only 0.0106 F2. Use Accuracy, F1, Balanced Accuracy and MCC "
        "alongside F2."
    )

    st.subheader("Final systems")
    left, right = st.columns(2)
    with left:
        st.markdown("**Informative task**")
        st.dataframe(binary, hide_index=True, width="stretch")
        figure = px.bar(
            pd.concat(
                [
                    dummy_binary[
                        dummy_binary["Model"] == "Always informative"
                    ][["Model", "Accuracy", "F1", "F2"]],
                    binary[["Model", "Accuracy", "F1", "F2"]],
                ],
                ignore_index=True,
            ),
            x="Model",
            y=["Accuracy", "F1", "F2"],
            barmode="group",
            title="Held-out informative performance vs dummy",
        )
        st.plotly_chart(figure, width="stretch")
    with right:
        st.markdown("**Humanitarian task**")
        st.dataframe(category, hide_index=True, width="stretch")
        figure = px.bar(
            pd.concat(
                [
                    dummy_category[
                        ["Model", "Macro F1", "Weighted F1"]
                    ],
                    category[["Model", "Macro F1", "Weighted F1"]],
                ],
                ignore_index=True,
            ),
            x="Model",
            y=["Macro F1", "Weighted F1"],
            barmode="group",
            title="Held-out humanitarian performance vs majority",
        )
        st.plotly_chart(figure, width="stretch")

    predictions = load_dashboard_data()
    predictions = predictions[
        predictions["evaluation_eligible"].astype(bool)
    ]
    labels = sorted(predictions["true_category"].unique())
    matrix = confusion_matrix(
        predictions["true_category"],
        predictions["fused_category"],
        labels=labels,
    )
    short = [label.replace("_", " ")[:18] for label in labels]
    figure = ff.create_annotated_heatmap(
        z=matrix,
        x=short,
        y=short,
        colorscale="Blues",
        showscale=True,
    )
    figure.update_layout(
        title="Late-fusion humanitarian confusion matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=650,
    )
    st.plotly_chart(figure, width="stretch")

    st.markdown("**Per-class error analysis**")
    st.dataframe(
        pd.read_csv(
            os.path.join(
                METRICS_DIR,
                "fusion_humanitarian_classification_report.csv",
            )
        ).head(8),
        hide_index=True,
        width="stretch",
    )
    st.warning(
        "The rarest class, missing/found people, has only five leakage-safe "
        "test cases. Its F1 is not reliable enough for autonomous decisions."
    )

with robustness_tab:
    comparison = pd.read_csv(
        os.path.join(METRICS_DIR, "robustness_metric_comparison.csv")
    )
    intervals = pd.read_csv(
        os.path.join(METRICS_DIR, "robust_bootstrap_intervals.csv")
    )
    events = pd.read_csv(
        os.path.join(METRICS_DIR, "robust_event_stability.csv")
    )
    classes = pd.read_csv(
        os.path.join(METRICS_DIR, "robust_class_stability.csv")
    )

    st.caption(
        "Sensitivity analysis on 2,032 test rows after 137 additional "
        "pairwise-verified near-duplicate exclusions. Models, fusion weights "
        "and thresholds remain locked from the canonical protocol."
    )
    core = comparison[
        comparison["Task"].isin(
            ["Informative Fusion", "Humanitarian Fusion", "Manual Review"]
        )
    ]
    st.subheader("Canonical versus robust")
    st.dataframe(core, hide_index=True, width="stretch")

    st.subheader("Stratified bootstrap uncertainty")
    st.dataframe(intervals, hide_index=True, width="stretch")
    gains = intervals[
        intervals["Task"].str.contains("vs Dummy", regex=False)
    ].copy()
    figure = px.bar(
        gains,
        x="Task",
        y="Estimate",
        error_y=gains["CI High"] - gains["Estimate"],
        error_y_minus=gains["Estimate"] - gains["CI Low"],
        title="Paired gain over dummy baseline with 95% bootstrap CI",
    )
    figure.update_yaxes(title="Metric gain")
    st.plotly_chart(figure, width="stretch")
    st.info(
        "Informative F2 gain is positive under row-level bootstrap, but the "
        "absolute gain remains about 0.01. This does not establish "
        "generalization to unseen disaster events."
    )

    st.subheader("Event and class stability")
    left, right = st.columns(2)
    with left:
        event_figure = px.bar(
            events.sort_values(
                "Humanitarian Macro F1 (present classes)"
            ),
            x="Humanitarian Macro F1 (present classes)",
            y="Event",
            orientation="h",
            hover_data=["Rows", "Informative F2", "Review F1"],
            title="Humanitarian performance by event",
        )
        event_figure.update_xaxes(range=[0, 1])
        st.plotly_chart(event_figure, width="stretch")
    with right:
        class_figure = px.bar(
            classes,
            x="Class",
            y=["Canonical F1", "Robust F1"],
            barmode="group",
            title="Per-class F1 sensitivity",
        )
        class_figure.update_xaxes(tickangle=35)
        class_figure.update_yaxes(range=[0, 1])
        st.plotly_chart(class_figure, width="stretch")
    st.dataframe(
        classes[
            [
                "Class",
                "Canonical F1",
                "Robust F1",
                "F1 Delta",
                "Robust Support",
                "Stability Flag",
            ]
        ],
        hide_index=True,
        width="stretch",
    )

with review_tab:
    review = pd.read_csv(
        os.path.join(METRICS_DIR, "manual_review_test.csv")
    )
    st.dataframe(review, hide_index=True, width="stretch")
    st.info(
        "The conflict threshold is tuned on dev under a 25% human-review "
        "capacity constraint. This prevents a high-recall rule from sending "
        "nearly every post to a supervisor."
    )

    st.markdown("**DSS policy sensitivity**")
    st.dataframe(
        pd.read_csv(
            os.path.join(METRICS_DIR, "dss_threshold_sensitivity.csv")
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Priority has no ground-truth label in CrisisMMD. Risk weights and "
        "thresholds are transparent policy assumptions, not learned optima."
    )
