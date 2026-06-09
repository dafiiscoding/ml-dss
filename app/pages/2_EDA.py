import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import FIGURES_DIR, REPORTS_DIR
from src.data_loader import load_dataset
from src.eda_analysis import add_text_features, top_tokens

st.set_page_config(page_title="EDA - Disaster Response DSS", layout="wide")
st.title("Exploratory Data Analysis")
st.caption(
    "Exploratory evidence uses the 13,608-row train split only. "
    "Corpus-wide file checks are reported separately in Image quality."
)

METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")


@st.cache_data
def get_data():
    train, _, _, _ = load_dataset(use_sample_if_missing=False)
    return add_text_features(train)


df = get_data()
text_tab, image_tab, multimodal_tab = st.tabs(
    ["Text and labels", "Image quality", "Multimodal evidence"]
)

with text_tab:
    left, right = st.columns(2)
    with left:
        figure = px.histogram(
            df,
            x="word_count",
            color="label",
            nbins=35,
            barmode="overlay",
            title="Tweet length by informativeness",
        )
        st.plotly_chart(figure, width="stretch")
    with right:
        events = (
            df.groupby(["event_name", "label"])
            .size()
            .reset_index(name="Count")
        )
        figure = px.bar(
            events,
            x="event_name",
            y="Count",
            color="label",
            title="Samples by disaster event",
        )
        st.plotly_chart(figure, width="stretch")

    tokens = top_tokens(df[df["label"] == "informative"]["tweet_text"], top_n=15)
    figure = px.bar(
        tokens,
        x="frequency",
        y="token",
        orientation="h",
        title="Top tokens in informative posts",
    )
    st.plotly_chart(figure, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("K-Means topic discovery")
        with open(
            os.path.join(METRICS_DIR, "kmeans_summary.json"),
            encoding="utf-8",
        ) as stream:
            cluster_summary = json.load(stream)
        st.metric("Silhouette score", cluster_summary["silhouette"])
        sweep = pd.read_csv(
            os.path.join(METRICS_DIR, "kmeans_silhouette_by_k.csv")
        )
        st.line_chart(sweep.set_index("k"))
        st.caption(
            "No clear elbow appears; k=8 is taxonomy-aligned for diagnostic "
            "comparison, not claimed as the statistically optimal k."
        )
        cluster_rows = [
            {"Cluster": cluster, "Top terms": ", ".join(terms[:8])}
            for cluster, terms in cluster_summary["cluster_terms"].items()
        ]
        st.dataframe(pd.DataFrame(cluster_rows), hide_index=True, width="stretch")
    with right:
        st.subheader("Apriori association rules")
        rules = pd.read_csv(os.path.join(METRICS_DIR, "apriori_rules.csv"))
        st.dataframe(rules.head(10), hide_index=True, width="stretch")

with image_tab:
    with open(
        os.path.join(METRICS_DIR, "data_quality_summary.json"),
        encoding="utf-8",
    ) as stream:
        quality = json.load(stream)
    image_quality = quality["images"]
    columns = st.columns(4)
    columns[0].metric("Referenced images", "18,082")
    columns[1].metric("Valid images", f"{image_quality['valid_images']:,}")
    columns[2].metric("Extractor metadata files", image_quality["invalid_files"])
    columns[3].metric("Exact duplicate extras", image_quality["exact_duplicate_extra_files"])
    st.image(
        os.path.join(FIGURES_DIR, "image_gallery.png"),
        caption="One real image per official multimodal humanitarian category",
        width="stretch",
    )
    st.image(
        os.path.join(FIGURES_DIR, "tsne_clip.png"),
        caption="t-SNE projection of cached 512-dimensional CLIP embeddings",
        width="stretch",
    )
    st.info(
        "The 22 invalid files are macOS `._*` extractor metadata and are not "
        "referenced by any annotation row. Exact duplicate image hashes across "
        "splits are excluded from formal dev/test evaluation."
    )

with multimodal_tab:
    st.image(
        os.path.join(FIGURES_DIR, "multimodal_conflict.png"),
        caption="Ground-truth disagreement between text and image labels",
        width="stretch",
    )
    conflict = pd.read_csv(
        os.path.join(METRICS_DIR, "conflict_by_category.csv")
    )
    st.dataframe(conflict, width="stretch")
    st.warning(
        "Disagreement is evidence for a human-review mechanism, not proof that "
        "one modality is universally superior. Fusion is evaluated against the "
        "same official multimodal target in the Model Evaluation page."
    )
