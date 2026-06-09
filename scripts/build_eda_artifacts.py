"""Persist reusable EDA tables for the dashboard and report."""
import json
import os

import pandas as pd

from src.config import REPORTS_DIR
from src.data_loader import load_dataset
from src.eda_analysis import (
    association_rules_analysis,
    add_text_features,
    kmeans_silhouette_sweep,
    kmeans_topic_clustering,
    label_distribution,
    multimodal_conflict_analysis,
)

METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")


def build():
    train_df, _, _, _ = load_dataset(use_sample_if_missing=False)
    df = add_text_features(train_df)
    os.makedirs(METRICS_DIR, exist_ok=True)

    label_distribution(df, "label").to_csv(
        os.path.join(METRICS_DIR, "train_informative_distribution.csv")
    )
    label_distribution(df, "label_top").to_csv(
        os.path.join(METRICS_DIR, "train_humanitarian_distribution.csv")
    )
    pd.crosstab(
        df["event_name"], df["label_top"], normalize="index"
    ).round(4).to_csv(
        os.path.join(METRICS_DIR, "train_event_category_profile.csv")
    )
    with open(
        os.path.join(METRICS_DIR, "train_text_summary.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            {
                "rows": len(df),
                "mean_char_length": round(float(df["char_length"].mean()), 2),
                "mean_word_count": round(float(df["word_count"].mean()), 2),
                "informative_mean_word_count": round(
                    float(df.groupby("label")["word_count"].mean()["informative"]),
                    2,
                ),
                "not_informative_mean_word_count": round(
                    float(
                        df.groupby("label")["word_count"].mean()[
                            "not_informative"
                        ]
                    ),
                    2,
                ),
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )

    _, rules = association_rules_analysis(df)
    rules.to_csv(os.path.join(METRICS_DIR, "apriori_rules.csv"), index=False)

    sweep = kmeans_silhouette_sweep(df)
    sweep.to_csv(
        os.path.join(METRICS_DIR, "kmeans_silhouette_by_k.csv"), index=False
    )
    clustering = kmeans_topic_clustering(
        df, n_clusters=8, sample_size=8000
    )
    with open(
        os.path.join(METRICS_DIR, "kmeans_summary.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            {
                "silhouette": clustering["silhouette"],
                "selection_note": (
                    "k=8 is domain-aligned with the eight humanitarian "
                    "categories; the silhouette sweep remains uniformly low."
                ),
                "cluster_terms": clustering["cluster_terms"],
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )
    clustering["crosstab"].to_csv(
        os.path.join(METRICS_DIR, "kmeans_crosstab.csv")
    )

    conflict = multimodal_conflict_analysis(df)
    conflict["by_category"].to_csv(
        os.path.join(METRICS_DIR, "conflict_by_category.csv")
    )
    print(
        f"Saved EDA artifacts: {len(rules)} association rules, "
        f"silhouette={clustering['silhouette']}."
    )


if __name__ == "__main__":
    build()
