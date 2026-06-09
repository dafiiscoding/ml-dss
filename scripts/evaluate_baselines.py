"""Generate leakage-safe dummy-baseline artifacts for dev and test."""
import json
import os

import pandas as pd

from src.baselines import (
    evaluate_humanitarian_baseline,
    evaluate_informative_baselines,
)
from src.config import REPORTS_DIR
from src.data_loader import load_dataset
from src.split_integrity import filter_evaluation_split

METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")


def evaluate():
    train_df, val_df, test_df, _ = load_dataset(use_sample_if_missing=False)
    val_df = filter_evaluation_split("val", val_df)
    test_df = filter_evaluation_split("test", test_df)
    os.makedirs(METRICS_DIR, exist_ok=True)

    results = {}
    for split, frame in (("validation", val_df), ("test", test_df)):
        informative = evaluate_informative_baselines(
            train_df["label"], frame["label"]
        )
        humanitarian = evaluate_humanitarian_baseline(
            train_df["label_top"], frame["label_top"]
        )
        informative.to_csv(
            os.path.join(
                METRICS_DIR, f"baseline_informative_{split}.csv"
            )
        )
        humanitarian.to_csv(
            os.path.join(
                METRICS_DIR, f"baseline_humanitarian_{split}.csv"
            )
        )
        results[split] = {
            "rows": len(frame),
            "informative_positive_rate": round(
                float((frame["label"] == "informative").mean()), 4
            ),
            "informative": informative.reset_index().to_dict("records"),
            "humanitarian": humanitarian.reset_index().to_dict("records"),
        }

    test_dummy = next(
        row
        for row in results["test"]["informative"]
        if row["Model"] == "Always informative"
    )
    test_dummy_f2 = float(test_dummy["F2"])
    fusion_path = os.path.join(
        METRICS_DIR, "fusion_informative_test.csv"
    )
    fusion_test_f2 = None
    if os.path.exists(fusion_path):
        fusion_metrics = pd.read_csv(fusion_path, index_col=0)
        fusion_test_f2 = float(fusion_metrics.loc["Late Fusion", "F2"])
    results["interpretation"] = {
        "warning": (
            "F2 strongly rewards recall. The always-informative baseline must "
            "be shown beside the final system."
        ),
        "always_informative_test_f2": test_dummy_f2,
        "fusion_test_f2": fusion_test_f2,
        "absolute_f2_gain_over_dummy": (
            round(fusion_test_f2 - test_dummy_f2, 4)
            if fusion_test_f2 is not None
            else None
        ),
    }
    with open(
        os.path.join(METRICS_DIR, "baseline_summary.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)

    print("=== Informative dummy baselines: leakage-safe test ===")
    print(
        evaluate_informative_baselines(
            train_df["label"], test_df["label"]
        ).to_string()
    )
    print("\n=== Humanitarian dummy baseline: leakage-safe test ===")
    print(
        evaluate_humanitarian_baseline(
            train_df["label_top"], test_df["label_top"]
        ).to_string()
    )
    gain = results["interpretation"]["absolute_f2_gain_over_dummy"]
    if gain is not None:
        print(f"\nF2 gain over always-informative dummy: {gain:.4f}")
    return results


if __name__ == "__main__":
    evaluate()
