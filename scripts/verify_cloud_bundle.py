"""Validate the GitHub bundle used by Streamlit Community Cloud."""
import os
from pathlib import Path

os.environ["ML_DSS_FORCE_CLOUD"] = "1"

from src.dashboard_data import load_dashboard_data
from src.fusion import MultimodalFusionPredictor
from src.runtime import load_processed_split, real_image_corpus_available

REPO = Path(__file__).resolve().parents[1]
REQUIRED_MODELS = (
    "text_vectorizer.pkl",
    "text_inf_clf.pkl",
    "text_cat_clf.pkl",
    "image_inf_clf.pkl",
    "image_cat_clf.pkl",
)


def main():
    missing = [
        name for name in REQUIRED_MODELS
        if not (REPO / "models" / name).is_file()
    ]
    if missing:
        raise RuntimeError("Missing cloud inference models: " + ", ".join(missing))

    train = load_processed_split("train")
    dashboard = load_dashboard_data()
    if len(train) != 13608:
        raise RuntimeError(f"Unexpected processed train row count: {len(train)}")
    if len(dashboard) != 2237:
        raise RuntimeError(
            f"Unexpected dashboard prediction row count: {len(dashboard)}"
        )
    if real_image_corpus_available():
        raise RuntimeError("Forced cloud mode must hide the local image corpus.")

    predictor = MultimodalFusionPredictor()
    result = predictor.predict(
        "Urgent rescue needed after a bridge collapsed in the flood.",
        None,
    )
    required = {
        "fused_informative_prob",
        "fused_category",
        "fused_category_confidence",
    }
    if not required.issubset(result):
        raise RuntimeError("Text-only cloud inference returned an incomplete result.")
    if result["image_present"]:
        raise RuntimeError("Text-only cloud smoke test unexpectedly used an image.")

    total_mb = sum(
        (REPO / "models" / name).stat().st_size for name in REQUIRED_MODELS
    ) / 1024 / 1024
    print(f"Cloud model bundle: {total_mb:.2f} MiB")
    print(f"Processed train rows: {len(train):,}")
    print(f"Dashboard rows: {len(dashboard):,}")
    print(
        "Text-only inference: "
        f"{result['fused_category']} "
        f"(P informative={result['fused_informative_prob']:.3f})"
    )
    print("[OK] Streamlit Cloud bundle is ready.")


if __name__ == "__main__":
    main()
