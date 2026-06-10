"""Runtime helpers shared by local and Streamlit Cloud execution."""
import os
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

REPO = Path(__file__).resolve().parents[1]
REAL_IMAGE_DIR = Path(RAW_DATA_DIR) / "CrisisMMD_v2.0" / "data_image"


def force_cloud_mode():
    return os.environ.get("ML_DSS_FORCE_CLOUD", "").lower() in {
        "1",
        "true",
        "yes",
    }


def real_image_corpus_available():
    """Return whether the local CrisisMMD image corpus can be browsed."""
    return not force_cloud_mode() and REAL_IMAGE_DIR.is_dir()


def load_processed_split(split):
    """Load a real exported split without falling back to synthetic data."""
    filename = {"dev": "val"}.get(split, split)
    path = Path(PROCESSED_DATA_DIR) / f"{filename}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed split is missing: {path}. Run `python -m scripts.run_all`."
        )
    return pd.read_csv(path)
