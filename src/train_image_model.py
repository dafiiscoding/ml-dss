"""
Train & compare IMAGE classifiers on cached CLIP embeddings, persist best.

Runs only after scripts/cache_image_embeddings.py has produced
models/X_{train,val,test}_img_emb.npy. Until then it exits gracefully so the
rest of the pipeline (text-only) is unaffected.

Saves:
  - models/image_inf_clf.pkl, models/image_cat_clf.pkl  (best per task)
  - reports/metrics/image_*_validation.csv  (development-set comparison)
  - reports/metrics/image_*_test.csv  (selected-model final test result)
"""
import os
import json
import pickle
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR
from src.data_loader import load_dataset
from src.modeling import run_comparison, pick_best, evaluate_fitted_model
from src.split_integrity import filter_evaluation_split

METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)


def _load_cached(split):
    emb = os.path.join(MODELS_DIR, f"X_{split}_img_emb.npy")
    meta = os.path.join(MODELS_DIR, f"img_{split}_meta.csv")
    if not (os.path.exists(emb) and os.path.exists(meta)):
        return None, None
    return np.load(emb), pd.read_csv(meta)


def train_image_models():
    Xtr, mtr = _load_cached("train")
    Xva, mva = _load_cached("val")
    Xte, mte = _load_cached("test")
    if Xtr is None or Xva is None or Xte is None:
        print("[SKIP] Cached image embeddings not found. "
              "Run `python -m scripts.cache_image_embeddings` after images are ready.")
        return None

    train_df, val_df, test_df, _ = load_dataset(use_sample_if_missing=False)
    for name, X, metadata, df in (
        ("train", Xtr, mtr, train_df),
        ("val", Xva, mva, val_df),
        ("test", Xte, mte, test_df),
    ):
        if len(X) != len(df):
            raise ValueError(f"{name} embeddings/data length mismatch: {len(X)} != {len(df)}")
        for column in ("tweet_id", "image_id"):
            if not metadata[column].astype(str).reset_index(drop=True).equals(
                df[column].astype(str).reset_index(drop=True)
            ):
                raise ValueError(f"{name} embedding metadata order mismatch: {column}")

    val_df, Xva = filter_evaluation_split("val", val_df, Xva)
    test_df, Xte = filter_evaluation_split("test", test_df, Xte)

    print(f"Loaded cached embeddings: train={Xtr.shape}, val={Xva.shape}, test={Xte.shape}")
    summary = {
        "evaluation_rows": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        }
    }

    print("\n=== IMAGE TASK 1: Informativeness (binary) ===")
    inf_df, inf_models, _ = run_comparison(
        Xtr, train_df["label"], Xva, val_df["label"],
        task="binary", feature_type="dense", pos_label="informative")
    print(inf_df.to_string())
    best_inf, key = pick_best(inf_df, "binary")
    inf_df.to_csv(os.path.join(METRICS_DIR, "image_informative_validation.csv"))
    inf_test = evaluate_fitted_model(
        inf_models[best_inf], Xte, test_df["label"],
        task="binary", pos_label="informative")
    inf_test.to_frame("Test").to_csv(
        os.path.join(METRICS_DIR, "image_informative_test.csv"))
    with open(os.path.join(MODELS_DIR, "image_inf_clf.pkl"), "wb") as f:
        pickle.dump(inf_models[best_inf], f)
    summary["image_informative"] = {
        "best": best_inf,
        "selection_metric": key,
        "validation_score": float(inf_df.loc[best_inf, key]),
        "test_metrics": {k: float(v) for k, v in inf_test.items()},
    }

    print("\n=== IMAGE TASK 2: Humanitarian category (8-class) ===")
    cat_df, cat_models, _ = run_comparison(
        Xtr, train_df["label_top"], Xva, val_df["label_top"],
        task="multiclass", feature_type="dense")
    print(cat_df.to_string())
    best_cat, key2 = pick_best(cat_df, "multiclass")
    cat_df.to_csv(os.path.join(METRICS_DIR, "image_humanitarian_validation.csv"))
    cat_test = evaluate_fitted_model(
        cat_models[best_cat], Xte, test_df["label_top"],
        task="multiclass")
    cat_test.to_frame("Test").to_csv(
        os.path.join(METRICS_DIR, "image_humanitarian_test.csv"))
    with open(os.path.join(MODELS_DIR, "image_cat_clf.pkl"), "wb") as f:
        pickle.dump(cat_models[best_cat], f)
    summary["image_humanitarian"] = {
        "best": best_cat,
        "selection_metric": key2,
        "validation_score": float(cat_df.loc[best_cat, key2]),
        "test_metrics": {k: float(v) for k, v in cat_test.items()},
    }

    with open(os.path.join(METRICS_DIR, "image_best_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nImage models + comparison saved. Best: {summary}")
    return summary


if __name__ == "__main__":
    train_image_models()
