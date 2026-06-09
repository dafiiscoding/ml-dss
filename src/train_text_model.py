"""
Train & compare TEXT classifiers, then persist the best model per task.

Delegates the 6-model comparison to src.modeling, saves:
  - models/text_vectorizer.pkl
  - models/text_inf_clf.pkl  (best informativeness model, used by fusion/DSS)
  - models/text_cat_clf.pkl  (best humanitarian model)
  - reports/metrics/text_*_validation.csv  (development-set comparison)
  - reports/metrics/text_*_test.csv  (selected-model final test result)
"""
import os
import json
import pickle

from src.config import MODELS_DIR, REPORTS_DIR
from src.data_loader import load_dataset
from src.text_preprocessing import TextVectorizerWrapper
from src.modeling import run_comparison, pick_best, evaluate_fitted_model
from src.split_integrity import filter_evaluation_split

METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)


def train_text_models(max_features=2000):
    print("Loading data for TEXT model training/comparison...")
    train_df, val_df, test_df, _ = load_dataset(use_sample_if_missing=False)
    val_df = filter_evaluation_split("val", val_df)
    test_df = filter_evaluation_split("test", test_df)
    print(
        f"Leakage-safe evaluation rows: val={len(val_df)}, test={len(test_df)}"
    )

    vec = TextVectorizerWrapper(max_features=max_features)
    X_train = vec.fit_transform(train_df["tweet_text"])
    X_val = vec.transform(val_df["tweet_text"])
    X_test = vec.transform(test_df["tweet_text"])
    vec.save()

    summary = {
        "evaluation_rows": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        }
    }

    # ---- Task 1: Informativeness (binary) -------------------------------
    print("\n=== TASK 1: Informativeness (binary) ===")
    inf_df, inf_models, _ = run_comparison(
        X_train, train_df["label"], X_val, val_df["label"],
        task="binary", feature_type="text", pos_label="informative")
    print(inf_df.to_string())
    best_inf, key = pick_best(inf_df, "binary")
    print(f"-> Best informativeness model by {key}: {best_inf}")
    inf_df.to_csv(os.path.join(METRICS_DIR, "text_informative_validation.csv"))
    inf_test = evaluate_fitted_model(
        inf_models[best_inf], X_test, test_df["label"],
        task="binary", pos_label="informative")
    inf_test.to_frame("Test").to_csv(
        os.path.join(METRICS_DIR, "text_informative_test.csv"))
    with open(os.path.join(MODELS_DIR, "text_inf_clf.pkl"), "wb") as f:
        pickle.dump(inf_models[best_inf], f)
    summary["text_informative"] = {
        "best": best_inf,
        "selection_metric": key,
        "validation_score": float(inf_df.loc[best_inf, key]),
        "test_metrics": {k: float(v) for k, v in inf_test.items()},
    }

    # ---- Task 2: Humanitarian category (8-class) ------------------------
    print("\n=== TASK 2: Humanitarian category (8-class) ===")
    cat_df, cat_models, _ = run_comparison(
        X_train, train_df["label_top"], X_val, val_df["label_top"],
        task="multiclass", feature_type="text")
    print(cat_df.to_string())
    best_cat, key2 = pick_best(cat_df, "multiclass")
    print(f"-> Best humanitarian model by {key2}: {best_cat}")
    cat_df.to_csv(os.path.join(METRICS_DIR, "text_humanitarian_validation.csv"))
    cat_test = evaluate_fitted_model(
        cat_models[best_cat], X_test, test_df["label_top"],
        task="multiclass")
    cat_test.to_frame("Test").to_csv(
        os.path.join(METRICS_DIR, "text_humanitarian_test.csv"))
    with open(os.path.join(MODELS_DIR, "text_cat_clf.pkl"), "wb") as f:
        pickle.dump(cat_models[best_cat], f)
    summary["text_humanitarian"] = {
        "best": best_cat,
        "selection_metric": key2,
        "validation_score": float(cat_df.loc[best_cat, key2]),
        "test_metrics": {k: float(v) for k, v in cat_test.items()},
    }

    with open(os.path.join(METRICS_DIR, "text_best_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nText models + comparison saved. Best: {summary}")
    return summary


if __name__ == "__main__":
    train_text_models()
