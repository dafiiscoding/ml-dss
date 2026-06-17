"""Đọc dữ liệu Kaggle, chuẩn hoá schema, tạo feature kỹ thuật, và chia
train/dev/test **chống rò rỉ (leakage)**.

Luồng chính cho người mới: `load_dataset()` -> `split_dataset()` ->
`export_processed_splits()`. `validate_feature_contract()` chặn mọi cột biết-sau-
giao-hàng. Thuật ngữ xem `docs/GLOSSARY.md`.
"""
import json
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from pizza_dss.config import (
    COMPACT_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    KAGGLE_DOWNLOAD_URL,
    LEAKAGE_COLUMNS,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    RAW_EXCEL_PATH,
    TARGET_COLUMN,
)

COLUMN_RENAME = {
    "Order ID": "order_id",
    "Restaurant Name": "restaurant_name",
    "Location": "location",
    "Order Time": "order_time",
    "Delivery Time": "delivery_time",
    "Delivery Duration (min)": "delivery_duration_min",
    "Pizza Size": "pizza_size",
    "Pizza Type": "pizza_type",
    "Toppings Count": "toppings_count",
    "Distance (km)": "distance_km",
    "Traffic Level": "traffic_level",
    "Payment Method": "payment_method",
    "Is Peak Hour": "is_peak_hour",
    "Is Weekend": "is_weekend",
    "Delivery Efficiency (min/km)": "delivery_efficiency_min_per_km",
    "Topping Density": "topping_density",
    "Order Month": "order_month",
    "Payment Category": "payment_category",
    "Estimated Duration (min)": "estimated_duration_min",
    "Delay (min)": "delay_min",
    "Is Delayed": "is_delayed",
    "Pizza Complexity": "pizza_complexity",
    "Traffic Impact": "traffic_impact",
    "Order Hour": "order_hour",
    "Restaurant Avg Time": "restaurant_avg_time",
}

SIZE_SCORE = {"Small": 1, "Medium": 2, "Large": 3, "XL": 4}


def add_engineered_features(df):
    out = df.copy()
    out["pizza_size_score"] = out["pizza_size"].map(SIZE_SCORE).astype("int64")
    out["pizza_size_code"] = out["pizza_size_score"].map(lambda value: f"{value:02d}")
    out["pizza_size_label"] = out["pizza_size_code"] + "_" + out["pizza_size"]
    out["order_period"] = out["order_time"].dt.to_period("M").astype(str)
    out["order_weekday"] = out["order_time"].dt.dayofweek
    out["order_weekday_name"] = out["order_time"].dt.day_name()
    out["time_segment"] = pd.cut(
        out["order_hour"],
        bins=[0, 11, 15, 17, 21, 24],
        labels=["Other", "Lunch", "Afternoon", "Dinner", "Late"],
        right=False,
        include_lowest=True,
    ).astype(str)
    out["complexity_band"] = pd.cut(
        out["pizza_complexity"],
        bins=[0, 4, 8, 12, 20],
        labels=["Low", "Medium", "High", "Very High"],
        include_lowest=True,
    ).astype(str)
    out["distance_band"] = pd.cut(
        out["distance_km"],
        bins=[0, 3, 6, 8, 11],
        labels=["0-3 km", "3-6 km", "6-8 km", "8+ km"],
        include_lowest=True,
    ).astype(str)
    return out


def download_dataset(force=False):
    """Download and extract the public Kaggle dataset via Kaggle's public API."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_EXCEL_PATH.exists() and not force:
        return RAW_EXCEL_PATH

    import urllib.request

    zip_path = RAW_DATA_DIR / "pizza_delivery_kaggle.zip"
    urllib.request.urlretrieve(KAGGLE_DOWNLOAD_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(RAW_DATA_DIR)
    if not RAW_EXCEL_PATH.exists():
        raise FileNotFoundError(f"Expected {RAW_EXCEL_PATH} after extraction.")
    return RAW_EXCEL_PATH


def ensure_dataset():
    if not RAW_EXCEL_PATH.exists():
        download_dataset()
    return RAW_EXCEL_PATH


def normalize_schema(df):
    out = df.rename(columns=COLUMN_RENAME).copy()
    missing = sorted(set(COLUMN_RENAME.values()) - set(out.columns))
    if missing:
        raise ValueError(f"Missing expected columns after rename: {missing}")

    out["restaurant_name"] = out["restaurant_name"].replace(
        {"Marco\u2019s Pizza": "Marco's Pizza"}
    )
    out["order_time"] = pd.to_datetime(out["order_time"])
    out["delivery_time"] = pd.to_datetime(out["delivery_time"])
    out["order_date"] = out["order_time"].dt.date.astype(str)
    out["order_year"] = out["order_time"].dt.year
    out["order_month_num"] = out["order_time"].dt.month
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(bool)
    out["is_peak_hour"] = out["is_peak_hour"].astype(bool)
    out["is_weekend"] = out["is_weekend"].astype(bool)
    return add_engineered_features(out)


def load_raw_data():
    path = ensure_dataset()
    return pd.read_excel(path)


def load_dataset():
    return normalize_schema(load_raw_data())


def validate_feature_contract(feature_columns=None):
    feature_columns = feature_columns or FEATURE_COLUMNS
    leaked = sorted(set(feature_columns) & set(LEAKAGE_COLUMNS + [TARGET_COLUMN]))
    if leaked:
        raise ValueError(f"Leakage columns cannot be model features: {leaked}")
    return True


def split_dataset(df=None, random_state=RANDOM_STATE):
    """Create a leakage-safe stratified train/dev/test split.

    A chronological last-20-percent split has no delayed samples in this file, so
    the coursework protocol uses fixed stratified splits and reports this choice
    in the data audit.
    """
    df = load_dataset() if df is None else df.copy()
    train_df, temp_df = train_test_split(
        df,
        test_size=0.40,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df[TARGET_COLUMN],
    )
    return (
        train_df.sort_values("order_id").reset_index(drop=True).assign(split="train"),
        dev_df.sort_values("order_id").reset_index(drop=True).assign(split="dev"),
        test_df.sort_values("order_id").reset_index(drop=True).assign(split="test"),
    )


def export_processed_splits():
    validate_feature_contract()
    train_df, dev_df, test_df = split_dataset()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train_df), ("dev", dev_df), ("test", test_df)):
        frame.to_csv(PROCESSED_DATA_DIR / f"{name}.csv", index=False)
    return {"train": len(train_df), "dev": len(dev_df), "test": len(test_df)}


def load_processed_splits():
    paths = [PROCESSED_DATA_DIR / f"{name}.csv" for name in ("train", "dev", "test")]
    required_columns = set(FEATURE_COLUMNS + COMPACT_FEATURE_COLUMNS + [TARGET_COLUMN])
    needs_rebuild = not all(path.exists() for path in paths)
    if not needs_rebuild:
        for path in paths:
            columns = set(pd.read_csv(path, nrows=0).columns)
            if required_columns - columns:
                needs_rebuild = True
                break
    if needs_rebuild:
        export_processed_splits()
    frames = []
    for path in paths:
        frame = pd.read_csv(path, parse_dates=["order_time", "delivery_time"])
        for col in ("is_peak_hour", "is_weekend", TARGET_COLUMN):
            if frame[col].dtype != bool:
                frame[col] = frame[col].astype(str).str.lower().map({"true": True, "false": False})
        frames.append(frame)
    return tuple(frames)


def audit_dataset(df=None):
    df = load_dataset() if df is None else df.copy()
    chronological_cut = int(len(df.sort_values("order_time")) * 0.8)
    chronological_test = df.sort_values("order_time").iloc[chronological_cut:]
    duration_rule = df["delivery_duration_min"] > 30
    delay_formula_error = (
        df["delay_min"] - (df["delivery_duration_min"] - df["estimated_duration_min"])
    ).abs().max()

    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_order_ids": int(df["order_id"].duplicated().sum()),
        "date_min": str(df["order_time"].min()),
        "date_max": str(df["order_time"].max()),
        "years": {str(k): int(v) for k, v in df["order_year"].value_counts().sort_index().items()},
        "target_distribution": {
            str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().items()
        },
        "delayed_rate": round(float(df[TARGET_COLUMN].mean()), 4),
        "is_delayed_equals_duration_gt_30": bool((df[TARGET_COLUMN] == duration_rule).all()),
        "max_delay_formula_error": round(float(delay_formula_error), 10),
        "chronological_last20_rows": int(len(chronological_test)),
        "chronological_last20_delayed": int(chronological_test[TARGET_COLUMN].sum()),
        "feature_columns": FEATURE_COLUMNS,
        "blocked_leakage_columns": LEAKAGE_COLUMNS,
    }
    return summary


def write_data_audit():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = audit_dataset()
    with open(METRICS_DIR / "data_quality_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    print(write_data_audit())
    print(export_processed_splits())
