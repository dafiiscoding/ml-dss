"""Cấu hình trung tâm: đường dẫn thư mục, danh sách feature, cột cấm (leakage),
seed ngẫu nhiên và ngưỡng priority. Sửa ở ĐÂY thay vì rải rác trong code.

Thuật ngữ (leakage, feature, stratified…) xem `docs/GLOSSARY.md`.
"""
from pathlib import Path

RANDOM_STATE = 42  # seed cố định để mọi lần chạy ra kết quả giống nhau (tái lập)
ANALYSIS_DATE = "2026-06-15"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

KAGGLE_DATASET_REF = "akshaygaikwad448/pizza-delivery-data-with-enhanced-features"
KAGGLE_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "akshaygaikwad448/pizza-delivery-data-with-enhanced-features"
)
RAW_EXCEL_NAME = "Enhanced_pizza_sell_data_2024-25.xlsx"
RAW_EXCEL_PATH = RAW_DATA_DIR / RAW_EXCEL_NAME

TARGET_COLUMN = "is_delayed"

NUMERIC_FEATURES = [
    "toppings_count",
    "distance_km",
    "topping_density",
    "estimated_duration_min",
    "pizza_complexity",
    "traffic_impact",
    "order_hour",
]

CATEGORICAL_FEATURES = [
    "restaurant_name",
    "location",
    "pizza_size",
    "pizza_type",
    "traffic_level",
    "payment_method",
    "is_peak_hour",
    "is_weekend",
    "order_month",
    "payment_category",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

COMPACT_NUMERIC_FEATURES = [
    "toppings_count",
    "distance_km",
    "order_hour",
    "pizza_size_score",
]

COMPACT_CATEGORICAL_FEATURES = [
    "restaurant_name",
    "location",
    "pizza_type",
    "traffic_level",
    "payment_category",
    "is_peak_hour",
    "is_weekend",
    "order_month",
]

COMPACT_FEATURE_COLUMNS = COMPACT_NUMERIC_FEATURES + COMPACT_CATEGORICAL_FEATURES

ACTIVE_NUMERIC_FEATURES = COMPACT_NUMERIC_FEATURES
ACTIVE_CATEGORICAL_FEATURES = COMPACT_CATEGORICAL_FEATURES
ACTIVE_FEATURE_COLUMNS = COMPACT_FEATURE_COLUMNS

LEAKAGE_COLUMNS = [
    "delivery_time",
    "delivery_duration_min",
    "delivery_efficiency_min_per_km",
    "delay_min",
    "restaurant_avg_time",
]

PRIORITY_LOW_MAX = 35
PRIORITY_MEDIUM_MAX = 65
