"""Data forensics: chứng minh dữ liệu được sinh tự động (synthetic).

Gồm: recover công thức tất định, suy luận ngưỡng nhãn, lượng tử hoá duration,
mutual information + permutation test (phân biệt tín hiệu thật vs nhiễu), kiểm
định đồng nhất brand + bootstrap ΔF2. Phục vụ notebook 06. Thuật ngữ (MI,
permutation test, deterministic, ablation…) xem `docs/GLOSSARY.md`.
"""
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import fbeta_score

from pizza_dss.config import (
    COMPACT_CATEGORICAL_FEATURES,
    COMPACT_FEATURE_COLUMNS,
    COMPACT_NUMERIC_FEATURES,
    FIGURES_DIR,
    METRICS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from pizza_dss.data_loader import SIZE_SCORE, load_dataset, load_processed_splits
from pizza_dss.modeling import build_pipeline, metric_row

try:
    from scipy.stats import chi2, chisquare, ks_2samp
except Exception:  # pragma: no cover - scipy is listed in requirements.
    chi2 = None
    chisquare = None
    ks_2samp = None


def _max_abs_error(actual, expected):
    return float((actual.astype(float) - expected.astype(float)).abs().max())


def deterministic_formula_audit(df=None, tolerance=1e-9):
    df = load_dataset() if df is None else df.copy()
    traffic_expected = df["traffic_level"].map({"Low": 1, "Medium": 2, "High": 3})
    rows = [
        {
            "column": "estimated_duration_min",
            "recovered_formula": "2.4 * distance_km",
            "max_abs_error": _max_abs_error(df["estimated_duration_min"], 2.4 * df["distance_km"]),
            "interpretation": "Deterministic transform of distance.",
            "how_found": "Scatter/ratio check: estimated_duration_min divided by distance_km is constant for every row.",
            "verification_method": "Compute max absolute error between the column and 2.4 * distance_km over all rows.",
        },
        {
            "column": "topping_density",
            "recovered_formula": "toppings_count / distance_km",
            "max_abs_error": _max_abs_error(df["topping_density"], df["toppings_count"] / df["distance_km"]),
            "interpretation": "Deterministic transform of toppings and distance.",
            "how_found": "Column name suggests a density; tested toppings_count normalized by distance_km.",
            "verification_method": "Compute max absolute error between the column and toppings_count / distance_km over all rows.",
        },
        {
            "column": "pizza_complexity",
            "recovered_formula": "toppings_count * pizza_size_score",
            "max_abs_error": _max_abs_error(
                df["pizza_complexity"], df["toppings_count"] * df["pizza_size"].map(SIZE_SCORE)
            ),
            "interpretation": "Deterministic transform of toppings and encoded size.",
            "how_found": "Ordinal-encode pizza size as Small=1, Medium=2, Large=3, XL=4, then test products with toppings_count.",
            "verification_method": "Compute max absolute error between the column and toppings_count * size_score over all rows.",
        },
        {
            "column": "traffic_impact",
            "recovered_formula": "Low=1, Medium=2, High=3",
            "max_abs_error": _max_abs_error(df["traffic_impact"], traffic_expected),
            "interpretation": "Ordinal encoding of traffic_level.",
            "how_found": "Unique traffic_impact values are 1/2/3 and align one-to-one with Low/Medium/High.",
            "verification_method": "Map traffic_level to Low=1, Medium=2, High=3 and compare with traffic_impact.",
        },
        {
            "column": "delay_min",
            "recovered_formula": "delivery_duration_min - estimated_duration_min",
            "max_abs_error": _max_abs_error(
                df["delay_min"], df["delivery_duration_min"] - df["estimated_duration_min"]
            ),
            "interpretation": "Post-delivery leakage transform.",
            "how_found": "Delay conventionally means actual duration minus expected duration; tested that identity.",
            "verification_method": "Compute max absolute error between delay_min and delivery_duration_min - estimated_duration_min.",
        },
        {
            "column": "delivery_efficiency_min_per_km",
            "recovered_formula": "delivery_duration_min / distance_km",
            "max_abs_error": _max_abs_error(
                df["delivery_efficiency_min_per_km"], df["delivery_duration_min"] / df["distance_km"]
            ),
            "interpretation": "Post-delivery leakage transform.",
            "how_found": "Column name indicates minutes per kilometer; tested actual duration divided by distance.",
            "verification_method": "Compute max absolute error between the column and delivery_duration_min / distance_km.",
        },
        {
            "column": TARGET_COLUMN,
            "recovered_formula": "delivery_duration_min > 30",
            "max_abs_error": _max_abs_error(
                df[TARGET_COLUMN].astype(int), (df["delivery_duration_min"] > 30).astype(int)
            ),
            "interpretation": "Target is a deterministic threshold over duration.",
            "how_found": "Threshold sweep over delivery_duration_min; the best rules have zero mismatches with the provided label.",
            "verification_method": "Compare is_delayed with candidate threshold rules and record the mismatch count.",
        },
    ]
    out = pd.DataFrame(rows)
    out["matches_exactly"] = out["max_abs_error"] <= tolerance
    return out


def _duration_design_matrix(df, features):
    parts = []
    for feature in features:
        if feature == "traffic_level":
            parts.append(pd.get_dummies(df[feature], prefix=feature, drop_first=True, dtype=float))
        else:
            parts.append(df[[feature]].astype(float))
    return pd.concat(parts, axis=1)


def _duration_model_recovery_with_residuals(df):
    y = df["delivery_duration_min"].astype(float)
    duration_grid_error = (y - (y / 5).round() * 5).abs()
    specs = [
        ("distance_only", ["distance_km"]),
        ("distance_traffic", ["distance_km", "traffic_level"]),
        ("distance_traffic_complexity", ["distance_km", "traffic_level", "pizza_complexity"]),
    ]
    rows = []
    residuals_by_spec = {}
    for spec_name, features in specs:
        X = _duration_design_matrix(df, features)
        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict(X)
        residual = y - prediction
        residuals_by_spec[spec_name] = pd.Series(residual, index=df.index)
        base = {
            "spec": spec_name,
            "r2": float(model.score(X, y)),
            "residual_mean": float(residual.mean()),
            "residual_std": float(residual.std()),
            "residual_abs_mean": float(residual.abs().mean()),
            "duration_grid_min": int(y.min()),
            "duration_grid_max": int(y.max()),
            "duration_grid_step": 5,
            "duration_multiple_of_5_share": float((duration_grid_error <= 1e-9).mean()),
            "unique_duration_values": ",".join(map(str, sorted(y.astype(int).unique()))),
        }
        rows.append({"term": "intercept", "coefficient": float(model.intercept_), **base})
        for term, coefficient in zip(X.columns, model.coef_):
            rows.append({"term": term, "coefficient": float(coefficient), **base})
    return pd.DataFrame(rows), residuals_by_spec["distance_traffic_complexity"]


def duration_model_recovery(df=None):
    df = load_dataset() if df is None else df.copy()
    result, _ = _duration_model_recovery_with_residuals(df)
    return result


def infer_delay_threshold(df=None):
    """Infer the effective duration boundary behind the provided delay label.

    The dataset only exposes ``is_delayed`` as a label. This audit tests simple
    one-dimensional threshold rules against that label instead of assuming a
    business SLA upfront.
    """
    df = load_dataset() if df is None else df.copy()
    duration = df["delivery_duration_min"].astype(float)
    actual = df[TARGET_COLUMN].astype(bool)
    thresholds = sorted(duration.dropna().unique().tolist())
    max_on_time = float(duration[~actual].max())
    min_delayed = float(duration[actual].min())

    rows = []
    for operator in [">", ">="]:
        for threshold in thresholds:
            if operator == ">":
                predicted = duration > threshold
            else:
                predicted = duration >= threshold
            false_positive = int((predicted & ~actual).sum())
            false_negative = int((~predicted & actual).sum())
            mismatches = false_positive + false_negative
            rows.append(
                {
                    "rule": f"delivery_duration_min {operator} {threshold:g}",
                    "operator": operator,
                    "threshold_minutes": float(threshold),
                    "mismatches": mismatches,
                    "mismatch_rate": float(mismatches / len(df)),
                    "accuracy": float(1 - mismatches / len(df)),
                    "false_positives": false_positive,
                    "false_negatives": false_negative,
                    "predicted_delayed": int(predicted.sum()),
                    "actual_delayed": int(actual.sum()),
                    "exact_match": bool(mismatches == 0),
                    "max_observed_on_time_duration": max_on_time,
                    "min_observed_delayed_duration": min_delayed,
                    "interpretation": (
                        "Observationally exact on this 5-minute duration grid."
                        if mismatches == 0
                        else "Does not reproduce the provided label exactly."
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["mismatches", "operator", "threshold_minutes"]).reset_index(drop=True)


def _mutual_information_bits(x, y):
    work = pd.DataFrame({"x": pd.Series(x).astype(str), "y": pd.Series(y).astype(str)})
    counts = pd.crosstab(work["x"], work["y"])
    total = counts.to_numpy().sum()
    if total == 0:
        return 0.0
    joint = counts.to_numpy(dtype=float) / total
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0
    return float((joint[mask] * np.log2(joint[mask] / expected[mask])).sum())


def _conditional_mi_bits(x, y, group):
    frame = pd.DataFrame({"x": x, "y": y, "group": group})
    total = len(frame)
    if total == 0:
        return 0.0
    score = 0.0
    for _, subset in frame.groupby("group", dropna=False):
        score += len(subset) / total * _mutual_information_bits(subset["x"], subset["y"])
    return float(score)


def feature_information_audit(df=None, weak_threshold_bits=0.02, strong_threshold_bits=0.06):
    df = load_dataset() if df is None else df.copy()
    variables = [
        "payment_method",
        "restaurant_name",
        "location",
        "pizza_type",
        "pizza_size",
        "is_weekend",
        "traffic_level",
    ]
    rows = []
    for variable in variables:
        raw_mi = _mutual_information_bits(df[variable], df[TARGET_COLUMN])
        conditional_mi = _conditional_mi_bits(df[variable], df[TARGET_COLUMN], df["distance_band"])
        if conditional_mi >= strong_threshold_bits:
            verdict = "strong_artifact_signal_after_distance_control"
        elif conditional_mi >= weak_threshold_bits:
            verdict = "weak_artifact_or_sampling_signal_after_distance_control"
        else:
            verdict = "noise_after_distance_control"
        rows.append(
            {
                "feature": variable,
                "unique_values": int(df[variable].nunique(dropna=True)),
                "raw_mi_bits": raw_mi,
                "conditional_mi_given_distance_band_bits": conditional_mi,
                "mi_drop_after_distance_control_bits": raw_mi - conditional_mi,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "conditional_mi_given_distance_band_bits", ascending=False
    )


def _chi_square_gof(observed, expected):
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    expected = expected * observed.sum() / expected.sum()
    if chisquare is None:
        statistic = float(((observed - expected) ** 2 / np.where(expected == 0, np.nan, expected)).sum())
        p_value = np.nan
    else:
        result = chisquare(observed, f_exp=expected)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    return statistic, p_value


def uniformity_tests(df=None):
    df = load_dataset() if df is None else df.copy()
    rows = []
    for column in ["restaurant_name", "location", "pizza_type", "pizza_size"]:
        counts = df[column].value_counts().sort_index()
        statistic, p_value = _chi_square_gof(counts.values, np.full(len(counts), len(df) / len(counts)))
        rows.append(
            {
                "variable": column,
                "test": "chi-square GOF vs uniform over observed categories",
                "categories": int(len(counts)),
                "observed_counts": json.dumps(counts.astype(int).to_dict(), ensure_ascii=False),
                "chi2": statistic,
                "p_value": p_value,
                "verdict": "consistent_with_uniform" if pd.isna(p_value) or p_value >= 0.05 else "not_uniform",
            }
        )

    topping_counts = df["toppings_count"].value_counts().reindex(range(1, 6), fill_value=0).sort_index()
    statistic, p_value = _chi_square_gof(topping_counts.values, np.full(5, len(df) / 5))
    rows.append(
        {
            "variable": "toppings_count",
            "test": "chi-square GOF vs uniform 1-5",
            "categories": 5,
            "observed_counts": json.dumps({int(k): int(v) for k, v in topping_counts.items()}),
            "chi2": statistic,
            "p_value": p_value,
            "verdict": "consistent_with_uniform" if pd.isna(p_value) or p_value >= 0.05 else "not_uniform",
        }
    )

    p_hat = float(np.clip((df["toppings_count"].mean() - 1) / 4, 1e-6, 1 - 1e-6))
    for label, p in [("1 + Binomial(4, 0.5)", 0.5), ("1 + Binomial(4, p_hat)", p_hat)]:
        probabilities = np.array([math.comb(4, k - 1) * (p ** (k - 1)) * ((1 - p) ** (5 - k)) for k in range(1, 6)])
        statistic, p_value = _chi_square_gof(topping_counts.values, probabilities * len(df))
        rows.append(
            {
                "variable": "toppings_count",
                "test": f"chi-square GOF vs {label}",
                "categories": 5,
                "observed_counts": json.dumps({int(k): int(v) for k, v in topping_counts.items()}),
                "chi2": statistic,
                "p_value": p_value,
                "verdict": "consistent_with_binomial" if pd.isna(p_value) or p_value >= 0.05 else "not_binomial",
            }
        )
    return pd.DataFrame(rows)


def _normal_delay_ci(delayed, count, z=1.96):
    rate = delayed / count if count else 0.0
    se = math.sqrt(max(rate * (1 - rate) / count, 0)) if count else 0.0
    return max(0.0, rate - z * se), min(1.0, rate + z * se)


def _distribution_chi_square_p(group_counts, pooled_counts):
    aligned = pooled_counts.reindex(group_counts.index, fill_value=0).astype(float)
    expected = aligned / aligned.sum() * group_counts.sum()
    statistic, p_value = _chi_square_gof(group_counts.values, expected.values)
    return statistic, p_value


def brand_homogeneity_tests(df=None):
    df = load_dataset() if df is None else df.copy()
    pooled_delay = float(df[TARGET_COLUMN].mean())
    pooled_size = df["pizza_size"].value_counts().sort_index()
    pooled_type = df["pizza_type"].value_counts().sort_index()
    rows = []
    for brand, group in df.groupby("restaurant_name"):
        rest = df[df["restaurant_name"] != brand]
        delayed = int(group[TARGET_COLUMN].sum())
        count = int(len(group))
        ci_low, ci_high = _normal_delay_ci(delayed, count)
        if ks_2samp is None:
            ks_stat = np.nan
            ks_p = np.nan
        else:
            ks = ks_2samp(group["distance_km"], rest["distance_km"])
            ks_stat = float(ks.statistic)
            ks_p = float(ks.pvalue)
        size_stat, size_p = _distribution_chi_square_p(
            group["pizza_size"].value_counts().sort_index(), pooled_size
        )
        type_stat, type_p = _distribution_chi_square_p(
            group["pizza_type"].value_counts().sort_index(), pooled_type
        )
        rows.append(
            {
                "restaurant_name": brand,
                "orders": count,
                "delayed": delayed,
                "delay_rate": delayed / count,
                "delay_rate_ci_low": ci_low,
                "delay_rate_ci_high": ci_high,
                "pooled_delay_rate": pooled_delay,
                "ks_distance_stat": ks_stat,
                "ks_distance_p_value": ks_p,
                "size_chi2": size_stat,
                "size_p_value": size_p,
                "type_chi2": type_stat,
                "type_p_value": type_p,
                "verdict": (
                    "brand_exchangeable"
                    if (pd.isna(ks_p) or ks_p >= 0.05) and size_p >= 0.05 and type_p >= 0.05
                    else "review_brand_difference"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("restaurant_name")


def _positive_probability(model, X):
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(True)]


def brand_ablation():
    train_df, dev_df, _ = load_processed_splits()
    specs = [
        {
            "feature_set": "compact_with_restaurant",
            "feature_columns": COMPACT_FEATURE_COLUMNS,
            "numeric_features": COMPACT_NUMERIC_FEATURES,
            "categorical_features": COMPACT_CATEGORICAL_FEATURES,
        },
        {
            "feature_set": "compact_without_restaurant",
            "feature_columns": [col for col in COMPACT_FEATURE_COLUMNS if col != "restaurant_name"],
            "numeric_features": COMPACT_NUMERIC_FEATURES,
            "categorical_features": [col for col in COMPACT_CATEGORICAL_FEATURES if col != "restaurant_name"],
        },
    ]
    rows = []
    for spec in specs:
        model = build_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            spec["numeric_features"],
            spec["categorical_features"],
        )
        model.fit(train_df[spec["feature_columns"]], train_df[TARGET_COLUMN])
        y_pred = model.predict(dev_df[spec["feature_columns"]])
        y_prob = _positive_probability(model, dev_df[spec["feature_columns"]])
        row = metric_row(spec["feature_set"], dev_df[TARGET_COLUMN], y_pred, y_prob)
        row["feature_count"] = len(spec["feature_columns"])
        rows.append(row)
    out = pd.DataFrame(rows)
    with_restaurant = out.loc[out["model"] == "compact_with_restaurant"].iloc[0]
    out["delta_f2_vs_with_restaurant"] = out["f2"] - float(with_restaurant["f2"])
    out["delta_mcc_vs_with_restaurant"] = out["mcc"] - float(with_restaurant["mcc"])
    out["interpretation"] = np.where(
        out["model"].eq("compact_without_restaurant") & (out["delta_f2_vs_with_restaurant"].abs() <= 0.02),
        "Restaurant name adds negligible dev F2 in the compact model.",
        np.where(
            out["model"].eq("compact_without_restaurant") & (out["delta_f2_vs_with_restaurant"].abs() <= 0.05),
            "Restaurant name has a small measurable dev F2 artifact; do not read it as real brand quality.",
            "Reference model or non-negligible difference.",
        ),
    )
    return out.sort_values("model")


def pooled_chain_summary(df=None):
    df = load_dataset() if df is None else df.copy()
    rows = [
        {
            "section": "overall",
            "key": "all_restaurants_collapsed",
            "orders": int(len(df)),
            "share": 1.0,
            "delay_rate": float(df[TARGET_COLUMN].mean()),
            "avg_distance_km": float(df["distance_km"].mean()),
            "note": f"{df['restaurant_name'].nunique()} brands treated as one chain.",
        }
    ]
    for section, column in [
        ("size_mix", "pizza_size_label"),
        ("type_mix", "pizza_type"),
        ("monthly_demand", "order_period"),
    ]:
        grouped = (
            df.groupby(column)
            .agg(
                orders=("order_id", "count"),
                delay_rate=(TARGET_COLUMN, "mean"),
                avg_distance_km=("distance_km", "mean"),
            )
            .reset_index()
            .sort_values(column)
        )
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "section": section,
                    "key": row[column],
                    "orders": int(row["orders"]),
                    "share": float(row["orders"] / len(df)),
                    "delay_rate": float(row["delay_rate"]),
                    "avg_distance_km": float(row["avg_distance_km"]),
                    "note": "Computed after collapsing restaurant_name into a single chain.",
                }
            )
    return pd.DataFrame(rows)


def mi_permutation_audit(df=None, n_perm=200, seed=RANDOM_STATE):
    """Permutation baseline for conditional MI.

    Plug-in mutual information is biased upward for high-cardinality features and
    for small per-group samples. By shuffling the label many times and recomputing
    conditional MI we get the *noise floor*: if the observed value is not above the
    permutation null, the "signal" is an estimation artifact, not real structure.
    """
    df = load_dataset() if df is None else df.copy()
    rng = np.random.default_rng(seed)
    variables = ["payment_method", "restaurant_name", "location", "pizza_type", "pizza_size", "is_weekend", "traffic_level"]
    y = df[TARGET_COLUMN]
    group = df["distance_band"]
    rows = []
    for variable in variables:
        x = df[variable]
        observed = _conditional_mi_bits(x, y, group)
        null = np.array([
            _conditional_mi_bits(x, pd.Series(rng.permutation(y.to_numpy()), index=y.index), group)
            for _ in range(n_perm)
        ])
        p_value = float((np.sum(null >= observed) + 1) / (n_perm + 1))
        rows.append(
            {
                "feature": variable,
                "unique_values": int(x.nunique(dropna=True)),
                "raw_mi_bits": _mutual_information_bits(x, y),
                "conditional_mi_bits": observed,
                "permutation_null_mean_bits": float(null.mean()),
                "permutation_null_p95_bits": float(np.percentile(null, 95)),
                "excess_over_null_bits": float(observed - null.mean()),
                "p_value": p_value,
                "verdict": "signal_beyond_chance" if p_value < 0.05 else "within_permutation_noise_floor",
            }
        )
    return pd.DataFrame(rows).sort_values("p_value")


def duration_generator_reconstruction(df=None):
    """Reconstruct the duration generator: round_to_5(base + noise).

    Fits duration on distance (+traffic) and reports how often rounding the linear
    prediction to the 5-minute grid reproduces the observed duration, plus the
    residual noise scale.
    """
    df = load_dataset() if df is None else df.copy()
    y = df["delivery_duration_min"].astype(float)
    X = _duration_design_matrix(df, ["distance_km", "traffic_level"])
    model = LinearRegression().fit(X, y)
    prediction = model.predict(X)
    residual = y - prediction
    reconstructed = np.round(prediction / 5) * 5
    rows = [
        {"quantity": "linear_r2", "value": round(float(model.score(X, y)), 4),
         "meaning": "Phần phương sai duration giải thích bởi distance+traffic."},
        {"quantity": "residual_std_min", "value": round(float(residual.std()), 4),
         "meaning": "Quy mô nhiễu quanh phần xác định (phút)."},
        {"quantity": "residual_within_2_5_min_share", "value": round(float((residual.abs() <= 2.5).mean()), 4),
         "meaning": "Tỷ lệ residual nằm trong nửa ô lưới 5 phút."},
        {"quantity": "round5_reconstruction_accuracy", "value": round(float((reconstructed == y).mean()), 4),
         "meaning": "Tỷ lệ round5(dự đoán tuyến tính) khớp đúng duration thật."},
        {"quantity": "intercept_min", "value": round(float(model.intercept_), 4),
         "meaning": "Thời gian nền khi distance=0, traffic gốc."},
    ]
    return pd.DataFrame(rows)


def bootstrap_brand_delta_f2(n_boot=2000, seed=RANDOM_STATE):
    """Bootstrap CI for the dev F2 gap from adding restaurant_name.

    Both models are fit once on train; dev rows are resampled to get the
    distribution of (F2_with - F2_without). If the CI spans 0, the brand effect is
    within noise.
    """
    train_df, dev_df, _ = load_processed_splits()
    rng = np.random.default_rng(seed)
    with_cols = COMPACT_FEATURE_COLUMNS
    without_cols = [c for c in COMPACT_FEATURE_COLUMNS if c != "restaurant_name"]
    without_cat = [c for c in COMPACT_CATEGORICAL_FEATURES if c != "restaurant_name"]

    model_with = build_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        COMPACT_NUMERIC_FEATURES, COMPACT_CATEGORICAL_FEATURES,
    ).fit(train_df[with_cols], train_df[TARGET_COLUMN])
    model_without = build_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        COMPACT_NUMERIC_FEATURES, without_cat,
    ).fit(train_df[without_cols], train_df[TARGET_COLUMN])

    y = dev_df[TARGET_COLUMN].to_numpy()
    pred_with = model_with.predict(dev_df[with_cols].reset_index(drop=True))
    pred_without = model_without.predict(dev_df[without_cols].reset_index(drop=True))
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        f2_with = fbeta_score(y[idx], pred_with[idx], beta=2, zero_division=0)
        f2_without = fbeta_score(y[idx], pred_without[idx], beta=2, zero_division=0)
        deltas.append(f2_with - f2_without)
    deltas = np.array(deltas)
    point = fbeta_score(y, pred_with, beta=2, zero_division=0) - fbeta_score(y, pred_without, beta=2, zero_division=0)
    ci_low, ci_high = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return pd.DataFrame([
        {
            "quantity": "delta_f2_with_minus_without_restaurant",
            "point_estimate": round(float(point), 4),
            "ci_low_2_5": round(ci_low, 4),
            "ci_high_97_5": round(ci_high, 4),
            "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
            "n_boot": int(len(deltas)),
            "verdict": "within_noise" if ci_low <= 0 <= ci_high else "measurable_artifact",
        }
    ])


def generator_reverse_engineering_summary(
    deterministic,
    threshold_inference,
    duration_recovery,
    information,
    uniformity,
    brand_tests,
    ablation,
    pooled,
):
    no_brand = ablation.loc[ablation["model"] == "compact_without_restaurant"].iloc[0]
    with_brand = ablation.loc[ablation["model"] == "compact_with_restaurant"].iloc[0]
    exact_thresholds = threshold_inference.loc[threshold_inference["exact_match"], "rule"].tolist()
    return {
        "dataset_rows": int(pooled.loc[pooled["section"] == "overall", "orders"].iloc[0]),
        "deterministic_columns_exact": deterministic.loc[
            deterministic["matches_exactly"], "column"
        ].tolist(),
        "delay_label_threshold_inference": {
            "exact_rules_on_observed_grid": exact_thresholds,
            "max_observed_on_time_duration": float(
                threshold_inference["max_observed_on_time_duration"].iloc[0]
            ),
            "min_observed_delayed_duration": float(
                threshold_inference["min_observed_delayed_duration"].iloc[0]
            ),
            "note": (
                "The source file provides is_delayed, not an SLA document. "
                "Because duration is snapped to 5-minute values, "
                "delivery_duration_min > 30 and delivery_duration_min >= 35 are "
                "observationally equivalent in this dataset."
            ),
        },
        "duration_rule": {
            "unique_values": sorted(load_dataset()["delivery_duration_min"].astype(int).unique().tolist()),
            "grid_step_minutes": 5,
            "multiple_of_5_share": float(duration_recovery["duration_multiple_of_5_share"].max()),
            "best_linear_spec_r2": float(
                duration_recovery.groupby("spec")["r2"].first().max()
            ),
        },
        "categorical_information_verdict": {
            row["feature"]: row["verdict"] for _, row in information.iterrows()
        },
        "uniformity_rejections_at_0_05": uniformity.loc[
            uniformity["p_value"].fillna(1.0) < 0.05, "variable"
        ].tolist(),
        "brand_verdict": {
            "brand_counts_min": int(brand_tests["orders"].min()),
            "brand_counts_max": int(brand_tests["orders"].max()),
            "brand_exchangeable_rows": int((brand_tests["verdict"] == "brand_exchangeable").sum()),
            "dev_f2_with_restaurant": float(with_brand["f2"]),
            "dev_f2_without_restaurant": float(no_brand["f2"]),
            "delta_f2_without_restaurant": float(no_brand["delta_f2_vs_with_restaurant"]),
        },
        "pseudo_spec": [
            "distance_km is sampled from a short discrete menu of 25 values.",
            "estimated_duration_min = 2.4 * distance_km.",
            "delivery_duration_min is snapped to a 5-minute grid from 15 to 50 minutes.",
            "The effective label boundary lies between 30 and 35 minutes: delivery_duration_min > 30 is equivalent to delivery_duration_min >= 35 on the observed grid.",
            "topping_density, pizza_complexity, traffic_impact, delay_min and delivery_efficiency are deterministic transforms.",
            "Some categorical columns still carry weak or strong artifact signal after distance-band control, so the generator is not purely independent uniform draws.",
            "The five restaurant brands can be collapsed for chain-level descriptive reporting, but homogeneity tests reject treating brands as statistically identical.",
        ],
    }


def _write_forensics_figures(
    deterministic,
    threshold_inference,
    residuals,
    information,
    uniformity,
    brand_tests,
):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = deterministic.sort_values("max_abs_error", ascending=True)
    ax.barh(plot_df["column"], plot_df["max_abs_error"])
    ax.set_title("Recovered deterministic formula errors")
    ax.set_xlabel("Max absolute error")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "generator_deterministic_formula_errors.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = threshold_inference.pivot_table(
        index="threshold_minutes", columns="operator", values="mismatches", aggfunc="min"
    ).sort_index()
    for operator in pivot.columns:
        ax.plot(pivot.index, pivot[operator], marker="o", label=f"duration {operator} threshold")
    ax.axvspan(30, 35, color="green", alpha=0.12, label="effective boundary")
    ax.set_title("Inferring the delayed-label threshold")
    ax.set_xlabel("Threshold minutes")
    ax.set_ylabel("Mismatches vs provided is_delayed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "delay_threshold_inference.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=20, edgecolor="white")
    for marker in range(int(math.floor(residuals.min() / 5) * 5), int(math.ceil(residuals.max() / 5) * 5) + 1, 5):
        ax.axvline(marker, color="black", alpha=0.15, linewidth=0.8)
    ax.set_title("Duration OLS residuals against 5-minute grid")
    ax.set_xlabel("Residual minutes")
    ax.set_ylabel("Orders")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "duration_model_residual_histogram.png", dpi=160)
    plt.close(fig)

    info_plot = information.sort_values("raw_mi_bits", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(info_plot))
    ax.barh(y - 0.18, info_plot["raw_mi_bits"], height=0.36, label="Raw MI")
    ax.barh(
        y + 0.18,
        info_plot["conditional_mi_given_distance_band_bits"],
        height=0.36,
        label="MI | distance band",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(info_plot["feature"])
    ax.set_title("Categorical feature information after distance control")
    ax.set_xlabel("Mutual information (bits)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_information_audit.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    p_values = uniformity["p_value"].fillna(1.0).clip(lower=1e-12)
    labels = uniformity["variable"] + "\n" + uniformity["test"].str.replace("chi-square GOF vs ", "", regex=False)
    ax.bar(range(len(uniformity)), -np.log10(p_values))
    ax.axhline(-math.log10(0.05), color="red", linestyle="--", linewidth=1, label="p=0.05")
    ax.set_xticks(range(len(uniformity)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Uniformity goodness-of-fit tests")
    ax.set_ylabel("-log10(p-value)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "uniformity_tests.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = brand_tests.sort_values("restaurant_name")
    lower = ordered["delay_rate"] - ordered["delay_rate_ci_low"]
    upper = ordered["delay_rate_ci_high"] - ordered["delay_rate"]
    ax.errorbar(
        ordered["restaurant_name"],
        ordered["delay_rate"],
        yerr=[lower, upper],
        fmt="o",
        capsize=4,
    )
    ax.axhline(ordered["pooled_delay_rate"].iloc[0], color="red", linestyle="--", label="Pooled chain mean")
    ax.set_title("Brand delay-rate homogeneity")
    ax.set_ylabel("Delay rate with 95% CI")
    ax.set_xlabel("Restaurant brand")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "brand_delay_rate_homogeneity.png", dpi=160)
    plt.close(fig)


def build_forensics_artifacts():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    deterministic = deterministic_formula_audit(df)
    deterministic.round(10).to_csv(METRICS_DIR / "generator_deterministic_formulas.csv", index=False)

    threshold_inference = infer_delay_threshold(df)
    threshold_inference.round(10).to_csv(METRICS_DIR / "delay_threshold_inference.csv", index=False)

    duration, residuals = _duration_model_recovery_with_residuals(df)
    duration.round(10).to_csv(METRICS_DIR / "duration_model_recovery.csv", index=False)

    information = feature_information_audit(df)
    information.round(10).to_csv(METRICS_DIR / "feature_information_audit.csv", index=False)

    uniformity = uniformity_tests(df)
    uniformity.round(10).to_csv(METRICS_DIR / "uniformity_tests.csv", index=False)

    brand_tests = brand_homogeneity_tests(df)
    brand_tests.round(10).to_csv(METRICS_DIR / "brand_homogeneity_tests.csv", index=False)

    ablation = brand_ablation()
    ablation.round(10).to_csv(METRICS_DIR / "brand_ablation.csv", index=False)

    pooled = pooled_chain_summary(df)
    pooled.round(10).to_csv(METRICS_DIR / "pooled_chain_summary.csv", index=False)

    mi_permutation_audit(df).round(10).to_csv(METRICS_DIR / "mi_permutation_audit.csv", index=False)
    duration_generator_reconstruction(df).to_csv(METRICS_DIR / "duration_generator_reconstruction.csv", index=False)
    bootstrap_brand_delta_f2().to_csv(METRICS_DIR / "brand_delta_f2_bootstrap.csv", index=False)

    summary = generator_reverse_engineering_summary(
        deterministic, threshold_inference, duration, information, uniformity, brand_tests, ablation, pooled
    )
    (METRICS_DIR / "generator_reverse_engineering_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _write_forensics_figures(deterministic, threshold_inference, residuals, information, uniformity, brand_tests)
    return summary


if __name__ == "__main__":
    print(build_forensics_artifacts())
