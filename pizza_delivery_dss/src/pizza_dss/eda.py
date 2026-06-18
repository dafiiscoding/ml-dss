"""EDA (phân tích khám phá): delay rate theo nhóm + Wilson CI, kiểm định,
clustering (K-Means) và association rules, kèm các hàm sinh hình.

Mỗi hàm trả về bảng/figure để notebook 02 trình bày. Nếu máy chặn
`sklearn.cluster` (WDAC), `_SimpleKMeans` thuần numpy được dùng thay thế. Thuật
ngữ (delay rate, Wilson CI, silhouette, lift…) xem `docs/GLOSSARY.md`.
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from pizza_dss.config import FEATURE_COLUMNS, FIGURES_DIR, METRICS_DIR, RANDOM_STATE, TARGET_COLUMN
from pizza_dss.data_loader import load_dataset
from pizza_dss.features import build_preprocessor

try:  # WDAC may block sklearn native neighbor/cluster DLLs; fall back to numpy.
    from sklearn.cluster import KMeans
except ImportError:  # pragma: no cover - environment-dependent
    KMeans = None


class _SimpleKMeans:
    """Pure-numpy Lloyd K-Means fallback used only when sklearn.cluster is blocked."""

    def __init__(self, n_clusters=3, random_state=RANDOM_STATE, n_init=10, max_iter=100):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = min(int(n_init), 5)  # cap to keep the fallback fast
        self.max_iter = max_iter

    def fit_predict(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best_labels, best_inertia = None, np.inf
        for _ in range(self.n_init):
            centers = X[rng.choice(len(X), self.n_clusters, replace=False)]
            labels = np.zeros(len(X), dtype=int)
            for _ in range(self.max_iter):
                distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                new_labels = distances.argmin(axis=1)
                new_centers = np.array([
                    X[new_labels == c].mean(axis=0) if np.any(new_labels == c) else centers[c]
                    for c in range(self.n_clusters)
                ])
                if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
                    centers, labels = new_centers, new_labels
                    break
                centers, labels = new_centers, new_labels
            inertia = float(((X - centers[labels]) ** 2).sum())
            if inertia < best_inertia:
                best_inertia, best_labels = inertia, labels
        return best_labels


def _make_kmeans(n_clusters):
    if KMeans is not None:
        return KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    return _SimpleKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)


def delay_rate_by(df, column):
    return (
        df.groupby(column)[TARGET_COLUMN]
        .agg(["count", "sum", "mean"])
        .rename(columns={"sum": "delayed", "mean": "delay_rate"})
        .sort_values("delay_rate", ascending=False)
        .reset_index()
    )


def delay_rate_with_ci(df, column, z=1.96):
    """Delay rate per category with a Wilson 95% confidence interval.

    Wilson is preferred over the normal approximation for small or extreme-rate
    groups, so the table does not over-trust categories with few orders.
    """
    grouped = (
        df.groupby(column)[TARGET_COLUMN]
        .agg(orders="count", delayed="sum", delay_rate="mean")
        .reset_index()
    )
    n = grouped["orders"]
    p = grouped["delay_rate"]
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    grouped["ci_low"] = (center - margin).clip(lower=0)
    grouped["ci_high"] = (center + margin).clip(upper=1)
    grouped["small_sample"] = n < 30
    return grouped.sort_values("delay_rate", ascending=False).reset_index(drop=True)


def kmeans_cluster_profile(df, k=4):
    """Fit K-Means with k clusters and profile each cluster.

    Silhouette tells us how many clusters; this table tells us *what each cluster
    means* operationally (size, delay rate, distance, dominant traffic).
    """
    X = build_preprocessor().fit_transform(df[FEATURE_COLUMNS])
    labels = _make_kmeans(k).fit_predict(X)
    work = df.copy()
    work["cluster"] = labels
    rows = []
    for cluster, group in work.groupby("cluster"):
        rows.append(
            {
                "cluster": int(cluster),
                "orders": int(len(group)),
                "order_share": round(len(group) / len(work), 4),
                "delay_rate": round(float(group[TARGET_COLUMN].mean()), 4),
                "avg_distance_km": round(float(group["distance_km"].mean()), 3),
                "avg_toppings": round(float(group["toppings_count"].mean()), 3),
                "dominant_traffic": group["traffic_level"].value_counts().index[0],
                "dominant_size": group["pizza_size"].value_counts().index[0],
            }
        )
    return pd.DataFrame(rows).sort_values("delay_rate", ascending=False).reset_index(drop=True)


def data_profile(df):
    return {
        "rows": len(df),
        "delayed_rate": float(df[TARGET_COLUMN].mean()),
        "restaurants": int(df["restaurant_name"].nunique()),
        "locations": int(df["location"].nunique()),
        "pizza_types": int(df["pizza_type"].nunique()),
        "traffic_levels": int(df["traffic_level"].nunique()),
    }


def add_distance_band(df):
    out = df.copy()
    out["distance_band"] = pd.cut(
        out["distance_km"],
        bins=[0, 3, 6, 8, 11],
        labels=["0-3 km", "3-6 km", "6-8 km", "8+ km"],
        include_lowest=True,
    )
    return out


def add_location_parts(df):
    """Derive city/state fields from the free-text location column.

    The source data has no official state/bang column. Most values look like
    "City, ST", so these fields are descriptive conveniences, not authoritative
    geocodes.
    """
    out = df.copy()
    extracted = out["location"].astype(str).str.extract(r"^\s*(?P<city>.*?)(?:,\s*(?P<state_code>[A-Z]{2}))?\s*$")
    out["city"] = extracted["city"].str.strip()
    out["state_code"] = extracted["state_code"].fillna("Unknown").str.strip()
    out["state_code"] = out["state_code"].fillna("Unknown")
    return out


def _top_value(series):
    counts = series.value_counts(dropna=False)
    return counts.index[0] if len(counts) else ""


def _top_count(series):
    counts = series.value_counts(dropna=False)
    return int(counts.iloc[0]) if len(counts) else 0


def _share(mask):
    return float(pd.Series(mask).mean()) if len(mask) else 0.0


def duration_delay_profile(df):
    """Profile realized duration and post-hoc delay diagnostics by label."""
    return (
        df.groupby(TARGET_COLUMN)
        .agg(
            orders=("order_id", "count"),
            duration_min=("delivery_duration_min", "min"),
            duration_q25=("delivery_duration_min", lambda s: float(s.quantile(0.25))),
            duration_median=("delivery_duration_min", "median"),
            duration_mean=("delivery_duration_min", "mean"),
            duration_q75=("delivery_duration_min", lambda s: float(s.quantile(0.75))),
            duration_max=("delivery_duration_min", "max"),
            estimated_duration_mean=("estimated_duration_min", "mean"),
            delay_min_mean=("delay_min", "mean"),
            delay_min_median=("delay_min", "median"),
            distance_mean=("distance_km", "mean"),
            complexity_mean=("pizza_complexity", "mean"),
            high_traffic_share=("traffic_level", lambda s: _share(s == "High")),
            peak_hour_share=("is_peak_hour", "mean"),
        )
        .reset_index()
        .sort_values(TARGET_COLUMN)
    )


def duration_grid_by_delay(df):
    rows = (
        df.groupby("delivery_duration_min")
        .agg(
            orders=("order_id", "count"),
            delayed=(TARGET_COLUMN, "sum"),
            delay_rate=(TARGET_COLUMN, "mean"),
            avg_distance_km=("distance_km", "mean"),
            avg_estimated_duration_min=("estimated_duration_min", "mean"),
            avg_delay_min=("delay_min", "mean"),
        )
        .reset_index()
        .sort_values("delivery_duration_min")
    )
    rows["on_time"] = rows["orders"] - rows["delayed"]
    rows["minutes_over_30_boundary"] = rows["delivery_duration_min"] - 30
    return rows


def delay_severity_distribution(df):
    work = df.copy()
    severity_order = ["On-time <=30", "Late 35", "Late 40", "Late 45-50"]
    work["delay_severity"] = pd.cut(
        work["delivery_duration_min"],
        bins=[0, 30, 35, 40, 50],
        labels=severity_order,
        include_lowest=True,
        right=True,
    )
    out = (
        work.groupby("delay_severity")
        .agg(
            orders=("order_id", "count"),
            delayed=(TARGET_COLUMN, "sum"),
            order_share=("order_id", lambda s: len(s) / len(work)),
            avg_distance_km=("distance_km", "mean"),
            avg_complexity=("pizza_complexity", "mean"),
            high_traffic_share=("traffic_level", lambda s: _share(s == "High")),
        )
        .reset_index()
    )
    out["delay_severity"] = out["delay_severity"].astype(str)
    out["_order"] = out["delay_severity"].map({name: i for i, name in enumerate(severity_order)})
    return out.sort_values("_order").drop(columns="_order")


def favorite_item_summary(df):
    total = len(df)
    rows = []
    for dimension, columns in [
        ("pizza_type", ["pizza_type"]),
        ("pizza_size", ["pizza_size_label", "pizza_size"]),
        ("type_size_combo", ["pizza_type", "pizza_size_label"]),
        ("restaurant_type_combo", ["restaurant_name", "pizza_type"]),
        ("state_type_combo", ["state_code", "pizza_type"]),
    ]:
        work = add_location_parts(df) if "state_code" in columns else df.copy()
        grouped = (
            work.groupby(columns)
            .agg(
                orders=("order_id", "count"),
                delayed=(TARGET_COLUMN, "sum"),
                delay_rate=(TARGET_COLUMN, "mean"),
                avg_duration_min=("delivery_duration_min", "mean"),
            )
            .reset_index()
            .sort_values("orders", ascending=False)
        )
        grouped["dimension"] = dimension
        grouped["order_share"] = grouped["orders"] / total
        grouped["key"] = grouped[columns].astype(str).agg(" | ".join, axis=1)
        rows.append(
            grouped[
                [
                    "dimension",
                    "key",
                    "orders",
                    "order_share",
                    "delayed",
                    "delay_rate",
                    "avg_duration_min",
                ]
            ].head(30)
        )
    return pd.concat(rows, ignore_index=True)


def restaurant_dependency_summary(df):
    return (
        df.groupby("restaurant_name")
        .agg(
            orders=("order_id", "count"),
            delayed=(TARGET_COLUMN, "sum"),
            delay_rate=(TARGET_COLUMN, "mean"),
            avg_duration_min=("delivery_duration_min", "mean"),
            avg_distance_km=("distance_km", "mean"),
            avg_complexity=("pizza_complexity", "mean"),
            high_traffic_share=("traffic_level", lambda s: _share(s == "High")),
            peak_hour_share=("is_peak_hour", "mean"),
            weekend_share=("is_weekend", "mean"),
            online_payment_share=("payment_category", lambda s: _share(s == "Online")),
            top_type=("pizza_type", _top_value),
            top_type_orders=("pizza_type", _top_count),
            top_size=("pizza_size", _top_value),
            top_size_orders=("pizza_size", _top_count),
            top_location=("location", _top_value),
            top_location_orders=("location", _top_count),
        )
        .reset_index()
        .sort_values("orders", ascending=False)
    )


def location_dependency_summary(df, min_orders=5):
    work = add_location_parts(df)
    summary = (
        work.groupby(["state_code", "city", "location"])
        .agg(
            orders=("order_id", "count"),
            delayed=(TARGET_COLUMN, "sum"),
            delay_rate=(TARGET_COLUMN, "mean"),
            avg_duration_min=("delivery_duration_min", "mean"),
            avg_distance_km=("distance_km", "mean"),
            high_traffic_share=("traffic_level", lambda s: _share(s == "High")),
            peak_hour_share=("is_peak_hour", "mean"),
            top_restaurant=("restaurant_name", _top_value),
            top_restaurant_orders=("restaurant_name", _top_count),
            top_type=("pizza_type", _top_value),
            top_type_orders=("pizza_type", _top_count),
            top_size=("pizza_size", _top_value),
            top_size_orders=("pizza_size", _top_count),
        )
        .reset_index()
    )
    summary["small_sample_flag"] = summary["orders"] < min_orders
    return summary.sort_values(["orders", "delay_rate"], ascending=[False, False])


def state_dependency_summary(df):
    work = add_location_parts(df)
    return (
        work.groupby("state_code")
        .agg(
            locations=("location", "nunique"),
            orders=("order_id", "count"),
            delayed=(TARGET_COLUMN, "sum"),
            delay_rate=(TARGET_COLUMN, "mean"),
            avg_duration_min=("delivery_duration_min", "mean"),
            avg_distance_km=("distance_km", "mean"),
            high_traffic_share=("traffic_level", lambda s: _share(s == "High")),
            top_city=("city", _top_value),
            top_restaurant=("restaurant_name", _top_value),
            top_type=("pizza_type", _top_value),
            top_size=("pizza_size", _top_value),
        )
        .reset_index()
        .sort_values("orders", ascending=False)
    )


def restaurant_mix_matrix(df, column):
    counts = pd.crosstab(df["restaurant_name"], df[column])
    shares = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    shares.insert(0, "orders", counts.sum(axis=1))
    return shares.reset_index()


def state_mix_matrix(df, column):
    work = add_location_parts(df)
    counts = pd.crosstab(work["state_code"], work[column])
    shares = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    shares.insert(0, "orders", counts.sum(axis=1))
    return shares.reset_index()


def kmeans_silhouette_sweep(df, k_values=range(2, 8)):
    X = build_preprocessor().fit_transform(df[FEATURE_COLUMNS])
    rows = []
    for k in k_values:
        labels = _make_kmeans(k).fit_predict(X)
        rows.append({"k": k, "silhouette": silhouette_score(X, labels)})
    return pd.DataFrame(rows)


def _transaction_for_order(row):
    items = {
        f"traffic={row['traffic_level']}",
        f"size={row['pizza_size']}",
        f"payment={row['payment_category']}",
        f"peak={bool(row['is_peak_hour'])}",
        f"weekend={bool(row['is_weekend'])}",
    }
    if row["distance_km"] >= 7:
        items.add("distance=high")
    elif row["distance_km"] >= 4:
        items.add("distance=medium")
    else:
        items.add("distance=low")
    if row["toppings_count"] >= 4:
        items.add("toppings=high")
    if bool(row[TARGET_COLUMN]):
        items.add("delayed=True")
    else:
        items.add("delayed=False")
    return items


def association_rules_to_delay(df, min_support=0.03):
    transactions = [_transaction_for_order(row) for _, row in df.iterrows()]
    total = len(transactions)
    target = "delayed=True"
    target_count = sum(target in tx for tx in transactions)
    target_support = target_count / total
    items = sorted(set().union(*transactions) - {target, "delayed=False"})
    rows = []
    for size in (1, 2):
        for antecedent in itertools.combinations(items, size):
            antecedent = set(antecedent)
            antecedent_count = sum(antecedent.issubset(tx) for tx in transactions)
            both_count = sum(antecedent.issubset(tx) and target in tx for tx in transactions)
            support = both_count / total
            if antecedent_count == 0 or support < min_support:
                continue
            confidence = both_count / antecedent_count
            lift = confidence / target_support if target_support else 0
            rows.append(
                {
                    "antecedent": ", ".join(sorted(antecedent)),
                    "consequent": target,
                    "support": support,
                    "confidence": confidence,
                    "lift": lift,
                    "matches": both_count,
                }
            )
    return pd.DataFrame(rows).sort_values(["lift", "confidence"], ascending=False)


def build_eda_artifacts():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = add_distance_band(load_dataset())
    df = add_location_parts(df)

    pd.Series(data_profile(df)).to_json(METRICS_DIR / "eda_profile.json", indent=2)
    for col in [
        "traffic_level",
        "pizza_size",
        "distance_band",
        "order_hour",
        "restaurant_name",
        "pizza_type",
        "complexity_band",
        "state_code",
        "payment_method",
    ]:
        delay_rate_by(df, col).to_csv(METRICS_DIR / f"delay_rate_by_{col}.csv", index=False)
    for col in ["traffic_level", "distance_band", "complexity_band", "restaurant_name", "state_code"]:
        delay_rate_with_ci(df, col).round(6).to_csv(
            METRICS_DIR / f"delay_rate_ci_by_{col}.csv", index=False
        )

    duration_delay_profile(df).round(6).to_csv(METRICS_DIR / "duration_delay_profile.csv", index=False)
    duration_grid_by_delay(df).round(6).to_csv(METRICS_DIR / "duration_grid_by_delay.csv", index=False)
    delay_severity_distribution(df).round(6).to_csv(METRICS_DIR / "delay_severity_distribution.csv", index=False)
    favorite_item_summary(df).round(6).to_csv(METRICS_DIR / "favorite_item_summary.csv", index=False)
    restaurant_dependency_summary(df).round(6).to_csv(METRICS_DIR / "restaurant_dependency_summary.csv", index=False)
    location_dependency_summary(df).round(6).to_csv(METRICS_DIR / "location_dependency_summary.csv", index=False)
    state_dependency_summary(df).round(6).to_csv(METRICS_DIR / "state_dependency_summary.csv", index=False)
    restaurant_mix_matrix(df, "pizza_type").round(6).to_csv(
        METRICS_DIR / "restaurant_pizza_type_mix.csv", index=False
    )
    restaurant_mix_matrix(df, "pizza_size").round(6).to_csv(
        METRICS_DIR / "restaurant_pizza_size_mix.csv", index=False
    )
    state_mix_matrix(df, "pizza_type").round(6).to_csv(METRICS_DIR / "state_pizza_type_mix.csv", index=False)
    state_mix_matrix(df, "pizza_size").round(6).to_csv(METRICS_DIR / "state_pizza_size_mix.csv", index=False)

    rules = association_rules_to_delay(df)
    rules.round(4).to_csv(METRICS_DIR / "association_rules.csv", index=False)

    silhouette = kmeans_silhouette_sweep(df)
    silhouette.round(4).to_csv(METRICS_DIR / "kmeans_silhouette.csv", index=False)
    cluster_profile = kmeans_cluster_profile(df, k=4)
    cluster_profile.round(6).to_csv(METRICS_DIR / "kmeans_cluster_profile.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    df[TARGET_COLUMN].value_counts().rename({False: "On time", True: "Delayed"}).plot.bar(ax=ax)
    ax.set_title("Delivery delay distribution")
    ax.set_ylabel("Orders")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "delay_distribution.png", dpi=160)
    plt.close(fig)

    traffic = delay_rate_by(df, "traffic_level")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(traffic["traffic_level"], traffic["delay_rate"])
    ax.set_title("Delay rate by traffic level")
    ax.set_ylabel("Delay rate")
    ax.set_ylim(0, max(traffic["delay_rate"].max() * 1.2, 0.1))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "delay_rate_by_traffic.png", dpi=160)
    plt.close(fig)

    duration_grid = duration_grid_by_delay(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(duration_grid["delivery_duration_min"], duration_grid["on_time"], width=3.8, label="On-time")
    ax.bar(
        duration_grid["delivery_duration_min"],
        duration_grid["delayed"],
        width=3.8,
        bottom=duration_grid["on_time"],
        label="Delayed",
    )
    ax.axvspan(30, 35, color="red", alpha=0.10, label="inferred boundary")
    ax.set_title("Duration grid by delay label")
    ax.set_xlabel("Delivery duration (min)")
    ax.set_ylabel("Orders")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "duration_grid_by_delay.png", dpi=160)
    plt.close(fig)

    severity = delay_severity_distribution(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(severity["delay_severity"], severity["orders"])
    ax.set_title("Delay severity buckets")
    ax.set_xlabel("Severity bucket")
    ax.set_ylabel("Orders")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "delay_severity_distribution.png", dpi=160)
    plt.close(fig)

    distance = delay_rate_by(df, "distance_band")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(distance["distance_band"].astype(str), distance["delay_rate"])
    ax.set_title("Delay rate by distance band")
    ax.set_xlabel("Distance band")
    ax.set_ylabel("Delay rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "delay_rate_by_distance_band.png", dpi=160)
    plt.close(fig)

    complexity = delay_rate_by(df, "complexity_band")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(complexity["complexity_band"].astype(str), complexity["delay_rate"])
    ax.set_title("Delay rate by pizza complexity")
    ax.set_xlabel("Complexity band")
    ax.set_ylabel("Delay rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "delay_rate_by_complexity_band.png", dpi=160)
    plt.close(fig)

    type_counts = df["pizza_type"].value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(type_counts.index, type_counts.values)
    ax.set_title("Favorite pizza types by order count")
    ax.set_xlabel("Orders")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_pizza_types.png", dpi=160)
    plt.close(fig)

    restaurant = restaurant_dependency_summary(df).sort_values("delay_rate")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(restaurant["restaurant_name"], restaurant["delay_rate"])
    ax.set_title("Delay rate by restaurant brand")
    ax.set_xlabel("Delay rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "restaurant_delay_rate.png", dpi=160)
    plt.close(fig)

    state = state_dependency_summary(df).head(15).sort_values("delay_rate")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(state["state_code"], state["delay_rate"])
    ax.set_title("Delay rate by derived state code")
    ax.set_xlabel("Delay rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "state_delay_rate_top15.png", dpi=160)
    plt.close(fig)

    for matrix, filename, title in [
        (
            restaurant_mix_matrix(df, "pizza_type").set_index("restaurant_name").drop(columns=["orders"]),
            "restaurant_pizza_type_mix_heatmap.png",
            "Pizza type mix by restaurant",
        ),
        (
            restaurant_mix_matrix(df, "pizza_size").set_index("restaurant_name").drop(columns=["orders"]),
            "restaurant_pizza_size_mix_heatmap.png",
            "Pizza size mix by restaurant",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        fig.colorbar(image, ax=ax, label="Within-restaurant share")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename, dpi=160)
        plt.close(fig)

    combo = pd.crosstab(df["pizza_type"], df["pizza_size"])
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(combo.to_numpy(dtype=float), aspect="auto", cmap="Greens")
    ax.set_title("Pizza type x size order counts")
    ax.set_xticks(range(len(combo.columns)))
    ax.set_xticklabels(combo.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(combo.index)))
    ax.set_yticklabels(combo.index)
    for i in range(combo.shape[0]):
        for j in range(combo.shape[1]):
            ax.text(j, i, int(combo.iloc[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Orders")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pizza_type_size_heatmap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(silhouette["k"], silhouette["silhouette"], marker="o")
    ax.set_title("K-Means silhouette sweep")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kmeans_silhouette.png", dpi=160)
    plt.close(fig)

    # Correlation heatmap of pre-dispatch numeric features plus the target. This
    # makes the deterministic/redundant relationships visible at a glance.
    corr_cols = [
        "distance_km",
        "toppings_count",
        "pizza_size_score",
        "pizza_complexity",
        "topping_density",
        "estimated_duration_min",
        "order_hour",
    ]
    corr = df[corr_cols + [TARGET_COLUMN]].assign(**{TARGET_COLUMN: df[TARGET_COLUMN].astype(int)}).corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Correlation of pre-dispatch features and delay")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    # Top association rules by lift toward delayed=True.
    top_rules = rules.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_rules["antecedent"], top_rules["lift"], color="#8e44ad")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="lift = 1 (no effect)")
    ax.set_title("Top association rules toward delayed=True (by lift)")
    ax.set_xlabel("Lift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "association_lift_top.png", dpi=160)
    plt.close(fig)
    return {
        "rules": len(rules),
        "silhouette_rows": len(silhouette),
        "detailed_tables": [
            "duration_delay_profile.csv",
            "duration_grid_by_delay.csv",
            "delay_severity_distribution.csv",
            "favorite_item_summary.csv",
            "restaurant_dependency_summary.csv",
            "location_dependency_summary.csv",
            "state_dependency_summary.csv",
        ],
    }


if __name__ == "__main__":
    print(build_eda_artifacts())
