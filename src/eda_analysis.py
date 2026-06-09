"""
Reusable EDA computation helpers for the CrisisMMD Disaster Response DSS.

The heavy / non-trivial analysis logic (TF-IDF keyword stats, K-Means
clustering, Apriori association rules, multimodal-conflict analysis) lives here
so that both the Jupyter notebook (notebooks/02_eda.ipynb) and the Streamlit
dashboard can call the exact same, tested code instead of duplicating it.

All functions are pure-compute and return pandas objects; plotting is left to
the caller (notebook / app) so charts render inline.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.text_preprocessing import clean_tweet_text
from src.config import CRISIS_KEYWORDS

# Domain noise tokens that dominate raw tweets without adding topical meaning
# (retweet marker, html entity, url fragments, and the disaster event names
# themselves — we want clusters by *theme*, not by which hurricane it was).
DOMAIN_STOPWORDS = {
    "rt", "amp", "http", "https", "co", "via", "new", "us",
    "harvey", "irma", "maria", "mexico", "nepal", "iran", "iraq",
    "texas", "florida", "california", "srilanka", "lanka",
    "hurricane", "hurricaneharvey", "hurricaneirma", "hurricanemaria",
    "earthquake", "wildfire", "wildfires", "flood", "floods",
}
CLUSTER_STOPWORDS = list(ENGLISH_STOP_WORDS | DOMAIN_STOPWORDS)


# ---------------------------------------------------------------------------
# Basic descriptive helpers
# ---------------------------------------------------------------------------
def add_text_features(df):
    """Add char/word length features used throughout the EDA."""
    out = df.copy()
    out["text_clean"] = out["tweet_text"].apply(clean_tweet_text)
    out["char_length"] = out["tweet_text"].astype(str).str.len()
    out["word_count"] = out["text_clean"].str.split().apply(len)
    return out


def label_distribution(df, col):
    """Counts + percentage for a categorical label column."""
    vc = df[col].value_counts()
    pct = (vc / len(df) * 100).round(2)
    return pd.DataFrame({"count": vc, "percent": pct})


def imbalance_ratio(df, col):
    """Ratio of the largest class to the smallest (a quick imbalance gauge)."""
    vc = df[col].value_counts()
    return round(vc.max() / vc.min(), 1)


# ---------------------------------------------------------------------------
# Keyword / token analysis (Text Mining)
# ---------------------------------------------------------------------------
def top_tokens(texts, top_n=20, extra_stopwords=None):
    """Most frequent informative tokens across a series of (already raw) texts."""
    stop = set(ENGLISH_STOP_WORDS)
    if extra_stopwords:
        stop |= set(extra_stopwords)
    counter = Counter()
    for t in texts:
        for w in clean_tweet_text(t).split():
            if len(w) > 2 and w not in stop and not w.isdigit():
                counter[w] += 1
    return pd.DataFrame(counter.most_common(top_n), columns=["token", "frequency"])


def top_tokens_per_category(df, category_col="label_top", top_n=10):
    """Return {category: DataFrame(top tokens)} for each humanitarian category."""
    result = {}
    for cat, grp in df.groupby(category_col):
        result[cat] = top_tokens(grp["tweet_text"], top_n=top_n)
    return result


# ---------------------------------------------------------------------------
# Clustering (DSS09c) — unsupervised topic discovery on TF-IDF
# ---------------------------------------------------------------------------
def kmeans_topic_clustering(df, n_clusters=8, max_features=2000, random_state=42,
                            sample_size=None):
    """
    Run K-Means over TF-IDF of the (informative) tweets to discover natural
    topic clusters, then characterise each cluster by its top terms and compare
    against the true humanitarian labels.

    Returns a dict with: silhouette, cluster_terms (dict), crosstab (DataFrame),
    and the cluster assignment series.
    """
    work = df.copy()
    if sample_size and len(work) > sample_size:
        work = work.sample(sample_size, random_state=random_state)

    vec = TfidfVectorizer(max_features=max_features, stop_words=CLUSTER_STOPWORDS,
                          ngram_range=(1, 2), min_df=3)
    X = vec.fit_transform(work["tweet_text"].apply(clean_tweet_text))

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    sil = round(float(silhouette_score(X, labels, sample_size=min(2000, X.shape[0]),
                                       random_state=random_state)), 3)

    # Top terms per cluster from centroid weights.
    terms = np.array(vec.get_feature_names_out())
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    cluster_terms = {
        int(c): terms[order_centroids[c, :10]].tolist() for c in range(n_clusters)
    }

    work = work.assign(cluster=labels)
    crosstab = pd.crosstab(work["cluster"], work["label_top"])

    return {
        "silhouette": sil,
        "cluster_terms": cluster_terms,
        "crosstab": crosstab,
        "assignments": work[["cluster", "label_top"]],
        "n_clusters": n_clusters,
    }


def kmeans_silhouette_sweep(
    df, k_values=range(2, 13), max_features=2000, random_state=42,
    sample_size=8000
):
    """Evaluate several k values on one fixed TF-IDF sample."""
    work = df.copy()
    if sample_size and len(work) > sample_size:
        work = work.sample(sample_size, random_state=random_state)
    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words=CLUSTER_STOPWORDS,
        ngram_range=(1, 2),
        min_df=3,
    )
    X = vec.fit_transform(work["tweet_text"].apply(clean_tweet_text))
    rows = []
    for k in k_values:
        model = KMeans(
            n_clusters=k, random_state=random_state, n_init=10
        )
        labels = model.fit_predict(X)
        score = silhouette_score(
            X,
            labels,
            sample_size=min(2000, X.shape[0]),
            random_state=random_state,
        )
        rows.append({"k": int(k), "silhouette": round(float(score), 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Association analysis (DSS09a) — Apriori on crisis keyword co-occurrence
# ---------------------------------------------------------------------------
def _crisis_transactions(df, keywords=None, include_hashtags=True):
    """Build one 'basket' of crisis keywords (+ hashtags) per tweet."""
    kw = set(keywords or CRISIS_KEYWORDS)
    transactions = []
    for text in df["tweet_text"].astype(str):
        hashtags = {
            tag.lower() for tag in re.findall(r"#(\w+)", text.lower())
        }
        cleaned = clean_tweet_text(text)
        tokens = set(cleaned.split())
        # Avoid tautological rules such as #earthquake -> earthquake by not
        # inserting the plain token when the same lexical item is a hashtag.
        items = (tokens & kw) - hashtags
        if include_hashtags:
            items |= {f"#{tag}" for tag in hashtags}
        if len(items) >= 2:
            transactions.append(items)
    return transactions


def association_rules_analysis(df, min_support=0.005, min_confidence=0.25,
                               keywords=None, top_n=20):
    """
    Apriori association-rule mining over crisis-keyword co-occurrence baskets.
    Returns (frequent_itemsets, rules) as DataFrames. Requires mlxtend.
    """
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules

    transactions = _crisis_transactions(df, keywords=keywords)
    if not transactions:
        return pd.DataFrame(), pd.DataFrame()

    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    onehot = pd.DataFrame(arr, columns=te.columns_)

    itemsets = apriori(onehot, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return itemsets, pd.DataFrame()

    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        def lexical_set(items):
            return {str(item).lstrip("#") for item in items}

        rules = rules[
            rules.apply(
                lambda row: lexical_set(row["antecedents"]).isdisjoint(
                    lexical_set(row["consequents"])
                ),
                axis=1,
            )
        ]
        rules = rules.sort_values("lift", ascending=False).head(top_n)
        # Pretty string columns for display.
        rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
        rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
        rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)
    return itemsets.sort_values("support", ascending=False).head(top_n), rules


# ---------------------------------------------------------------------------
# Multimodal conflict analysis (motivates the DSS Manual-Review rule)
# ---------------------------------------------------------------------------
def multimodal_conflict_analysis(df):
    """
    Quantify how often the text and image labels disagree, overall and per
    humanitarian category. Uses the real `multimodal_agree` ground-truth column
    (Positive = agree, Negative = disagree) shipped with CrisisMMD v2.0.
    """
    if "multimodal_agree" not in df.columns:
        return None

    overall = df["multimodal_agree"].value_counts(normalize=True).mul(100).round(1)

    by_cat = (
        df.assign(disagree=(df["multimodal_agree"] == "Negative").astype(int))
          .groupby("label_top")["disagree"]
          .agg(["mean", "count"])
          .assign(disagree_pct=lambda d: (d["mean"] * 100).round(1))
          .sort_values("disagree_pct", ascending=False)[["disagree_pct", "count"]]
    )
    return {"overall_pct": overall, "by_category": by_cat}


if __name__ == "__main__":
    from src.data_loader import load_dataset

    train_df, val_df, test_df, _ = load_dataset()
    df = add_text_features(pd.concat([train_df, val_df, test_df], ignore_index=True))
    print(f"Total samples: {len(df)}")

    print("\n[1] Informative distribution:")
    print(label_distribution(df, "label"))
    print(f"\nHumanitarian imbalance ratio (max/min): {imbalance_ratio(df, 'label_top')}")

    print("\n[2] K-Means topic clustering (k=8):")
    clus = kmeans_topic_clustering(df, n_clusters=8, sample_size=5000)
    print(f"  Silhouette: {clus['silhouette']}")
    for c, terms in list(clus["cluster_terms"].items())[:3]:
        print(f"  Cluster {c}: {terms}")

    print("\n[3] Apriori association rules (top 5 by lift):")
    _, rules = association_rules_analysis(df)
    print(rules.head(5).to_string(index=False) if not rules.empty else "  (no rules)")

    print("\n[4] Multimodal conflict:")
    mc = multimodal_conflict_analysis(df)
    if mc:
        print(mc["overall_pct"])
        print(mc["by_category"])
