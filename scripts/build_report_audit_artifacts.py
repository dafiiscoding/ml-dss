"""Build reproducible report artefacts for the V2 revision.

Produces:
  - reports/metrics/text_cleaning_audit.json  (cleaning + OOV audit)
  - reports/figures/pca_clip.png              (PCA explained variance + 2D)

Run: python -m scripts.build_report_audit_artifacts
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

from src import config
from src.text_preprocessing import clean_tweet_text

PROC = config.PROCESSED_DATA_DIR
MODELS = config.MODELS_DIR
METRICS = os.path.join(config.REPORTS_DIR, "metrics")
FIGS = config.FIGURES_DIR
os.makedirs(METRICS, exist_ok=True)


def load_split(name):
    return pd.read_csv(os.path.join(PROC, f"{name}.csv"))


def cleaning_audit():
    splits = {"train": load_split("train"), "val": load_split("val"), "test": load_split("test")}
    audit = {}
    for name, df in splits.items():
        raw = df["tweet_text"].fillna("").astype(str)
        cleaned = raw.map(clean_tweet_text)
        empty = int((cleaned.str.strip() == "").sum())
        raw_len = raw.str.len().replace(0, np.nan)
        removed_ratio = (1 - cleaned.str.len() / raw_len).clip(lower=0)
        audit[name] = {
            "n_rows": int(len(df)),
            "empty_after_clean": empty,
            "empty_after_clean_pct": round(100 * empty / len(df), 3),
            "mean_char_removed_pct": round(float(removed_ratio.mean() * 100), 2),
            "median_char_removed_pct": round(float(removed_ratio.median() * 100), 2),
        }
    # OOV: vocab from cleaned train, measure token coverage on val/test
    train_clean = splits["train"]["tweet_text"].fillna("").astype(str).map(clean_tweet_text)
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    cv.fit(train_clean)
    vocab = set(cv.vocabulary_.keys())
    audit["train_vocab_size"] = len(vocab)
    for name in ["val", "test"]:
        clean = splits[name]["tweet_text"].fillna("").astype(str).map(clean_tweet_text)
        toks, oov = 0, 0
        for t in clean:
            for w in t.split():
                toks += 1
                if w not in vocab:
                    oov += 1
        audit[name]["oov_token_pct"] = round(100 * oov / max(toks, 1), 2)
    # before/after example
    sample = splits["train"]["tweet_text"].fillna("").astype(str)
    examples = []
    for s in sample:
        c = clean_tweet_text(s)
        if "http" in s.lower() and c.strip():
            examples.append({"before": s[:120], "after": c[:120]})
        if len(examples) >= 2:
            break
    audit["examples"] = examples
    return audit


def pca_artifacts():
    X = np.load(os.path.join(MODELS, "X_train_img_emb.npy"))
    meta = pd.read_csv(os.path.join(MODELS, "img_train_meta.csv"))
    labels = meta["label_top"].values if len(meta) == len(X) else None

    pca = PCA(n_components=50, random_state=42)
    Z = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    def comps_for(th):
        idx = np.argmax(cum >= th)
        return int(idx + 1) if cum[-1] >= th else None

    summary = {
        "pc1_pct": round(float(evr[0] * 100), 2),
        "pc2_pct": round(float(evr[1] * 100), 2),
        "pc1_2_pct": round(float(cum[1] * 100), 2),
        "comps_50pct": comps_for(0.50),
        "comps_80pct": comps_for(0.80),
        "comps_90pct": comps_for(0.90),
        "cum_at_50comp_pct": round(float(cum[-1] * 100), 2),
    }

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(range(1, 51), cum * 100, marker="o", ms=3, color="#003366")
    ax[0].axhline(80, ls="--", color="#C51818", lw=0.9)
    ax[0].set_xlabel("So thanh phan chinh (PCA)")
    ax[0].set_ylabel("Phuong sai tich luy (%)")
    ax[0].set_title("CLIP embedding: phuong sai tich luy theo PCA")
    ax[0].grid(alpha=0.3)

    if labels is not None:
        cats = pd.Series(labels)
        for cat in cats.unique():
            m = cats.values == cat
            ax[1].scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.4, label=str(cat)[:14])
        ax[1].legend(fontsize=5, markerscale=2, loc="best")
    else:
        ax[1].scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.4, color="#003366")
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC2")
    ax[1].set_title("Hinh chieu PCA 2D (tuyen tinh)")
    fig.tight_layout()
    out = os.path.join(FIGS, "pca_clip.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    summary["figure"] = out
    return summary


def seeded_gallery():
    """Deterministic 2-images-per-class gallery (seed 42), not hand-picked."""
    from PIL import Image
    image_base = os.path.join(config.RAW_DATA_DIR, "CrisisMMD_v2.0")
    df = load_split("train")
    cats = sorted(df["label_top"].unique())
    rng = np.random.RandomState(42)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    panel = 0
    for cat in cats:
        rows = df[df["label_top"] == cat].sample(frac=1.0, random_state=rng).reset_index(drop=True)
        shown = 0
        for _, row in rows.iterrows():
            name = os.path.basename(str(row["image"]))
            if name.startswith("._"):
                continue
            path = os.path.join(image_base, str(row["image"]))
            try:
                with Image.open(path) as im:
                    axes[panel].imshow(im.convert("RGB"))
            except Exception:
                continue
            axes[panel].axis("off")
            axes[panel].set_title(cat.replace("_", " "), fontsize=8)
            panel += 1
            shown += 1
            if shown >= 2 or panel >= len(axes):
                break
    for k in range(panel, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    out = os.path.join(FIGS, "image_gallery_seeded.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"figure": out, "panels": panel}


def case_studies():
    """Three real test cases rendered as image + DSS outputs."""
    import textwrap
    from PIL import Image
    image_base = os.path.join(config.RAW_DATA_DIR, "CrisisMMD_v2.0")
    df = pd.read_csv(os.path.join(PROC, "dashboard_test_predictions.csv"))
    a = df[(df["priority"] == "High") & (~df["manual_review"])].sort_values(
        "risk_score", ascending=False).head(1)
    b = df[df["manual_review"]].sort_values("conflict_score", ascending=False).head(1)
    c = df[df["true_category"] == "missing_or_found_people"].head(1)
    cases = [("A. Đồng thuận, hành động", a.iloc[0]),
             ("B. Xung đột cao, review", b.iloc[0]),
             ("C. Phủ định / lớp hiếm", c.iloc[0])]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8),
                             gridspec_kw={"height_ratios": [3, 2]})
    for j, (label, r) in enumerate(cases):
        path = os.path.join(image_base, str(r["image"]))
        try:
            with Image.open(path) as im:
                axes[0, j].imshow(im.convert("RGB"))
        except Exception:
            pass
        axes[0, j].axis("off")
        axes[0, j].set_title(label, fontsize=11, color="#003366", fontweight="bold")
        txt = "\n".join(textwrap.wrap(str(r["tweet_text"])[:160], 42))
        review = "CO" if r["manual_review"] else "KHONG"
        info = (f"\"{txt}\"\n\n"
                f"P(informative) = {r['fused_informative_prob']:.2f}\n"
                f"Conflict = {r['conflict_score']:.2f}\n"
                f"Category = {str(r['fused_category']).replace('_',' ')[:28]}\n"
                f"Risk = {r['risk_score']:.1f}  ->  {r['priority']}\n"
                f"Team = {r['assigned_team']}\n"
                f"Manual Review = {review}")
        axes[1, j].axis("off")
        axes[1, j].text(0.0, 1.0, info, va="top", ha="left", fontsize=8.5,
                        family="monospace")
    fig.tight_layout()
    out = os.path.join(FIGS, "case_studies.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {"figure": out, "cases": [str(r["tweet_id"]) for _, r in cases]}


if __name__ == "__main__":
    audit = cleaning_audit()
    pca = pca_artifacts()
    gallery = seeded_gallery()
    cases = case_studies()
    print("=== CASE STUDIES ===")
    print(json.dumps(cases, ensure_ascii=False, indent=2))
    print("=== GALLERY ===")
    print(json.dumps(gallery, ensure_ascii=False, indent=2))
    with open(os.path.join(METRICS, "text_cleaning_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print("=== CLEANING AUDIT ===")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("=== PCA SUMMARY ===")
    print(json.dumps(pca, ensure_ascii=False, indent=2))
