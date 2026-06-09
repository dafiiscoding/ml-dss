"""Audit annotations, split integrity, embeddings, and the image corpus."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.data_loader import _derive_informative, load_dataset
from src.text_preprocessing import clean_tweet_text

METRICS_DIR = Path(REPORTS_DIR) / "metrics"
INTEGRITY_PATH = Path(PROCESSED_DATA_DIR) / "split_integrity.csv"


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_digest(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _same_identity_order(left, right):
    cols = ["tweet_id", "image_id"]
    return all(
        left[col].astype(str).reset_index(drop=True).equals(
            right[col].astype(str).reset_index(drop=True)
        )
        for col in cols
    )


def _same_targets(left, right):
    required = ["label", "label_top"]
    if any(column not in right.columns for column in required):
        return False
    return all(
        left[column].astype(str).reset_index(drop=True).equals(
            right[column].astype(str).reset_index(drop=True)
        )
        for column in required
    )


def _build_integrity_manifest(splits, image_base):
    rows = []
    digest_cache = {}
    for split, df in splits.items():
        for row_index, row in df.reset_index(drop=True).iterrows():
            path = Path(image_base) / row["image"]
            rel = str(row["image"])
            digest_cache.setdefault(rel, _file_digest(path))
            cleaned = clean_tweet_text(row["tweet_text"])
            rows.append(
                {
                    "split": split,
                    "row_index": row_index,
                    "tweet_id": row["tweet_id"],
                    "image_id": row["image_id"],
                    "image": rel,
                    "image_sha256": digest_cache[rel],
                    "raw_text_sha256": _text_digest(
                        str(row["tweet_text"]).strip().lower()
                    ),
                    "clean_text_sha256": _text_digest(cleaned),
                }
            )

    manifest = pd.DataFrame(rows)
    prior_images = set()
    prior_clean_texts = set()
    eligible = []
    reasons = []
    for split in ("train", "val", "test"):
        part = manifest[manifest["split"] == split]
        for _, row in part.iterrows():
            image_seen = row["image_sha256"] in prior_images
            text_seen = row["clean_text_sha256"] in prior_clean_texts
            keep = split == "train" or not (image_seen or text_seen)
            eligible.append(keep)
            reason = []
            if image_seen:
                reason.append("image_hash_seen_in_prior_split")
            if text_seen:
                reason.append("clean_text_seen_in_prior_split")
            reasons.append(";".join(reason))
        prior_images.update(part["image_sha256"])
        prior_clean_texts.update(part["clean_text_sha256"])

    manifest["evaluation_eligible"] = eligible
    manifest["exclusion_reason"] = reasons
    return manifest, digest_cache


def _overlap_summary(manifest, left, right):
    ldf = manifest[manifest["split"] == left]
    rdf = manifest[manifest["split"] == right]
    result = {}
    for column, label in (
        ("tweet_id", "tweet_ids"),
        ("image_id", "image_ids"),
        ("image_sha256", "image_hashes"),
        ("raw_text_sha256", "raw_text_hashes"),
        ("clean_text_sha256", "clean_text_hashes"),
    ):
        shared = set(ldf[column]) & set(rdf[column])
        result[f"shared_{label}"] = len(shared)
        result[f"{left}_affected_rows_{label}"] = int(ldf[column].isin(shared).sum())
        result[f"{right}_affected_rows_{label}"] = int(rdf[column].isin(shared).sum())
    left_pairs = set(zip(ldf["tweet_id"], ldf["image_id"]))
    right_pairs = set(zip(rdf["tweet_id"], rdf["image_id"]))
    result["shared_tweet_image_pairs"] = len(left_pairs & right_pairs)
    return result


def _embedding_summary(split, df):
    emb_path = Path(MODELS_DIR) / f"X_{split}_img_emb.npy"
    meta_path = Path(MODELS_DIR) / f"img_{split}_meta.csv"
    if not emb_path.exists() or not meta_path.exists():
        return {"available": False}
    values = np.load(emb_path)
    metadata = pd.read_csv(meta_path)
    norms = np.linalg.norm(values, axis=1)
    return {
        "available": True,
        "shape": list(values.shape),
        "std": round(float(values.std()), 6),
        "mean_l2_norm": round(float(norms.mean()), 6),
        "zero_norm_rows": int((norms == 0).sum()),
        "metadata_rows": len(metadata),
        "metadata_order_match": _same_identity_order(df, metadata),
    }


def audit():
    train_df, val_df, test_df, image_base = load_dataset(
        use_sample_if_missing=False
    )
    splits = {"train": train_df, "val": val_df, "test": test_df}
    manifest, referenced_digests = _build_integrity_manifest(splits, image_base)

    summary = {
        "protocol": {
            "master_split": "humanitarian",
            "informative_target": "joined official informative label",
            "evaluation_filter": (
                "exclude dev/test rows whose exact image or cleaned text "
                "appeared in an earlier split"
            ),
            "old_derived_informative_mismatches": int(
                sum(
                    (_derive_informative(df["label_top"]) != df["label"]).sum()
                    for df in splits.values()
                )
            ),
            "modality_label_derivation_mismatches": int(
                sum(
                    (
                        _derive_informative(df["label_text_cat"])
                        != df["label_text_inf"]
                    ).sum()
                    + (
                        _derive_informative(df["label_image_cat"])
                        != df["label_image_inf"]
                    ).sum()
                    for df in splits.values()
                )
            ),
        },
        "splits": {},
        "overlap": {},
        "embeddings": {},
    }

    for name, df in splits.items():
        referenced = [Path(image_base) / rel for rel in df["image"]]
        part = manifest[manifest["split"] == name]
        processed_path = Path(PROCESSED_DATA_DIR) / f"{name}.csv"
        processed_match = False
        processed_targets_match = False
        if processed_path.exists():
            processed = pd.read_csv(processed_path)
            processed_match = _same_identity_order(df, processed)
            processed_targets_match = _same_targets(df, processed)
        summary["splits"][name] = {
            "rows": len(df),
            "unique_tweet_ids": int(df["tweet_id"].nunique()),
            "unique_image_ids": int(df["image_id"].nunique()),
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_tweet_image_pairs": int(
                df.duplicated(["tweet_id", "image_id"]).sum()
            ),
            "missing_image_files": int(sum(not path.exists() for path in referenced)),
            "humanitarian_classes": int(df["label_top"].nunique()),
            "informative_distribution": df["label"].value_counts().to_dict(),
            "evaluation_eligible_rows": int(part["evaluation_eligible"].sum()),
            "excluded_prior_split_duplicates": int(
                (~part["evaluation_eligible"]).sum()
            ),
            "processed_csv_order_match": processed_match,
            "processed_csv_targets_match": processed_targets_match,
        }
        summary["embeddings"][name] = _embedding_summary(name, df)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        summary["overlap"][f"{left}_{right}"] = _overlap_summary(
            manifest, left, right
        )

    image_root = Path(image_base) / "data_image"
    files = [path for path in image_root.rglob("*") if path.is_file()]
    valid = 0
    broken = []
    dimensions = Counter()
    modes = Counter()
    all_hashes = Counter()
    for path in files:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
            rel = str(path.relative_to(Path(image_base)))
            digest = referenced_digests.get(rel) or _file_digest(path)
            all_hashes[digest] += 1
            valid += 1
        except Exception:
            broken.append(str(path.relative_to(image_root)))

    referenced_hashes = Counter(manifest["image_sha256"])
    summary["images"] = {
        "files_on_disk": len(files),
        "valid_images": valid,
        "invalid_files": len(broken),
        "invalid_file_examples": broken[:10],
        "exact_duplicate_extra_files": int(
            sum(count - 1 for count in all_hashes.values() if count > 1)
        ),
        "referenced_unique_hashes": len(referenced_hashes),
        "referenced_duplicate_extra_rows": int(
            sum(count - 1 for count in referenced_hashes.values() if count > 1)
        ),
        "color_modes": dict(modes),
        "top_dimensions": dimensions.most_common(15),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    manifest.to_csv(INTEGRITY_PATH, index=False)
    with (METRICS_DIR / "data_quality_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    combined = pd.concat(splits.values(), ignore_index=True)
    categories = sorted(combined["label_top"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, category in zip(axes.ravel(), categories):
        axis.axis("off")
        axis.set_title(category.replace("_", " "), fontsize=9)
        for _, row in train_df[train_df["label_top"] == category].iterrows():
            path = Path(image_base) / row["image"]
            try:
                with Image.open(path) as image:
                    axis.imshow(image.convert("RGB"))
                break
            except Exception:
                continue
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(
        os.path.join(FIGURES_DIR, "image_gallery.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    audit()
