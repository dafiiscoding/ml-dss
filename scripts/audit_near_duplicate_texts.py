"""Audit semantic and lexical cross-split text near-duplicates."""
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.data_loader import load_dataset
from src.image_preprocessing import extract_text_embeddings
from src.text_preprocessing import clean_tweet_text
from src.text_similarity import (
    build_lexical_similarity_space,
    sparse_row_cosine,
    token_jaccard,
)

INTEGRITY_PATH = Path(PROCESSED_DATA_DIR) / "split_integrity.csv"
TEXT_CLIP_META_PATH = Path(PROCESSED_DATA_DIR) / "text_clip_meta.csv"
METRICS_DIR = Path(REPORTS_DIR) / "metrics"
CANDIDATE_PATH = METRICS_DIR / "text_near_duplicate_candidates.csv"
SUMMARY_PATH = METRICS_DIR / "text_near_duplicate_summary.json"
SEMANTIC_THRESHOLD = 0.98
LEXICAL_THRESHOLD = 0.80
NEIGHBORS_PER_SIGNAL = 5
SPLIT_ORDER = ("train", "val", "test")


def _embedding_path(split):
    return Path(MODELS_DIR) / f"X_{split}_text_clip_emb.npy"


def _build_work_frame(splits, manifest):
    rows = []
    for split, frame in splits.items():
        selected = frame.reset_index(drop=True)[
            ["tweet_text", "event_name", "label", "label_top"]
        ].copy()
        selected["split"] = split
        selected["row_index"] = range(len(selected))
        selected["cleaned_text"] = selected["tweet_text"].map(
            clean_tweet_text
        )
        rows.append(selected)
    work = pd.concat(rows, ignore_index=True)
    integrity = manifest[
        [
            "split",
            "row_index",
            "clean_text_sha256",
            "evaluation_eligible",
        ]
    ]
    return work.merge(
        integrity,
        on=["split", "row_index"],
        how="left",
        validate="one_to_one",
    )


def _cache_matches(meta, work):
    required = ["split", "row_index", "clean_text_sha256"]
    if any(column not in meta.columns for column in required):
        return False
    expected = work[required].reset_index(drop=True).astype(str)
    actual = meta[required].reset_index(drop=True).astype(str)
    if not actual.equals(expected):
        return False
    return all(
        _embedding_path(split).exists()
        and len(np.load(_embedding_path(split), mmap_mode="r"))
        == int((work["split"] == split).sum())
        for split in SPLIT_ORDER
    )


def _load_or_build_embeddings(work):
    if TEXT_CLIP_META_PATH.exists():
        meta = pd.read_csv(TEXT_CLIP_META_PATH)
        if _cache_matches(meta, work):
            print(f"Reusing CLIP text cache: {TEXT_CLIP_META_PATH}")
            return np.vstack(
                [np.load(_embedding_path(split)) for split in SPLIT_ORDER]
            )

    arrays = []
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    for split in SPLIT_ORDER:
        texts = work.loc[
            work["split"] == split, "cleaned_text"
        ].tolist()
        values = extract_text_embeddings(texts)
        np.save(_embedding_path(split), values)
        arrays.append(values)
    TEXT_CLIP_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    work[
        ["split", "row_index", "clean_text_sha256"]
    ].to_csv(TEXT_CLIP_META_PATH, index=False)
    return np.vstack(arrays)


def _nearest(model_matrix, query_indices, prior_end):
    neighbors = min(NEIGHBORS_PER_SIGNAL, prior_end)
    model = NearestNeighbors(
        n_neighbors=neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit(model_matrix[:prior_end])
    distances, indices = model.kneighbors(model_matrix[query_indices])
    return 1.0 - distances, indices


def _quantiles(values):
    levels = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)
    return {
        str(level): round(float(value), 6)
        for level, value in zip(levels, np.quantile(values, levels))
    }


def audit(
    semantic_threshold=SEMANTIC_THRESHOLD,
    lexical_threshold=LEXICAL_THRESHOLD,
):
    train_df, val_df, test_df, _ = load_dataset(
        use_sample_if_missing=False
    )
    if not INTEGRITY_PATH.exists():
        raise FileNotFoundError(
            "Integrity manifest missing. Run `python -m scripts.audit_data`."
        )
    manifest = pd.read_csv(INTEGRITY_PATH)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    work = _build_work_frame(splits, manifest)
    clip_embeddings = _load_or_build_embeddings(work)
    lexical = build_lexical_similarity_space(
        work["cleaned_text"].tolist()
    )

    split_sizes = {name: len(frame) for name, frame in splits.items()}
    starts = {
        "train": 0,
        "val": split_sizes["train"],
        "test": split_sizes["train"] + split_sizes["val"],
    }
    candidates = []
    nearest_distributions = {}

    for split in ("val", "test"):
        start = starts[split]
        prior_end = start
        eligible_rows = work.loc[
            (work["split"] == split) & work["evaluation_eligible"],
            "row_index",
        ].astype(int).to_numpy()
        query_indices = start + eligible_rows
        clip_scores, clip_indices = _nearest(
            clip_embeddings, query_indices, prior_end
        )
        lexical_scores, lexical_indices = _nearest(
            lexical.combined, query_indices, prior_end
        )
        nearest_distributions[split] = {
            "eligible_rows": len(query_indices),
            "clip_nearest_quantiles": _quantiles(clip_scores[:, 0]),
            "lexical_nearest_quantiles": _quantiles(
                lexical_scores[:, 0]
            ),
        }

        for position, query_index in enumerate(query_indices):
            query = work.iloc[query_index]
            prior_indices = set(clip_indices[position].tolist())
            prior_indices.update(lexical_indices[position].tolist())
            pair_rows = []
            for prior_index in prior_indices:
                prior = work.iloc[prior_index]
                if (
                    prior["clean_text_sha256"]
                    == query["clean_text_sha256"]
                ):
                    continue
                clip_similarity = float(
                    np.dot(
                        clip_embeddings[query_index],
                        clip_embeddings[prior_index],
                    )
                )
                lexical_similarity = sparse_row_cosine(
                    lexical.combined, query_index, prior_index
                )
                if (
                    clip_similarity < semantic_threshold
                    and lexical_similarity < lexical_threshold
                ):
                    continue
                pair_rows.append(
                    (
                        max(
                            clip_similarity / semantic_threshold,
                            lexical_similarity / lexical_threshold,
                        ),
                        clip_similarity,
                        lexical_similarity,
                        prior_index,
                    )
                )
            if not pair_rows:
                continue
            (
                _,
                clip_similarity,
                lexical_similarity,
                prior_index,
            ) = max(pair_rows)
            prior = work.iloc[prior_index]
            semantic_hit = clip_similarity >= semantic_threshold
            lexical_hit = lexical_similarity >= lexical_threshold
            candidates.append(
                {
                    "query_split": split,
                    "query_row_index": int(query["row_index"]),
                    "query_event": query["event_name"],
                    "query_label": query["label"],
                    "query_label_top": query["label_top"],
                    "query_text": query["tweet_text"],
                    "query_cleaned_text": query["cleaned_text"],
                    "prior_split": prior["split"],
                    "prior_row_index": int(prior["row_index"]),
                    "prior_event": prior["event_name"],
                    "prior_label": prior["label"],
                    "prior_label_top": prior["label_top"],
                    "prior_text": prior["tweet_text"],
                    "prior_cleaned_text": prior["cleaned_text"],
                    "clip_cosine": round(clip_similarity, 6),
                    "lexical_cosine": round(lexical_similarity, 6),
                    "character_cosine": round(
                        sparse_row_cosine(
                            lexical.character,
                            query_index,
                            prior_index,
                        ),
                        6,
                    ),
                    "word_cosine": round(
                        sparse_row_cosine(
                            lexical.word, query_index, prior_index
                        ),
                        6,
                    ),
                    "token_jaccard": round(
                        token_jaccard(
                            query["cleaned_text"],
                            prior["cleaned_text"],
                        ),
                        6,
                    ),
                    "sequence_ratio": round(
                        SequenceMatcher(
                            None,
                            query["cleaned_text"],
                            prior["cleaned_text"],
                        ).ratio(),
                        6,
                    ),
                    "candidate_signal": (
                        "both"
                        if semantic_hit and lexical_hit
                        else "clip_semantic"
                        if semantic_hit
                        else "tfidf_lexical"
                    ),
                    "same_event": (
                        query["event_name"] == prior["event_name"]
                    ),
                    "same_informative_label": (
                        query["label"] == prior["label"]
                    ),
                    "same_humanitarian_label": (
                        query["label_top"] == prior["label_top"]
                    ),
                    "current_evaluation_eligible": True,
                    "review_status": "candidate_not_verified",
                }
            )

    candidate_df = pd.DataFrame(candidates)
    if not candidate_df.empty:
        candidate_df = candidate_df.sort_values(
            [
                "candidate_signal",
                "clip_cosine",
                "lexical_cosine",
            ],
            ascending=[True, False, False],
        )
    signal_counts = Counter(
        candidate_df["candidate_signal"]
        if not candidate_df.empty
        else []
    )
    split_counts = Counter(
        candidate_df["query_split"] if not candidate_df.empty else []
    )
    summary = {
        "method": {
            "semantic_signal": (
                "normalized CLIP text embedding cosine"
            ),
            "lexical_signal": (
                "0.6 char 3-5 gram TF-IDF + 0.4 word 1-2 gram TF-IDF"
            ),
            "semantic_threshold": float(semantic_threshold),
            "lexical_threshold": float(lexical_threshold),
            "comparison": (
                "validation against train; test against train+validation"
            ),
            "scope": (
                "only rows still eligible under the exact SHA/text mask"
            ),
            "decision": (
                "candidate only; no evaluation row is removed automatically"
            ),
        },
        "rows_embedded": len(work),
        "nearest_similarity_distributions": nearest_distributions,
        "additional_candidate_rows": len(candidate_df),
        "candidate_rows_by_split": {
            key: int(value) for key, value in sorted(split_counts.items())
        },
        "candidate_rows_by_signal": {
            key: int(value) for key, value in sorted(signal_counts.items())
        },
        "same_event_candidates": int(
            candidate_df["same_event"].sum()
        ) if not candidate_df.empty else 0,
        "same_informative_label_candidates": int(
            candidate_df["same_informative_label"].sum()
        ) if not candidate_df.empty else 0,
        "same_humanitarian_label_candidates": int(
            candidate_df["same_humanitarian_label"].sum()
        ) if not candidate_df.empty else 0,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(CANDIDATE_PATH, index=False)
    with SUMMARY_PATH.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return candidate_df, summary


if __name__ == "__main__":
    audit()
