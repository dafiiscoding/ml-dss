import os
from functools import lru_cache

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_DIR, SAMPLE_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_STATE

# ---------------------------------------------------------------------------
# Real CrisisMMD v2.0 layout
# ---------------------------------------------------------------------------
# Annotations (small) live in:
#   data/raw/datasplit/crisismmd_datasplit_all/task_{task}_text_img_{split}.tsv
# Images (large tarball) extract to:
#   data/raw/CrisisMMD_v2.0/data_image/<event>/<date>/<image_id>.jpg
# The "image" column in the TSV is already relative to CrisisMMD_v2.0, e.g.
#   data_image/california_wildfires/10_10_2017/917791291823591425_0.jpg
REAL_SPLIT_DIR = os.path.join(RAW_DATA_DIR, "datasplit", "crisismmd_datasplit_all")
REAL_IMAGE_BASE_DIR = os.path.join(RAW_DATA_DIR, "CrisisMMD_v2.0")

# Each TSV row: event_name, tweet_id, image_id, tweet_text, image,
#               label, label_text, label_image, label_text_image
TASKS = ("informative", "humanitarian")
SPLITS = ("train", "dev", "test")


def _real_data_available():
    """True if at least the informative + humanitarian train splits are present."""
    needed = [
        os.path.join(REAL_SPLIT_DIR, f"task_{t}_text_img_{s}.tsv")
        for t in TASKS for s in SPLITS
    ]
    return all(os.path.exists(p) for p in needed)


def _load_task_split(task, split):
    path = os.path.join(REAL_SPLIT_DIR, f"task_{task}_text_img_{split}.tsv")
    return pd.read_csv(path, sep="\t")


def _derive_informative(category_series):
    """Derive modality-specific informativeness from a modality category.

    This relation is exact for ``label_text`` and ``label_image`` in
    CrisisMMD. It is *not* valid for the common multimodal ``label`` field, so
    the operational informative target is joined from the informative task.
    """
    return np.where(category_series == "not_humanitarian", "not_informative", "informative")


@lru_cache(maxsize=1)
def _informative_lookup():
    """Return all official informative annotations keyed by tweet/image pair.

    CrisisMMD publishes different split assignments for the informative and
    humanitarian tasks. The project uses the humanitarian split as one unified
    multimodal partition, then joins informative labels by identity. This
    preserves every row while keeping one train/dev/test boundary for both
    targets.
    """
    frames = []
    for source_split in SPLITS:
        part = _load_task_split("informative", source_split).copy()
        part["informative_source_split"] = source_split
        frames.append(part)
    info = pd.concat(frames, ignore_index=True)
    keys = ["tweet_id", "image_id"]
    if info.duplicated(keys).any():
        raise ValueError("Informative annotations contain duplicate tweet/image pairs.")
    return info[
        keys
        + [
            "label",
            "label_text",
            "label_image",
            "label_text_image",
            "informative_source_split",
        ]
    ].rename(
        columns={
            "label": "label",
            "label_text": "label_text_inf",
            "label_image": "label_image_inf",
            "label_text_image": "informative_agree",
        }
    )


def _build_real_split(split):
    """
    Build one tidy split using the humanitarian split as the master partition.

    The two tasks use different split assignments. Joining same-named split
    files would silently drop rows, so informative labels are joined from the
    complete annotation lookup by pair identity.
    """
    hum = _load_task_split("humanitarian", split).rename(
        columns={
            "label": "label_top",
            "label_text": "label_text_cat",
            "label_image": "label_image_cat",
            "label_text_image": "multimodal_agree",
        }
    )
    merged = hum.merge(
        _informative_lookup(),
        on=["tweet_id", "image_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged["label"].isna().any():
        raise ValueError(f"Missing informative labels after joining split '{split}'.")

    df = pd.DataFrame({
        "event_name": merged["event_name"],
        "tweet_id": merged["tweet_id"],
        "image_id": merged["image_id"],
        "tweet_text": merged["tweet_text"],
        "image": merged["image"],
        # Official multimodal target plus modality-specific diagnostic labels.
        "label_top": merged["label_top"],
        "label_text_cat": merged["label_text_cat"],
        "label_image_cat": merged["label_image_cat"],
        # Official informative task labels, not inferred from humanitarian.
        "label": merged["label"],
        "label_text_inf": merged["label_text_inf"],
        "label_image_inf": merged["label_image_inf"],
        # Category and informative agreement flags shipped by CrisisMMD.
        "multimodal_agree": merged["multimodal_agree"],
        "informative_agree": merged["informative_agree"],
        "informative_source_split": merged["informative_source_split"],
    })
    df["split"] = split
    return df


def load_real_dataset(sample_frac=None, events=None, random_state=RANDOM_STATE):
    """
    Load the real CrisisMMD v2.0 dataset (informative + humanitarian merged).

    Parameters
    ----------
    sample_frac : float or None
        If given (0-1), stratified-ish downsample of each split to speed up
        heavy image-embedding steps. None = use the full dataset.
    events : list[str] or None
        Optional filter to a subset of disaster events.
    """
    train_df = _build_real_split("train")
    val_df = _build_real_split("dev")
    test_df = _build_real_split("test")

    if events:
        train_df = train_df[train_df["event_name"].isin(events)].copy()
        val_df = val_df[val_df["event_name"].isin(events)].copy()
        test_df = test_df[test_df["event_name"].isin(events)].copy()

    if sample_frac and 0 < sample_frac < 1.0:
        def _stratified_sample(df):
            return (
                df.groupby("label_top", group_keys=False)
                  .apply(lambda g: g.sample(frac=sample_frac, random_state=random_state))
                  .reset_index(drop=True)
            )
        train_df = _stratified_sample(train_df)
        val_df = _stratified_sample(val_df)
        test_df = _stratified_sample(test_df)

    return train_df, val_df, test_df, REAL_IMAGE_BASE_DIR


# ---------------------------------------------------------------------------
# Synthetic fallback (kept so the repo still runs end-to-end without the
# real ~1.8GB dataset, e.g. for a quick smoke test on another machine).
# ---------------------------------------------------------------------------
def create_synthetic_data(num_samples=200):
    """Generate small synthetic CrisisMMD-style data when the real set is absent."""
    print("Generating synthetic CrisisMMD-style dataset (fallback only)...")

    events = ["hurricane_harvey", "mexico_earthquake", "california_wildfires", "nepal_earthquake"]
    categories = [
        "injured_or_dead_people",
        "rescue_volunteering_or_donation_effort",
        "infrastructure_and_utility_damage",
        "affected_individuals",
        "other_relevant_information",
        "not_humanitarian",
    ]

    tweet_texts = {
        "injured_or_dead_people": [
            "Emergency! 3 people are seriously injured under the collapsed building on Main St. Help needed!",
            "Multiple casualties reported after the earthquake hit. Rescue teams are on their way.",
        ],
        "rescue_volunteering_or_donation_effort": [
            "Volunteers needed to help distribute food and clean water at the city center shelter.",
            "Please donate blankets and canned food at the school gym. Every little bit helps!",
        ],
        "infrastructure_and_utility_damage": [
            "The main bridge over the river has collapsed due to high water levels. Avoid the area!",
            "Power lines are down across the entire neighborhood. No electricity for 24 hours.",
        ],
        "affected_individuals": [
            "Our family has lost everything in the fire. We are looking for shelter tonight.",
            "So many families are displaced and sleeping in their cars. The situation is desperate.",
        ],
        "other_relevant_information": [
            "Weather update: Hurricane Harvey has made landfall. Stay indoors and stay safe.",
            "Updates on the wildfire containment: firefighters have contained 30% of the blaze.",
        ],
        "not_humanitarian": [
            "Look at this beautiful sunset before the storm hit. Nature is so unpredictable.",
            "Sharing a meme to keep our spirits up during this difficult time. Stay strong!",
        ],
    }

    color_map = {
        "injured_or_dead_people": (255, 0, 0),
        "rescue_volunteering_or_donation_effort": (0, 255, 0),
        "infrastructure_and_utility_damage": (0, 0, 255),
        "affected_individuals": (255, 255, 0),
        "other_relevant_information": (128, 0, 128),
        "not_humanitarian": (128, 128, 128),
    }

    img_dir = os.path.join(SAMPLE_DATA_DIR, "data_image")
    os.makedirs(img_dir, exist_ok=True)

    data = []
    rng = np.random.default_rng(RANDOM_STATE)
    for i in range(num_samples):
        cat = rng.choice(categories)
        event = rng.choice(events)
        tweet_text = rng.choice(tweet_texts[cat])
        tweet_text = f"{tweet_text} #disaster #{event} [ID: {100000 + i}]"
        label = "informative" if cat != "not_humanitarian" else "not_informative"

        img = Image.new("RGB", (224, 224), color=color_map[cat])
        img_path_rel = os.path.join("data_image", f"sample_img_{i}.png")
        img.save(os.path.join(SAMPLE_DATA_DIR, img_path_rel))

        data.append({
            "event_name": event,
            "tweet_id": 1234567890 + i,
            "image_id": f"{1234567890 + i}_0",
            "image": img_path_rel,
            "tweet_text": tweet_text,
            "label": label,
            "label_top": cat,
            "label_text_inf": label,
            "label_text_cat": cat,
            "label_image_inf": label,
            "label_image_cat": cat,
            "multimodal_agree": "Positive",
        })

    df = pd.DataFrame(data)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE, stratify=df["label_top"])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_STATE, stratify=train_df["label_top"])
    train_df.to_csv(os.path.join(SAMPLE_DATA_DIR, "task_train.tsv"), sep="\t", index=False)
    val_df.to_csv(os.path.join(SAMPLE_DATA_DIR, "task_val.tsv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(SAMPLE_DATA_DIR, "task_test.tsv"), sep="\t", index=False)
    print(f"Synthetic data saved to {SAMPLE_DATA_DIR}")


def _load_synthetic():
    paths = {s: os.path.join(SAMPLE_DATA_DIR, f"task_{s}.tsv") for s in ("train", "val", "test")}
    if not all(os.path.exists(p) for p in paths.values()):
        create_synthetic_data()
    train_df = pd.read_csv(paths["train"], sep="\t")
    val_df = pd.read_csv(paths["val"], sep="\t")
    test_df = pd.read_csv(paths["test"], sep="\t")
    return train_df, val_df, test_df, SAMPLE_DATA_DIR


def load_dataset(use_sample_if_missing=True, sample_frac=None, events=None):
    """
    Load train/val/test datasets.

    Prefers the real CrisisMMD v2.0 dataset; falls back to the small synthetic
    set only if the real annotations are missing (and the fallback is allowed).
    Returns (train_df, val_df, test_df, image_base_dir).
    """
    if _real_data_available():
        print("Real CrisisMMD v2.0 annotations found -> loading real dataset.")
        return load_real_dataset(sample_frac=sample_frac, events=events)

    if use_sample_if_missing:
        print("Real CrisisMMD not found -> falling back to synthetic sample.")
        return _load_synthetic()

    raise FileNotFoundError(
        "CrisisMMD annotations not found in data/raw/datasplit and synthetic "
        "fallback is disabled."
    )


def export_processed_splits():
    """Persist the normalized, fixed splits used by every downstream stage."""
    train_df, val_df, test_df, _ = load_dataset(use_sample_if_missing=False)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        df.to_csv(os.path.join(PROCESSED_DATA_DIR, f"{name}.csv"), index=False)
    return {"train": len(train_df), "val": len(val_df), "test": len(test_df)}


if __name__ == "__main__":
    train, val, test, base_dir = load_dataset()
    print(f"Loaded: Train={train.shape}, Val={val.shape}, Test={test.shape}")
    print(f"Image base dir: {base_dir}")
    print("\nHumanitarian category distribution (train):")
    print(train["label_top"].value_counts())
    print("\nInformative distribution (train):")
    print(train["label"].value_counts())
    print("\nMultimodal agreement (train):")
    print(train["multimodal_agree"].value_counts())
