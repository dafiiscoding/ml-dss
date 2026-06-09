"""
Extract & cache CLIP image embeddings for all splits (run once, after the
CrisisMMD image tarball is extracted to data/raw/CrisisMMD_v2.0/).

Saves: models/X_{train,val,test}_img_emb.npy
These caches are consumed by train_image_model.py and the dashboard so the slow
CLIP forward pass only happens once.

Usage:
    python -m scripts.cache_image_embeddings            # full dataset
    python -m scripts.cache_image_embeddings --frac 0.2 # 20% stratified subset
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR
from src.data_loader import load_dataset, REAL_IMAGE_BASE_DIR
from src.image_preprocessing import extract_image_embeddings


def main(frac=None):
    img_dir = os.path.join(REAL_IMAGE_BASE_DIR, "data_image")
    if not os.path.isdir(img_dir):
        print(f"[ABORT] Images not found at {img_dir}. "
              "Extract CrisisMMD_v2.0.tar.gz into data/raw/ first.")
        sys.exit(1)

    train_df, val_df, test_df, base = load_dataset(
        use_sample_if_missing=False, sample_frac=frac
    )
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        _extract_split_resumable(name, df, base, chunk=1000)
    print("Done. Image embeddings cached.")


def _extract_split_resumable(name, df, base, chunk=1000):
    """Extract a split in resumable chunks: each chunk is saved to a temp dir so
    an interruption (e.g. machine sleep/kill) only loses the in-flight chunk.
    Re-running skips chunks already on disk and stitches the final array."""
    out = os.path.join(MODELS_DIR, f"X_{name}_img_emb.npy")
    meta_out = os.path.join(MODELS_DIR, f"img_{name}_meta.csv")
    meta_cols = [
        "tweet_id", "image_id", "label", "label_top", "label_text_inf",
        "label_text_cat", "label_image_inf", "label_image_cat",
        "multimodal_agree",
    ]
    df.reset_index(drop=True)[meta_cols].to_csv(meta_out, index=False)
    if os.path.exists(out):
        print(f"[skip] {name}: embeddings exist; metadata refreshed.")
        return

    tmp = os.path.join(MODELS_DIR, "_emb_tmp", name)
    os.makedirs(tmp, exist_ok=True)
    paths = df["image"].tolist()
    n = len(paths)
    print(f"Extracting CLIP embeddings for {name} ({n} images), chunk={chunk}...")
    for start in range(0, n, chunk):
        cpath = os.path.join(tmp, f"chunk_{start:06d}.npy")
        if os.path.exists(cpath):
            print(f"  [resume] chunk {start} already done.")
            continue
        emb = extract_image_embeddings(paths[start:start+chunk], base, batch_size=32)
        np.save(cpath, emb)
        print(f"  chunk {start}-{min(start+chunk, n)} saved ({(start+chunk)*100//n}% approx)")

    # Stitch all chunks in order.
    chunks = [np.load(os.path.join(tmp, f"chunk_{s:06d}.npy"))
              for s in range(0, n, chunk)]
    full = np.vstack(chunks)
    np.save(out, full)
    # Clean temp chunks for this split.
    for s in range(0, n, chunk):
        os.remove(os.path.join(tmp, f"chunk_{s:06d}.npy"))
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    print(f"  saved {out}  shape={full.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=None,
                    help="optional stratified subsample fraction (0-1) to speed up")
    main(ap.parse_args().frac)
