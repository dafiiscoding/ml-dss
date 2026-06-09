"""Build quantitative diagnostics and contact sheets for image review."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.data_loader import REAL_IMAGE_BASE_DIR

METRICS_DIR = Path(REPORTS_DIR) / "metrics"
CANDIDATE_PATH = METRICS_DIR / "image_near_duplicate_candidates.csv"
DIAGNOSTIC_PATH = METRICS_DIR / "image_near_duplicate_diagnostics.csv"
SUMMARY_PATH = METRICS_DIR / "image_near_duplicate_diagnostics_summary.json"
SHEET_DIR = Path(FIGURES_DIR) / "image_duplicate_review"


def _font(size):
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _image_metrics(query_path, prior_path):
    with Image.open(query_path) as query_image:
        query_size = query_image.size
        query = np.asarray(
            query_image.convert("RGB").resize((64, 64)),
            dtype=np.float32,
        ) / 255.0
    with Image.open(prior_path) as prior_image:
        prior_size = prior_image.size
        prior = np.asarray(
            prior_image.convert("RGB").resize((64, 64)),
            dtype=np.float32,
        ) / 255.0

    query_gray = query.mean(axis=2).ravel()
    prior_gray = prior.mean(axis=2).ravel()
    if query_gray.std() > 1e-6 and prior_gray.std() > 1e-6:
        pixel_correlation = float(
            np.corrcoef(query_gray, prior_gray)[0, 1]
        )
    else:
        pixel_correlation = float(
            np.allclose(query_gray, prior_gray, atol=0.02)
        )

    histograms = []
    for values in (query, prior):
        channels = []
        for channel in range(3):
            histogram, _ = np.histogram(
                values[:, :, channel], bins=32, range=(0.0, 1.0)
            )
            channels.append(histogram.astype(np.float32))
        histogram = np.concatenate(channels)
        histogram /= max(float(np.linalg.norm(histogram)), 1e-12)
        histograms.append(histogram)

    return {
        "pixel_corr_64": pixel_correlation,
        "pixel_mae_64": float(np.abs(query - prior).mean()),
        "histogram_cosine": float(np.dot(*histograms)),
        "query_width": query_size[0],
        "query_height": query_size[1],
        "prior_width": prior_size[0],
        "prior_height": prior_size[1],
        "same_dimensions": query_size == prior_size,
        "aspect_ratio_delta": abs(
            query_size[0] / query_size[1]
            - prior_size[0] / prior_size[1]
        ),
    }


def _evidence_band(row):
    if (
        row["clip_cosine_image"] >= 0.99
        and row["pixel_corr_64"] >= 0.95
        and row["pixel_mae_64"] <= 0.05
    ):
        return "strong"
    if (
        row["clip_cosine_image"] >= 0.95
        and row["pixel_corr_64"] >= 0.85
        and row["pixel_mae_64"] <= 0.12
    ):
        return "moderate"
    return "weak"


def _thumbnail(path, size):
    with Image.open(path) as image:
        return ImageOps.contain(
            image.convert("RGB"), size, Image.Resampling.LANCZOS
        )


def _draw_sheet(rows, output_path):
    columns = 2
    cell_width = 900
    cell_height = 330
    image_size = (390, 240)
    sheet_rows = math.ceil(len(rows) / columns)
    canvas = Image.new(
        "RGB",
        (columns * cell_width, sheet_rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(20)
    detail_font = _font(16)

    for position, row in enumerate(rows.itertuples(index=False)):
        left = (position % columns) * cell_width
        top = (position // columns) * cell_height
        query = _thumbnail(
            Path(REAL_IMAGE_BASE_DIR) / row.query_image, image_size
        )
        prior = _thumbnail(
            Path(REAL_IMAGE_BASE_DIR) / row.prior_image, image_size
        )
        query_x = left + 20 + (image_size[0] - query.width) // 2
        prior_x = left + 470 + (image_size[0] - prior.width) // 2
        image_y = top + 55
        canvas.paste(query, (query_x, image_y))
        canvas.paste(prior, (prior_x, image_y))
        draw.rectangle(
            (left, top, left + cell_width - 1, top + cell_height - 1),
            outline="#9ca3af",
            width=2,
        )
        draw.text(
            (left + 20, top + 10),
            (
                f"{row.query_split}:{row.query_row_index} -> "
                f"{row.prior_split}:{row.prior_row_index} | "
                f"pHash={row.phash_distance} CLIP={row.clip_cosine_image:.3f} "
                f"corr={row.pixel_corr_64:.3f} MAE={row.pixel_mae_64:.3f}"
            ),
            fill="black",
            font=title_font,
        )
        draw.text(
            (left + 20, top + 300),
            (
                f"query: {row.query_event} / {row.query_label_top} | "
                f"prior: {row.prior_event} / {row.prior_label_top}"
            ),
            fill="#374151",
            font=detail_font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def build(rows_per_sheet=20):
    candidates = pd.read_csv(CANDIDATE_PATH)
    embeddings = {
        split: np.load(Path(MODELS_DIR) / f"X_{split}_img_emb.npy")
        for split in ("train", "val", "test")
    }
    diagnostics = []
    for row in candidates.itertuples(index=False):
        metrics = _image_metrics(
            Path(REAL_IMAGE_BASE_DIR) / row.query_image,
            Path(REAL_IMAGE_BASE_DIR) / row.prior_image,
        )
        metrics["clip_cosine_image"] = float(
            np.dot(
                embeddings[row.query_split][int(row.query_row_index)],
                embeddings[row.prior_split][int(row.prior_row_index)],
            )
        )
        diagnostics.append(metrics)

    reviewed = pd.concat(
        [candidates.reset_index(drop=True), pd.DataFrame(diagnostics)],
        axis=1,
    )
    reviewed["evidence_band"] = reviewed.apply(_evidence_band, axis=1)
    reviewed["review_status"] = "candidate_not_verified"
    reviewed["review_sheet"] = ""
    reviewed["review_sheet_position"] = 0

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    for old_sheet in SHEET_DIR.glob("*.jpg"):
        old_sheet.unlink()
    sheet_paths = []
    for band in ("strong", "moderate", "weak"):
        part = reviewed[reviewed["evidence_band"] == band].sort_values(
            ["clip_cosine_image", "pixel_corr_64"],
            ascending=False,
        )
        for start in range(0, len(part), rows_per_sheet):
            number = start // rows_per_sheet + 1
            output_path = SHEET_DIR / f"{band}_{number:02d}.jpg"
            chunk = part.iloc[start:start + rows_per_sheet]
            _draw_sheet(chunk, output_path)
            reviewed.loc[chunk.index, "review_sheet"] = output_path.name
            reviewed.loc[
                chunk.index, "review_sheet_position"
            ] = range(1, len(chunk) + 1)
            sheet_paths.append(str(output_path))
    reviewed.to_csv(DIAGNOSTIC_PATH, index=False)

    band_counts = Counter(reviewed["evidence_band"])
    summary = {
        "method": {
            "metrics": [
                "CLIP image cosine",
                "64x64 grayscale pixel correlation",
                "64x64 RGB mean absolute error",
                "96-bin RGB histogram cosine",
                "dimension and aspect-ratio comparison",
            ],
            "purpose": (
                "triage for pairwise visual review; evidence bands are not "
                "automatic duplicate decisions"
            ),
        },
        "candidate_rows": len(reviewed),
        "evidence_band_counts": {
            key: int(value) for key, value in sorted(band_counts.items())
        },
        "contact_sheets": sheet_paths,
        "canonical_evaluation_mask_changed": False,
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return reviewed, summary


if __name__ == "__main__":
    build()
