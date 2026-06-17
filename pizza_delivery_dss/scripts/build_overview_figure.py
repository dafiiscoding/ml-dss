"""Sinh sơ đồ pipeline tổng (schematic tĩnh) cho người mới định hướng.

Không phụ thuộc dữ liệu nên KHÔNG nằm trong scripts.run_all; chạy riêng khi cần:
    python -m scripts.build_overview_figure
Kết quả: reports/figures/pipeline_overview.png
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from pizza_dss.config import FIGURES_DIR

STEPS = [
    ("1. Dữ liệu &\nchống leakage", "NB01", "#3498db"),
    ("2. Forensics:\ndata là đồ giả?", "NB06", "#9b59b6"),
    ("3. EDA &\ninsight", "NB02", "#16a085"),
    ("4. Mô hình\ndự báo trễ", "NB03", "#e67e22"),
    ("5. Tầng quyết định\n(Risk/Priority)", "NB04", "#c0392b"),
    ("6. Dashboard &\nPower BI", "NB04", "#2c3e50"),
]


def build():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, len(STEPS) * 2.1)
    ax.set_ylim(0, 4)
    ax.axis("off")

    centers = []
    for i, (label, nb, color) in enumerate(STEPS):
        x = i * 2.1 + 0.15
        box = FancyBboxPatch(
            (x, 1.9), 1.7, 1.2,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.5, edgecolor=color, facecolor=color + "22",
        )
        ax.add_patch(box)
        ax.text(x + 0.85, 2.65, label, ha="center", va="center", fontsize=10, weight="bold", color=color)
        ax.text(x + 0.85, 2.05, nb, ha="center", va="center", fontsize=8, color="#555555", style="italic")
        centers.append(x + 1.7)
        if i > 0:
            ax.add_patch(FancyArrowPatch(
                (centers[i - 1] + 0.0, 2.5), (x, 2.5),
                arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color="#888888",
            ))

    # Nhánh phụ: phân tích kinh doanh (NB05) tách từ EDA.
    eda_x = 2 * 2.1 + 0.15 + 0.85
    ax.add_patch(FancyBboxPatch(
        (eda_x - 0.85, 0.45), 1.7, 0.95,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        linewidth=1.3, edgecolor="#7f8c8d", facecolor="#7f8c8d22",
    ))
    ax.text(eda_x, 0.92, "Behavior, forecast,\nrecommendation (NB05)",
            ha="center", va="center", fontsize=8.5, color="#7f8c8d")
    ax.add_patch(FancyArrowPatch(
        (eda_x, 1.88), (eda_x, 1.42),
        arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#aaaaaa",
    ))

    ax.text(len(STEPS) * 1.05, 3.7,
            "Pizza Delivery DSS — luồng tổng: từ dữ liệu thô đến quyết định vận hành",
            ha="center", va="center", fontsize=12, weight="bold")
    ax.text(len(STEPS) * 1.05, 0.12,
            "Dữ liệu là synthetic (đồ giả) → không overclaim; mọi kết luận kèm baseline + caveat.",
            ha="center", va="center", fontsize=8.5, color="#c0392b", style="italic")

    fig.tight_layout()
    out = FIGURES_DIR / "pipeline_overview.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")
    return out


if __name__ == "__main__":
    build()
