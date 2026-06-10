"""
SYNC RUNBOOK - rebuild the full audited CrisisMMD pipeline.

The run exports normalized data, audits exact/perceptual/semantic duplicates,
rebuilds EDA artifacts, trains both modalities, evaluates fusion/baselines/DSS,
rebuilds notebooks, and finishes with an integration test.

Usage:
    python -m scripts.run_all
"""
import os
import sys
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.data_loader import REAL_IMAGE_BASE_DIR


def _run(title, fn):
    print(f"\n{'='*60}\n>> {title}\n{'='*60}", flush=True)
    fn()
    print(f"[DONE] {title}", flush=True)


def _py(module):
    subprocess.run([sys.executable, "-m", module], check=True, cwd=REPO)


def _script(path):
    subprocess.run([sys.executable, path], check=True, cwd=REPO)


def main():
    if not os.path.isdir(os.path.join(REAL_IMAGE_BASE_DIR, "data_image")):
        print(f"[ABORT] Chưa thấy ảnh ở {REAL_IMAGE_BASE_DIR}/data_image. "
              "Giải nén CrisisMMD_v2.0.tar.gz vào data/raw/ trước.")
        sys.exit(1)

    from scripts.cache_image_embeddings import main as cache_main
    from src.train_text_model import train_text_models
    from src.train_image_model import train_image_models
    from src.data_loader import export_processed_splits
    from src.tune_selected_models import tune_selected_models
    from src.evaluate_fusion import evaluate
    from scripts.evaluate_baselines import evaluate as evaluate_baselines
    from scripts.evaluate_robustness import evaluate as evaluate_robustness
    from scripts.evaluate_stability import evaluate as evaluate_stability
    from src.dashboard_data import build_dashboard_data
    from src.test_integration import run_integration_test
    from scripts.audit_data import audit
    from scripts.audit_near_duplicate_images import audit as audit_near_images
    from scripts.audit_near_duplicate_texts import audit as audit_near_texts
    from scripts.build_image_duplicate_review_artifacts import (
        build as build_image_review_artifacts,
    )
    from scripts.review_near_duplicate_images import review as review_near_images
    from scripts.review_near_duplicate_texts import review as review_near_texts
    from scripts.build_robust_evaluation_mask import (
        build as build_robust_mask,
    )
    from scripts.audit_dss import build as audit_dss
    from scripts.build_eda_artifacts import build as build_eda_artifacts

    _run("1/21 Export normalized splits", export_processed_splits)
    _run("2/21 Cache CLIP image embeddings", lambda: cache_main(None))
    _run("3/21 Audit data and exact duplicates", audit)
    _run("4/21 Audit perceptual near-duplicate images", audit_near_images)
    _run("5/21 Build image review diagnostics/sheets", build_image_review_artifacts)
    _run("6/21 Review near-duplicate image candidates", review_near_images)
    _run("7/21 Audit semantic near-duplicate texts", audit_near_texts)
    _run("8/21 Review near-duplicate text candidates", review_near_texts)
    _run("9/21 Build robust evaluation mask", build_robust_mask)
    _run("10/21 Build reusable EDA artifacts", build_eda_artifacts)
    _run("11/21 Train + compare TEXT models", lambda: train_text_models())
    _run("12/21 Train + compare IMAGE models", lambda: train_image_models())
    _run("13/21 Tune selected classifier families", tune_selected_models)
    _run("14/21 Tune and evaluate fusion", evaluate)
    _run("15/21 Evaluate dummy baselines", evaluate_baselines)
    _run("16/21 Evaluate locked-config robustness", evaluate_robustness)
    _run("17/21 Bootstrap/event/class stability", evaluate_stability)
    _run("18/21 Build dashboard cache", build_dashboard_data)
    _run("19/21 Audit DSS rules and policy sensitivity", audit_dss)
    _run("20/21 Rebuild evidence notebooks", lambda: (
        _script(os.path.join("scripts", "build_eda_nb.py")),
        _script(os.path.join("scripts", "build_modeling_nb.py")),
    ))
    _run("21/21 Integration test", run_integration_test)
    print("\n[ALL DONE] Sync pipeline complete. Check notebooks/, models/, reports/metrics/.")


if __name__ == "__main__":
    main()
