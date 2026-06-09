# Multimodal Disaster Response DSS

Decision support system for filtering, classifying, prioritizing, and routing
disaster-related social-media posts using CrisisMMD v2.0 text and images.

## Current pipeline

```text
CrisisMMD annotations/images
  -> text cleaning + TF-IDF
  -> CLIP image embeddings
  -> six-model comparison on dev
  -> tuned late fusion
  -> Risk Score / Priority / Routing / Manual Review
  -> Streamlit dashboard
```

The project joins the official informative target into the humanitarian master
split. Models are fitted on train, selected/tuned on dev, and reported once on
test. Dev/test metrics exclude rows whose exact image hash or cleaned text
already appeared in an earlier split.

`data/processed/robust_evaluation_mask.csv` additionally excludes visually
verified image near-duplicates and manually reviewed text near-copies. It is
kept separate from the canonical mask for robustness/sensitivity reporting.

## Verified results

| Task | Best final system | Test result |
|---|---|---:|
| Informative filtering | Late Fusion | Accuracy = 0.7072; F1 = 0.8079; F2 = 0.9035 |
| Informative dummy | Always informative | F2 = 0.8939; MCC = 0 |
| Humanitarian 8-class | Late Fusion | Macro-F1 = 0.4005 |
| Humanitarian dummy | Train majority | Macro-F1 = 0.0686 |
| Manual Review | Capacity-constrained conflict rule | Precision = 0.7536 |
| Robust sensitivity | Locked Late Fusion | F2 = 0.9006; Macro-F1 = 0.3908 |

F2 is not interpreted alone: the always-informative dummy is only 0.0096 below
fusion because the positive rate is 62.75%.

The robust row is a no-retuning sensitivity check after 137 additional,
pairwise-verified near-duplicate test rows are excluded. Canonical test results
remain the primary reported results.

On 2,000 stratified bootstrap resamples of the robust test set, informative F2
has a 95% interval of `[0.8938, 0.9068]`; its paired gain over the
always-informative dummy is `[0.0032, 0.0161]`. The gain is positive under the
row-bootstrap assumption but remains small in practical magnitude.

## Setup

```powershell
pip install -r requirements.txt
```

The verified environment uses Python 3.14.2. `.python-version` records this
version for tools that support it.

Raw CrisisMMD files and generated model/embedding binaries are intentionally
excluded from Git. See [`docs/DATA.md`](docs/DATA.md) for the source, expected
layout, and rebuild procedure.

Expected raw-data layout:

```text
data/raw/
  datasplit/crisismmd_datasplit_all/*.tsv
  CrisisMMD_v2.0/data_image/...
```

## Rebuild

The full command exports normalized splits, audits exact and near duplicates,
refreshes CLIP metadata, trains six text/image classifiers, tunes fusion on
dev, builds dashboard cache, executes both notebooks, and runs integration:

```powershell
python -m scripts.run_all
```

Existing CLIP embeddings are reused unless their `.npy` files are removed.

## Run and verify

```powershell
python -m unittest discover -s tests -v
python -m scripts.audit_data
python -m scripts.audit_near_duplicate_images
python -m scripts.build_image_duplicate_review_artifacts
python -m scripts.review_near_duplicate_images
python -m scripts.audit_near_duplicate_texts
python -m scripts.review_near_duplicate_texts
python -m scripts.build_robust_evaluation_mask
python -m scripts.evaluate_baselines
python -m src.evaluate_fusion
python -m scripts.evaluate_robustness
python -m scripts.evaluate_stability
python -m scripts.audit_dss
python -m src.test_integration
streamlit run app/streamlit_app.py
```

Dashboard URL: `http://localhost:8501`

## Finalize the submission

Edit only `docs/team_info.json`, then run:

```powershell
python -m scripts.finalize_submission
```

Before team information is complete, the GitHub-safe file set can still be
checked with:

```powershell
python -m scripts.apply_team_info
python -m scripts.submission_preflight
```

GitHub Actions runs a lightweight repository preflight. The complete 37-test
ML/DSS suite requires the locally downloaded CrisisMMD corpus and generated
model caches, so it is verified locally rather than pretending to run without
its required inputs.

## Publish to GitHub

Create an empty GitHub repository, then connect and push:

```powershell
git remote add origin https://github.com/<account>/<repository>.git
git push -u origin main
```

After filling `docs/team_info.json`, finalize and publish the last update:

```powershell
python -m scripts.finalize_submission
git add docs/team_info.json docs/contribution_log.md `
  reports/latex/chapters/team_info.tex reports/latex/main.pdf
git commit -m "Finalize team information"
git push
```

## Main artifacts

- `notebooks/02_eda.ipynb`: executed EDA, K-Means, Apriori, image audit, CLIP t-SNE.
- `notebooks/03_modeling.ipynb`: leakage-safe model comparison and fusion evidence.
- `reports/metrics/`: generated metrics and tuning outputs.
- `reports/figures/screenshots/`: browser-verified dashboard evidence.
- `reports/latex/main.pdf`: Vietnamese report.
- `docs/PROGRESS.md`: phase-by-phase completion status.
- `docs/FINAL_HANDOFF.md`: final QA and the remaining identity-only fields.

Raw images and generated model binaries are intentionally not suitable for Git
because of their size. Reproduce them with the commands above.
