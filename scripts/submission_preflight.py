"""Check whether the repository is safe and complete enough for GitHub."""
import argparse
import os
import re
import subprocess
from pathlib import Path

from scripts.apply_team_info import apply as apply_team_info

REPO = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 50 * 1024 * 1024
FORBIDDEN_PREFIXES = (
    ".ai/",
    "data/raw/",
    "reports/pdf_review",
    "reports/figures/pdf_qa/",
    "reports/figures/image_duplicate_review/",
    "slide kiến thức môn học/",
)
FORBIDDEN_SUFFIXES = (".npy", ".tar.gz")
ALLOWED_MODEL_BINARIES = {
    "models/text_vectorizer.pkl",
    "models/text_inf_clf.pkl",
    "models/text_cat_clf.pkl",
    "models/image_inf_clf.pkl",
    "models/image_cat_clf.pkl",
}
REQUIRED_FILES = (
    "INSTRUCTOR.md",
    "README.md",
    "BAO_CAO_NHOM_29.pdf",
    "requirements.txt",
    "docs/team_info.json",
    "notebooks/02_eda.ipynb",
    "notebooks/03_modeling.ipynb",
    "reports/latex/main.pdf",
    "reports/metrics/robustness_summary.json",
    "models/text_vectorizer.pkl",
    "models/text_inf_clf.pkl",
    "models/text_cat_clf.pkl",
    "models/image_inf_clf.pkl",
    "models/image_cat_clf.pkl",
    "scripts/test_streamlit_cloud.py",
    "scripts/verify_cloud_bundle.py",
    "src/data_loader.py",
    "tests/test_pipeline.py",
)
TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".tex",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|secret|password|token)\b\s*[:=]\s*"
        r"[\"'][^\"']{12,}[\"']"
    ),
)


def _git_candidate_files():
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [
        Path(line)
        for line in result.stdout.splitlines()
        if line.strip() and (REPO / line).is_file()
    ]


def check(final=False):
    errors = []
    warnings = []
    files = _git_candidate_files()
    total_bytes = 0
    for relative in files:
        normalized = relative.as_posix()
        size = (REPO / relative).stat().st_size
        total_bytes += size
        if normalized.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"Forbidden submission path: {normalized}")
        if normalized.endswith(FORBIDDEN_SUFFIXES):
            errors.append(f"Generated binary/cache must not be tracked: {normalized}")
        if normalized.endswith(".pkl") and normalized not in ALLOWED_MODEL_BINARIES:
            errors.append(f"Unexpected pickle artifact in submission: {normalized}")
        if size > MAX_FILE_BYTES:
            errors.append(
                f"File exceeds 50 MiB GitHub safety limit: {normalized}"
            )
        if relative.suffix.lower() in TEXT_SUFFIXES and size <= 2 * 1024 * 1024:
            text = (REPO / relative).read_text(
                encoding="utf-8", errors="ignore"
            )
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    errors.append(f"Possible secret in: {normalized}")
                    break

    for required in REQUIRED_FILES:
        if not (REPO / required).exists():
            errors.append(f"Required artifact is missing: {required}")

    try:
        missing_team = apply_team_info(
            check=True,
            require_complete=final,
        )
        if missing_team:
            warnings.append(
                "Team identity/contribution fields remain incomplete: "
                + ", ".join(missing_team)
            )
    except RuntimeError as exc:
        errors.append(str(exc))

    remotes = subprocess.run(
        ["git", "remote"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if not remotes:
        warnings.append(
            "No Git remote is configured yet; add the GitHub repository "
            "as `origin` before pushing."
        )

    print(
        f"GitHub candidate set: {len(files)} files, "
        f"{total_bytes / 1024 / 1024:.2f} MiB"
    )
    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[ERROR] {error}")
    if errors:
        raise SystemExit(1)
    print("[OK] Submission preflight passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final",
        action="store_true",
        help="Require all team identity/contribution fields.",
    )
    args = parser.parse_args()
    check(final=args.final)


if __name__ == "__main__":
    main()
