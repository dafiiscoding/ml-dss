"""Apply team metadata, compile the report, and run final GitHub checks."""
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.apply_team_info import apply

REPO = Path(__file__).resolve().parents[1]
LATEX_DIR = REPO / "reports" / "latex"
REPORT_OUTPUT = REPO / "BAO_CAO_NHOM_29.pdf"


def main():
    apply(require_complete=True)
    for _ in range(2):
        subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "main.tex",
            ],
            cwd=LATEX_DIR,
            check=True,
        )
    shutil.copy2(LATEX_DIR / "main.pdf", REPORT_OUTPUT)
    subprocess.run(
        [sys.executable, "-m", "scripts.submission_preflight", "--final"],
        cwd=REPO,
        check=True,
    )
    print("Final submission files are synchronized and validated.")


if __name__ == "__main__":
    main()
