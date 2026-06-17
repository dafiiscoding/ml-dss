import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATEX_DIR = PROJECT_ROOT / "reports" / "latex"
MAIN_TEX = LATEX_DIR / "main.tex"


def main():
    if not MAIN_TEX.exists():
        raise FileNotFoundError(MAIN_TEX)
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
    pdf = LATEX_DIR / "main.pdf"
    target = PROJECT_ROOT / "reports" / "PIZZA_DSS_REPORT.pdf"
    target.write_bytes(pdf.read_bytes())
    print(f"Saved report PDF -> {target}")


if __name__ == "__main__":
    main()
