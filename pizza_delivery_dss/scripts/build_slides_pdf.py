import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLIDES_DIR = PROJECT_ROOT / "slides"
MAIN_TEX = SLIDES_DIR / "pizza_dss_slides.tex"


def main():
    if not MAIN_TEX.exists():
        raise FileNotFoundError(MAIN_TEX)
    for _ in range(2):
        subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "pizza_dss_slides.tex",
            ],
            cwd=SLIDES_DIR,
            check=True,
        )
    target = SLIDES_DIR / "PIZZA_DSS_SLIDE_DECK.pdf"
    target.write_bytes((SLIDES_DIR / "pizza_dss_slides.pdf").read_bytes())
    print(f"Saved slides PDF -> {target}")


if __name__ == "__main__":
    main()
