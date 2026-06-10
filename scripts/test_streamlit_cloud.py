"""Run every Streamlit entrypoint with local-only resources disabled."""
import os
from pathlib import Path

os.environ["ML_DSS_FORCE_CLOUD"] = "1"

from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[1]


def main():
    paths = [
        REPO / "app" / "streamlit_app.py",
        *sorted((REPO / "app" / "pages").glob("*.py")),
    ]
    failures = []
    for path in paths:
        app = AppTest.from_file(str(path), default_timeout=45).run()
        errors = [str(item.value) for item in app.exception]
        relative = path.relative_to(REPO)
        if errors:
            failures.append(f"{relative}: {errors}")
            print(f"[FAIL] {relative}")
        else:
            print(f"[OK] {relative}")
    if failures:
        raise RuntimeError(
            "Streamlit cloud smoke failures:\n" + "\n".join(failures)
        )
    print(f"[OK] {len(paths)}/{len(paths)} cloud entrypoints passed.")


if __name__ == "__main__":
    main()
