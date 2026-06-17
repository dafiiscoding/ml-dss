import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    from streamlit.testing.v1 import AppTest

    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=120)
    app.run(timeout=120)
    if app.exception:
        raise AssertionError(app.exception)
    print("Streamlit AppTest passed.")


if __name__ == "__main__":
    main()
