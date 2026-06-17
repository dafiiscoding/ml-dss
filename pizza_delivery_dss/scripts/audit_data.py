import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pizza_dss.data_loader import export_processed_splits, write_data_audit


def main():
    summary = write_data_audit()
    splits = export_processed_splits()
    print(summary)
    print(splits)


if __name__ == "__main__":
    main()
