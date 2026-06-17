import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pizza_dss.business_analysis import build_business_artifacts


if __name__ == "__main__":
    print(build_business_artifacts())
