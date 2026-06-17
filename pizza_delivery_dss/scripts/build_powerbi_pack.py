import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pizza_dss.powerbi import build_powerbi_pack


if __name__ == "__main__":
    print(build_powerbi_pack())
