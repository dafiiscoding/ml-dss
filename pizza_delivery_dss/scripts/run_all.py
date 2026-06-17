import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(title, module_args):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}", flush=True)
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-m", *module_args.split()],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def main():
    _run("1/18 Audit data and export processed splits", "scripts.audit_data")
    _run("2/18 Build EDA artifacts", "scripts.build_eda_artifacts")
    _run("3/18 Build business, forecasting, and synthetic-data artifacts", "scripts.build_business_artifacts")
    _run("4/18 Train, select on dev, evaluate on test", "scripts.train_models")
    _run("5/18 Build data forensics and generator reverse-engineering artifacts", "scripts.build_forensics_artifacts")
    _run("6/18 Build dashboard queue", "pizza_dss.dashboard_data")
    _run("7/18 Build transportation optimization scenario", "scripts.build_transport_optimization")
    _run("8/18 Build PowerBI-ready data pack", "scripts.build_powerbi_pack")
    _run("9/18 Rebuild executed data notebook", "scripts.build_data_nb")
    _run("10/18 Rebuild executed EDA notebook", "scripts.build_eda_nb")
    _run("11/18 Rebuild executed modeling notebook", "scripts.build_modeling_nb")
    _run("12/18 Rebuild executed DSS/PowerBI notebook", "scripts.build_dss_nb")
    _run("13/18 Rebuild executed business notebook", "scripts.build_business_nb")
    _run("14/18 Rebuild executed forensics notebook", "scripts.build_forensics_nb")
    _run("15/18 Build LaTeX report PDF", "scripts.build_report_pdf")
    _run("16/18 Build Beamer slide PDF", "scripts.build_slides_pdf")
    _run("17/18 Run unit tests", "unittest discover -s tests -v")
    _run("18/18 Run Streamlit smoke test", "scripts.test_streamlit_app")


if __name__ == "__main__":
    main()
