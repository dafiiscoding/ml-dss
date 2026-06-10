"""Rebuild every downstream artifact after classifier hyperparameter tuning."""

from scripts.audit_dss import build as audit_dss
from scripts.evaluate_baselines import evaluate as evaluate_baselines
from scripts.evaluate_robustness import evaluate as evaluate_robustness
from scripts.evaluate_stability import evaluate as evaluate_stability
from src.dashboard_data import build_dashboard_data
from src.evaluate_fusion import evaluate as evaluate_fusion
from src.test_integration import run_integration_test


def _run(name, function):
    print(f"\n=== {name} ===", flush=True)
    function()
    print(f"[DONE] {name}", flush=True)


def main():
    _run("Tune fusion and evaluate canonical test", evaluate_fusion)
    _run("Evaluate dummy baselines", evaluate_baselines)
    _run("Evaluate robust duplicate mask", evaluate_robustness)
    _run("Bootstrap and stability analysis", evaluate_stability)
    _run("Build dashboard data", build_dashboard_data)
    _run("Audit DSS policies", audit_dss)
    _run("Run integration test", run_integration_test)
    print("\nPost-tuning rebuild complete.", flush=True)


if __name__ == "__main__":
    main()
