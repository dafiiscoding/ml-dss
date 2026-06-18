import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pizza_dss.config import FEATURE_COLUMNS, LEAKAGE_COLUMNS
from pizza_dss.business_analysis import (
    forecast_method_comparison,
    forecast_monthly_demand,
    mann_kendall_trend,
    preference_trend_forecast,
    preference_trend_tests,
    recommendation_rules,
    synthetic_data_audit,
)
from pizza_dss.data_loader import audit_dataset, load_dataset, split_dataset, validate_feature_contract
from pizza_dss.data_forensics import (
    bootstrap_brand_delta_f2,
    brand_ablation,
    deterministic_formula_audit,
    duration_generator_reconstruction,
    feature_information_audit,
    infer_delay_threshold,
    mi_permutation_audit,
    uniformity_tests,
)
from pizza_dss.decision_rules import (
    RISK_COMPONENT_WEIGHTS,
    calculate_delay_risk_score,
    explain_delay_risk_score,
    get_dss_decision,
    risk_component_policy_spec,
)
from pizza_dss.eda import (
    delay_rate_with_ci,
    duration_delay_profile,
    favorite_item_summary,
    kmeans_cluster_profile,
    location_dependency_summary,
    restaurant_dependency_summary,
    state_dependency_summary,
)
from pizza_dss.modeling import (
    bootstrap_test_metrics,
    compare_default_vs_tuned_lr,
    compare_models,
    cross_validate_models,
    evaluate_baselines,
    evaluate_threshold_policy_transfer,
    fbeta_threshold_analysis,
    load_best_model,
    model_stability_audit,
    summarize_model_stability,
    tune_selected_model,
)
from pizza_dss.transport_optimization import solve_transport_assignment, transport_cost_policy_spec


class PizzaDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()

    def test_schema_and_shape(self):
        self.assertEqual(len(self.df), 1004)
        self.assertIn("is_delayed", self.df.columns)
        self.assertIn("distance_km", self.df.columns)
        self.assertIn("pizza_size_score", self.df.columns)
        self.assertIn("complexity_band", self.df.columns)

    def test_leakage_contract(self):
        self.assertTrue(validate_feature_contract())
        self.assertFalse(set(FEATURE_COLUMNS) & set(LEAKAGE_COLUMNS))

    def test_delay_label_definition_is_audited(self):
        audit = audit_dataset(self.df)
        self.assertTrue(audit["is_delayed_equals_duration_gt_30"])
        self.assertEqual(audit["missing_total"], 0)

    def test_split_is_stratified(self):
        train, dev, test = split_dataset(self.df)
        self.assertEqual((len(train), len(dev), len(test)), (602, 201, 201))
        for frame in (train, dev, test):
            self.assertGreater(frame["is_delayed"].sum(), 0)

    def test_synthetic_audit_flags_deterministic_columns(self):
        audit = synthetic_data_audit(self.df)
        self.assertIn("Label is deterministic from delivery duration", set(audit["check"]))
        self.assertGreaterEqual(int(audit["severity"].isin(["warning", "critical"]).sum()), 5)


class PizzaModelAndDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.dev, cls.test = split_dataset(load_dataset())

    def test_baselines_include_operational_context(self):
        metrics = evaluate_baselines(self.dev["is_delayed"])
        self.assertEqual(set(metrics["model"]), {"Always on-time", "Always delayed"})

    def test_model_comparison_runs(self):
        metrics, fitted = compare_models(self.train, self.dev)
        self.assertGreaterEqual(len(metrics), 6)
        self.assertIn(metrics.iloc[0]["model"], fitted)

    def test_tune_selected_model(self):
        results, best_params, best_score = tune_selected_model(self.train, k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("mean_test_f2", results.columns)
        self.assertGreaterEqual(best_score, 0)
        self.assertIn("model__C", best_params)
        comparison, _, locked_variant = compare_default_vs_tuned_lr(self.train, self.dev, results)
        self.assertEqual(set(comparison["model"]), {"default_lr", "tuned_lr"})
        self.assertIn(locked_variant, {"default_lr", "tuned_lr"})
        self.assertIn("delta_dev_f2_vs_default", comparison.columns)

    def test_decision_rules_escalate_high_probability_case(self):
        order = self.test.iloc[0].copy()
        order["traffic_level"] = "High"
        order["distance_km"] = 9.5
        decision = get_dss_decision(order, 0.95)
        self.assertEqual(decision["priority"], "High")
        self.assertIn("backup driver", decision["recommended_action"])

    def test_risk_score_breakdown_matches_score(self):
        order = self.test.iloc[0].copy()
        explanation = explain_delay_risk_score(order, 0.75)
        self.assertEqual(
            {row["component"] for row in explanation},
            set(RISK_COMPONENT_WEIGHTS),
        )
        self.assertAlmostEqual(
            sum(row["weighted_contribution"] for row in explanation),
            calculate_delay_risk_score(order, 0.75),
            delta=0.06,
        )
        policy = risk_component_policy_spec()
        self.assertAlmostEqual(sum(row["weight"] for row in policy), 1.0)
        self.assertTrue(all(row["normalization"] for row in policy))

    def test_transport_assignment_uses_unique_driver_slots(self):
        assignments = solve_transport_assignment(top_n=8)
        self.assertEqual(len(assignments), 8)
        self.assertEqual(assignments["driver_slot"].nunique(), 8)
        self.assertTrue((assignments["estimated_assignment_cost"] >= 0).all())
        policy = transport_cost_policy_spec()
        self.assertGreaterEqual(len(policy), 4)
        self.assertTrue(all(item["formula"] and item["source"] for item in policy))

    def test_business_forecast_and_recommendations_run(self):
        df = load_dataset()
        forecast = forecast_monthly_demand(df, horizon=3)
        self.assertEqual((forecast["split"] == "future").sum(), 3)
        rules = recommendation_rules(df)
        self.assertGreater(len(rules), 0)

    def test_detailed_eda_tables_cover_delay_preference_and_geo(self):
        df = load_dataset()
        duration = duration_delay_profile(df)
        self.assertEqual(set(duration["is_delayed"]), {False, True})
        favorites = favorite_item_summary(df)
        self.assertIn("pizza_type", set(favorites["dimension"]))
        restaurants = restaurant_dependency_summary(df)
        self.assertEqual(len(restaurants), df["restaurant_name"].nunique())
        locations = location_dependency_summary(df)
        self.assertIn("state_code", locations.columns)
        states = state_dependency_summary(df)
        self.assertGreaterEqual(len(states), 1)


class PizzaForensicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()

    def test_recovered_deterministic_formulas_match(self):
        audit = deterministic_formula_audit(self.df)
        self.assertTrue((audit["max_abs_error"] <= 1e-9).all())
        self.assertTrue(audit["matches_exactly"].all())

    def test_delay_threshold_is_inferred_from_label(self):
        audit = infer_delay_threshold(self.df)
        exact_rules = set(audit.loc[audit["exact_match"], "rule"])
        self.assertIn("delivery_duration_min > 30", exact_rules)
        self.assertIn("delivery_duration_min >= 35", exact_rules)
        self.assertEqual(audit["max_observed_on_time_duration"].iloc[0], 30)
        self.assertEqual(audit["min_observed_delayed_duration"].iloc[0], 35)

    def test_payment_information_is_weak_after_distance_control(self):
        audit = feature_information_audit(self.df)
        payment = audit[audit["feature"] == "payment_method"].iloc[0]
        self.assertLess(payment["conditional_mi_given_distance_band_bits"], 0.06)

    def test_uniformity_audit_covers_topping_generator(self):
        audit = uniformity_tests(self.df)
        topping_tests = audit[audit["variable"] == "toppings_count"]
        self.assertEqual(len(topping_tests), 3)
        self.assertEqual(set(self.df["toppings_count"].unique()), {1, 2, 3, 4, 5})

    def test_brand_ablation_is_small_but_measurable(self):
        ablation = brand_ablation()
        without_brand = ablation[ablation["model"] == "compact_without_restaurant"].iloc[0]
        self.assertLess(abs(without_brand["delta_f2_vs_with_restaurant"]), 0.05)

    def test_preference_trend_forecast_share_mass(self):
        forecast = preference_trend_forecast(self.df, horizon=3)
        future = forecast[forecast["split"] == "future"].copy()
        category_count = future.groupby("dimension")["category"].nunique().sum()
        self.assertEqual(len(future), 3 * category_count)
        for _, group in future.groupby(["dimension", "order_period"]):
            self.assertAlmostEqual(group["forecast_share"].sum(), 1.0, places=6)


class PizzaDepthAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.train, cls.dev, cls.test = split_dataset(cls.df)

    def test_cross_validate_reports_mean_and_std(self):
        cv = cross_validate_models(self.train, k=3)
        self.assertGreaterEqual(len(cv), 6)
        for col in ("f2_mean", "f2_std", "mcc_mean", "roc_auc_mean"):
            self.assertIn(col, cv.columns)
        self.assertTrue((cv["f2_mean"] >= 0).all())

    def test_bootstrap_test_metrics_ci_is_ordered(self):
        bt = bootstrap_test_metrics(load_best_model(), self.test, n_boot=200)
        self.assertEqual(set(bt["metric"]), {"f2", "balanced_accuracy", "mcc", "recall"})
        self.assertTrue((bt["ci_low_2_5"] <= bt["ci_high_97_5"]).all())

    def test_fbeta_threshold_analysis_marks_one_best_per_beta(self):
        model = load_best_model()
        audit = fbeta_threshold_analysis(model, self.dev, thresholds=[0.3, 0.5, 0.7])
        self.assertEqual(set(audit["beta"]), {1.0, 2.0, 3.0})
        self.assertTrue((audit["threshold"].between(0, 1)).all())
        self.assertEqual(int(audit["is_best_for_beta"].sum()), 3)
        self.assertIn("fbeta_score", audit.columns)
        transfer = evaluate_threshold_policy_transfer(model, self.dev, self.test, audit)
        self.assertEqual(set(transfer["split"]), {"dev", "test"})
        self.assertIn("dev_best_f2", set(transfer["model"]))

    def test_model_stability_audit_uses_multiple_splits(self):
        audit = model_stability_audit(self.train, self.dev, n_runs=5)
        self.assertEqual(len(audit), 5)
        self.assertTrue((audit["f2"] >= 0).all())
        summary = summarize_model_stability(audit)
        self.assertEqual(summary["n_runs"], 5)
        self.assertIn("f2", summary["metrics"])
        self.assertIn("p05", summary["metrics"]["f2"])

    def test_delay_rate_with_ci_bounds(self):
        out = delay_rate_with_ci(self.df, "traffic_level")
        self.assertTrue((out["ci_low"] >= 0).all() and (out["ci_high"] <= 1).all())
        self.assertTrue((out["ci_low"] <= out["ci_high"]).all())

    def test_kmeans_cluster_profile_covers_all_rows(self):
        profile = kmeans_cluster_profile(self.df, k=4)
        self.assertEqual(len(profile), 4)
        self.assertEqual(int(profile["orders"].sum()), len(self.df))

    def test_mann_kendall_detects_direction(self):
        self.assertEqual(mann_kendall_trend([1, 2, 3, 4, 5, 6, 7, 8])["trend"], "increasing")
        self.assertEqual(mann_kendall_trend([5, 5, 5, 5, 5])["trend"], "no_trend")

    def test_preference_trend_and_forecast_comparison(self):
        self.assertGreater(len(preference_trend_tests(self.df)), 0)
        methods = set(forecast_method_comparison(self.df)["method"])
        self.assertEqual(methods, {"seasonal_naive_prior_year", "moving_average_3"})

    def test_mi_permutation_audit_has_noise_floor(self):
        audit = mi_permutation_audit(self.df, n_perm=40)
        self.assertEqual(len(audit), 7)
        self.assertIn("p_value", audit.columns)
        self.assertIn("verdict", audit.columns)

    def test_duration_generator_reconstruction_accuracy_in_range(self):
        recon = duration_generator_reconstruction(self.df)
        acc = float(recon.loc[recon["quantity"] == "round5_reconstruction_accuracy", "value"].iloc[0])
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_bootstrap_brand_delta_f2_ci(self):
        result = bootstrap_brand_delta_f2(n_boot=200)
        self.assertIn("ci_includes_zero", result.columns)
        self.assertLessEqual(result["ci_low_2_5"].iloc[0], result["ci_high_97_5"].iloc[0])


if __name__ == "__main__":
    unittest.main()
