import json

import pandas as pd

from pizza_dss.config import PROJECT_ROOT
from pizza_dss.dashboard_data import load_dashboard_data
from pizza_dss.data_loader import load_dataset

POWERBI_DIR = PROJECT_ROOT / "powerbi"


def _write_readme():
    content = """# PowerBI Dashboard Pack - Pizza Delivery DSS

This folder contains Power BI-ready CSV tables and DAX measures. The repository
does not generate a `.pbix` file automatically because Power BI Desktop is not a
scriptable dependency in this environment.

For detailed page-by-page build instructions, relationships, visuals, and page
notes, read `POWERBI_BUILD_GUIDE.md`.

## Tables

- `fact_orders.csv`: cleaned source order table.
- `fact_delay_queue.csv`: model/DSS output sorted by delay risk.
- `fact_transport_assignment.csv`: transportation-problem scenario assignments.
- `fact_transport_cost_policy.csv`: transparent cost formula for the assignment scenario.
- `fact_monthly_demand_forecast.csv`: historical monthly demand and seasonal-naive forecast.
- `fact_hourly_staffing_plan.csv`: scenario staffing by order hour.
- `fact_recommendation_rules.csv`: popularity-based recommendation rules.
- `fact_hypothesis_tests.csv`: chi-square hypothesis tests.
- `fact_data_realism_audit.csv`: synthetic/data-quality warning flags.
- `fact_risk_component_policy.csv`: Risk Score component formulas, weights, and rationale.
- `fact_product_type_mix.csv`: order-share and delay-rate by pizza type.
- `fact_product_size_mix.csv`: order-share and delay-rate by encoded pizza size.
- `dim_restaurant.csv`: restaurant dimension.
- `dim_location.csv`: location dimension.
- `dim_date.csv`: date dimension from order dates.

## Suggested Power BI Pages

1. Executive Overview
   - Cards: Orders, Delayed Orders, Delay Rate, High Priority Orders.
   - Bar: Delay rate by traffic level.
   - Line/column: Orders by month.

2. Delay Drivers
   - Matrix: Traffic Level x Pizza Size with delay rate.
   - Bar: Delay rate by restaurant.
   - Scatter: Distance vs delayed probability.

3. Model and DSS Queue
   - Table: order_id, restaurant, location, risk score, priority, action.
   - Table: Risk Score component weights and normalization.
   - Slicer: priority, traffic, restaurant.

4. Transportation Scenario
   - Table: order-driver assignments.
   - Table: cost formula terms and data source.
   - Bar: assigned orders by driver.
   - KPI: mean assignment cost.

5. Demand Forecasting and Staffing
   - Line: historical orders and forecast orders by month.
   - Bar: scenario orders by hour.
   - KPI: recommended peak-hour staff.

6. Customer Behavior and Recommendation
   - Bar: order share by size/type.
   - Matrix: pizza type by restaurant.
   - Table: recommendation rules with confidence.

7. Data Realism Audit
   - Table: synthetic-data checks by severity.
   - Bar: count of warnings/critical flags.

## Import Steps

1. Open Power BI Desktop.
2. Get Data -> Text/CSV for each CSV in this folder.
3. Create relationships:
   - `fact_orders[order_id]` -> `fact_delay_queue[order_id]`
   - `fact_delay_queue[order_id]` -> `fact_transport_assignment[order_id]`
   - `fact_orders[restaurant_name]` -> `dim_restaurant[restaurant_name]`
   - `fact_orders[location]` -> `dim_location[location]`
   - `fact_orders[order_date]` -> `dim_date[order_date]`
4. Paste measures from `measures.dax` into a Measures table.
5. Build the pages using `dashboard_spec.md`.

## Scope Note

The predictive model and queue use real Kaggle orders. Driver capacity in the
transportation page is a deterministic coursework scenario because the Kaggle
file has no driver table.
"""
    (POWERBI_DIR / "README.md").write_text(content, encoding="utf-8")


def _write_measures():
    content = """Orders = COUNTROWS(fact_orders)

Delayed Orders = CALCULATE([Orders], fact_orders[is_delayed] = TRUE())

Delay Rate = DIVIDE([Delayed Orders], [Orders])

Average Delivery Duration = AVERAGE(fact_orders[delivery_duration_min])

Average Distance = AVERAGE(fact_orders[distance_km])

High Priority Orders =
CALCULATE(COUNTROWS(fact_delay_queue), fact_delay_queue[priority] = "High")

Average Delay Risk = AVERAGE(fact_delay_queue[delay_risk_score])

Assigned Scenario Orders = COUNTROWS(fact_transport_assignment)

Average Assignment Cost = AVERAGE(fact_transport_assignment[estimated_assignment_cost])

Forecast Orders = SUM(fact_monthly_demand_forecast[forecast_orders])

Scenario Orders Per Day = SUM(fact_hourly_staffing_plan[scenario_orders_per_day])

Recommended Staff = SUM(fact_hourly_staffing_plan[recommended_staff])

Significant Hypothesis Tests =
CALCULATE(COUNTROWS(fact_hypothesis_tests), fact_hypothesis_tests[significant_at_0_05] = TRUE())

Synthetic Warning Flags =
CALCULATE(COUNTROWS(fact_data_realism_audit), fact_data_realism_audit[severity] IN {"warning", "critical"})
"""
    (POWERBI_DIR / "measures.dax").write_text(content, encoding="utf-8")


def _write_dashboard_spec():
    content = """# PowerBI Dashboard Specification

Detailed import steps, relationships, visual field choices and caveats are in
`POWERBI_BUILD_GUIDE.md`. This file is the short page specification used by the
pipeline artifact.

## Page 1 - Executive Overview

Purpose: show whether delivery delay is an operational issue and where workload
is concentrated.

Visuals:

- KPI cards: Orders, Delayed Orders, Delay Rate, High Priority Orders.
- Clustered bar: Delay Rate by Traffic Level.
- Column chart: Orders by Order Month.
- Donut: Priority split from `fact_delay_queue`.

## Page 2 - Delay Drivers

Purpose: explain the conditions associated with delayed orders.

Visuals:

- Matrix: rows `traffic_level`, columns `pizza_size`, values Delay Rate.
- Bar: Delay Rate by Restaurant.
- Scatter: `distance_km` vs `delayed_probability`, color `priority`.
- Slicers: Restaurant, Traffic Level, Pizza Size.

## Page 3 - Model and DSS Queue

Purpose: turn model output into an action queue.

Visuals:

- Table sorted by `delay_risk_score`: order_id, restaurant, location, priority,
  delayed_probability, recommended_action.
- Bar: High/Medium/Low counts.
- Card: Average Delay Risk.
- Table: `fact_risk_component_policy` with component, weight, formula and rationale.

## Page 4 - Transportation Scenario

Purpose: demonstrate prescriptive DSS inspired by the transportation problem.

Visuals:

- Table: order-driver assignments.
- Table: `fact_transport_cost_policy` with term, formula, source and effect.
- Bar: Assigned Orders by Driver.
- Card: Average Assignment Cost.
- Bar: Assignment Cost by Order.

Scope note: driver data is simulated because the source dataset does not include
drivers, depots, or capacity.

## Page 5 - Demand Forecasting and Staffing

Purpose: support planning around future demand and shift allocation.

Visuals:

- Line chart: `actual_orders` and `forecast_orders` by `order_period`.
- Bar: `scenario_orders_per_day` by `order_hour`.
- Card: peak recommended staff.
- Note: seasonal-naive forecast is a coursework planning demo.

## Page 6 - Customer Behavior and Recommendation

Purpose: show size/type preference, trend and simple recommendation rules.

Visuals:

- Bar: order share by `pizza_size_label`.
- Bar: order share by `pizza_type`.
- Table: top restaurant for each pizza type.
- Table: recommendation rules sorted by confidence.

## Page 7 - Data Realism Audit

Purpose: make the synthetic nature of the dataset explicit.

Visuals:

- Bar: count of checks by severity.
- Table: check, evidence, interpretation.
- Table: redundant feature audit for modeling decisions.
"""
    (POWERBI_DIR / "dashboard_spec.md").write_text(content, encoding="utf-8")


def build_powerbi_pack():
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    orders = load_dataset()
    queue = load_dashboard_data()
    assignment_path = PROJECT_ROOT / "reports" / "metrics" / "transport_assignment.csv"
    if assignment_path.exists():
        assignments = pd.read_csv(assignment_path)
    else:
        assignments = pd.DataFrame()
    metric_tables = {
        "fact_monthly_demand_forecast.csv": "monthly_demand_forecast.csv",
        "fact_hourly_staffing_plan.csv": "hourly_staffing_plan.csv",
        "fact_recommendation_rules.csv": "recommendation_rules.csv",
        "fact_hypothesis_tests.csv": "hypothesis_tests.csv",
        "fact_data_realism_audit.csv": "synthetic_data_audit.csv",
        "fact_risk_component_policy.csv": "risk_component_policy_spec.csv",
        "fact_transport_cost_policy.csv": "transport_cost_policy_spec.csv",
        "fact_product_type_mix.csv": "type_mix.csv",
        "fact_product_size_mix.csv": "size_mix.csv",
        "fact_top_restaurant_by_type.csv": "top_restaurant_by_type.csv",
        "fact_redundant_feature_audit.csv": "redundant_feature_audit.csv",
    }

    orders.to_csv(POWERBI_DIR / "fact_orders.csv", index=False)
    queue.to_csv(POWERBI_DIR / "fact_delay_queue.csv", index=False)
    assignments.to_csv(POWERBI_DIR / "fact_transport_assignment.csv", index=False)
    for output_name, metric_name in metric_tables.items():
        path = PROJECT_ROOT / "reports" / "metrics" / metric_name
        table = pd.read_csv(path) if path.exists() else pd.DataFrame()
        table.to_csv(POWERBI_DIR / output_name, index=False)

    orders[["restaurant_name"]].drop_duplicates().sort_values("restaurant_name").to_csv(
        POWERBI_DIR / "dim_restaurant.csv", index=False
    )
    orders[["location"]].drop_duplicates().sort_values("location").to_csv(
        POWERBI_DIR / "dim_location.csv", index=False
    )
    dim_date = (
        orders[["order_date", "order_year", "order_month_num", "order_month"]]
        .drop_duplicates()
        .sort_values("order_date")
    )
    dim_date.to_csv(POWERBI_DIR / "dim_date.csv", index=False)

    _write_readme()
    _write_measures()
    _write_dashboard_spec()

    manifest = {
        "tables": [
            "fact_orders.csv",
            "fact_delay_queue.csv",
            "fact_transport_assignment.csv",
            *metric_tables.keys(),
            "dim_restaurant.csv",
            "dim_location.csv",
            "dim_date.csv",
        ],
        "dax": "measures.dax",
        "dashboard_spec": "dashboard_spec.md",
        "build_guide": "POWERBI_BUILD_GUIDE.md",
        "note": "Create the .pbix manually in Power BI Desktop using this pack.",
    }
    (POWERBI_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(build_powerbi_pack())
