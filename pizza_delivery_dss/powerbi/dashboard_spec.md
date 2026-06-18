# PowerBI Dashboard Specification

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
