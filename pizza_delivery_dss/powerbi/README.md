# PowerBI Dashboard Pack - Pizza Delivery DSS

This folder contains Power BI-ready CSV tables and DAX measures. The repository
does not generate a `.pbix` file automatically because Power BI Desktop is not a
scriptable dependency in this environment.

## Tables

- `fact_orders.csv`: cleaned source order table.
- `fact_delay_queue.csv`: model/DSS output sorted by delay risk.
- `fact_transport_assignment.csv`: transportation-problem scenario assignments.
- `fact_monthly_demand_forecast.csv`: historical monthly demand and seasonal-naive forecast.
- `fact_hourly_staffing_plan.csv`: scenario staffing by order hour.
- `fact_recommendation_rules.csv`: popularity-based recommendation rules.
- `fact_hypothesis_tests.csv`: chi-square hypothesis tests.
- `fact_data_realism_audit.csv`: synthetic/data-quality warning flags.
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
   - Slicer: priority, traffic, restaurant.

4. Transportation Scenario
   - Table: order-driver assignments.
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
