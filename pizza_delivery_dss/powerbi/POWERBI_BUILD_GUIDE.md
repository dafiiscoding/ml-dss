# Hướng dẫn dựng Power BI dashboard

Repo cung cấp Power BI-ready data pack trong thư mục `powerbi/`, gồm CSV, DAX
measures và dashboard spec. Repo không tự sinh `.pbix` vì Power BI Desktop không
phải dependency scriptable trong môi trường này.

## 1. File cần dùng

| File | Vai trò |
|---|---|
| `fact_orders.csv` | Bảng đơn hàng đã chuẩn hóa, dùng cho KPI, traffic, distance, behavior |
| `fact_delay_queue.csv` | Output DSS: probability, risk score, priority, recommended action |
| `fact_transport_assignment.csv` | Kịch bản gán tài xế giả lập |
| `fact_transport_cost_policy.csv` | Công thức cost và nguồn dữ liệu của kịch bản assignment |
| `fact_monthly_demand_forecast.csv` | Demand forecast theo tháng |
| `fact_hourly_staffing_plan.csv` | Staffing scenario theo giờ |
| `fact_recommendation_rules.csv` | Recommendation rules theo popularity/context |
| `fact_hypothesis_tests.csv` | Kiểm định giả thiết |
| `fact_data_realism_audit.csv` | Audit dữ liệu synthetic/rác |
| `fact_risk_component_policy.csv` | Công thức, normalization, weight và lý do của Risk Score |
| `fact_product_type_mix.csv` | Mix và delay rate theo pizza type |
| `fact_product_size_mix.csv` | Mix và delay rate theo pizza size |
| `fact_top_restaurant_by_type.csv` | Restaurant nổi bật theo từng pizza type |
| `fact_redundant_feature_audit.csv` | Bằng chứng feature deterministic/trùng thông tin |
| `dim_restaurant.csv` | Dimension restaurant |
| `dim_location.csv` | Dimension location |
| `dim_date.csv` | Dimension date |
| `measures.dax` | Các measure DAX cần tạo trong Power BI |
| `dashboard_spec.md` | Spec ngắn theo từng page |

## 2. Import và model relationship

1. Mở Power BI Desktop.
2. Chọn Get Data -> Text/CSV, import toàn bộ CSV trong `powerbi/`.
3. Đặt data type:
   - `order_date`: Date.
   - `order_month`, `order_hour`, `distance_km`, probabilities, risk score:
     numeric.
   - `is_delayed`, `true_is_delayed`: Boolean nếu Power BI nhận diện được; nếu
     không thì giữ Text và xử lý filter theo `"True"`/`"False"`.
4. Tạo relationship:
   - `fact_orders[order_id]` -> `fact_delay_queue[order_id]`
   - `fact_delay_queue[order_id]` -> `fact_transport_assignment[order_id]`
   - `fact_orders[restaurant_name]` -> `dim_restaurant[restaurant_name]`
   - `fact_orders[location]` -> `dim_location[location]`
   - `fact_orders[order_date]` -> `dim_date[order_date]`
5. Tạo một Measures table rỗng, paste toàn bộ measure trong `measures.dax`.

## 3. Page-by-page dashboard design

### Page 1 - Executive Overview

Mục tiêu: cho quản lý thấy tình hình giao trễ tổng quan.

Visual nên có:

- Cards: `Orders`, `Delayed Orders`, `Delay Rate`, `High Priority Orders`.
- Donut: count theo `fact_delay_queue[priority]`.
- Column chart: orders theo `fact_orders[order_month]`.
- Bar chart: `Delay Rate` theo `fact_orders[traffic_level]`.
- Slicers: restaurant, location, traffic level.

Thông điệp: delayed là lớp thiểu số nhưng đủ quan trọng để cần DSS; High
priority là nhóm cần xử lý trước.

### Page 2 - Delay Drivers

Mục tiêu: giải thích yếu tố nào đi cùng rủi ro trễ.

Visual nên có:

- Matrix: rows `traffic_level`, columns `pizza_size`, values `Delay Rate`.
- Scatter: `distance_km` vs `delayed_probability`, legend `priority`.
- Bar: delay rate theo restaurant hoặc location.
- Table nhỏ: top group có delay rate cao.

Thông điệp: traffic và distance là driver vận hành; brand/location có artifact
vì dữ liệu sinh, không kết luận chất lượng thật.

### Page 3 - Model and DSS Queue

Mục tiêu: biến dự báo thành hàng đợi hành động.

Visual nên có:

- Table sorted descending theo `delay_risk_score`: `order_id`,
  `restaurant_name`, `location`, `priority`, `delayed_probability`,
  `recommended_action`.
- Cards: `Average Delay Risk`, `High Priority Orders`.
- Bar: số đơn theo priority.
- Table nhỏ: `fact_risk_component_policy[component]`, `weight`,
  `score_formula`, `rationale` để giải thích Risk Score.
- Slicers: priority, restaurant, traffic.

Thông điệp: DSS không chỉ trả xác suất, mà chuyển thành Priority/action cụ thể
và giải thích được từng điểm Risk Score cho quản lý.

### Page 4 - Transportation Scenario

Mục tiêu: minh họa bài toán phân công/vận tải.

Visual nên có:

- Table: `order_id`, `driver_id`, `driver_slot`, `estimated_assignment_cost`.
- Table nhỏ: `fact_transport_cost_policy[term]`, `formula`, `source`, `effect`.
- Bar: assigned orders by driver.
- Card: `Average Assignment Cost`, `Assigned Scenario Orders`.
- Bar: assignment cost by order.

Thông điệp: đơn là từ Kaggle, nhưng driver/fleet là giả lập vì dataset không có
bảng tài xế. Đây là prototype prescriptive DSS.

### Page 5 - Demand Forecasting and Staffing

Mục tiêu: hỗ trợ planning nhu cầu và ca làm.

Visual nên có:

- Line chart: actual vs forecast orders theo tháng từ
  `fact_monthly_demand_forecast.csv`.
- Bar: `scenario_orders_per_day` theo `order_hour`.
- Card: peak recommended staff.
- Table: hourly staffing plan.

Thông điệp: seasonal-naive forecast chỉ minh họa phương pháp, MAPE cao nên không
dùng như dự báo sản xuất.

### Page 6 - Customer Behavior and Recommendation

Mục tiêu: thể hiện preference và rule gợi ý.

Visual nên có:

- Bar: order share theo `pizza_size_label`.
- Bar: order share theo `pizza_type`.
- Matrix/table: `fact_top_restaurant_by_type.csv`.
- Table: recommendation rules sorted by confidence hoặc lift.

Thông điệp: Medium và Non-Veg nổi bật trong dữ liệu; recommendation là
popularity-based, chưa phải collaborative filtering.

### Page 7 - Data Realism Audit

Mục tiêu: nói rõ dữ liệu synthetic/rác để tránh overclaim.

Visual nên có:

- Bar: count theo severity trong `fact_data_realism_audit.csv`.
- Table: check, evidence, interpretation.
- Table: redundant feature audit.
- Card: synthetic warning flags.

Thông điệp: nhóm kiểm soát data quality thay vì che điểm yếu; kết quả model cao
phải đọc cùng caveat synthetic.

## 4. Sau khi dựng xong `.pbix`

- Lưu file là `powerbi/Pizza_Delivery_DSS.pbix` nếu lớp yêu cầu nộp native
  dashboard.
- Không commit file `.pbix` nếu repo/GitHub quá nặng hoặc lớp chỉ cần source.
- Nếu nộp PDF/report, có thể chụp 1-2 screenshot Power BI để đưa vào phụ lục.

## 5. Rebuild data pack

Nếu pipeline thay đổi, sinh lại pack bằng:

```powershell
cd pizza_delivery_dss
.\.venv\Scripts\python.exe -m scripts.build_powerbi_pack
```

Full rebuild:

```powershell
cd pizza_delivery_dss
.\.venv\Scripts\python.exe -m scripts.run_all
```
