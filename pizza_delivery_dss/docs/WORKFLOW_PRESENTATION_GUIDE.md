# Hướng dẫn đọc hiểu và trình bày project Pizza DSS

Tài liệu này dùng khi một thành viên mới, giảng viên hoặc người review cần hiểu
dự án theo đúng thứ tự làm việc. Nên đọc theo luồng dưới đây thay vì mở file
ngẫu nhiên.

## 1. Đọc nhanh trong 10 phút

1. Mở `docs/PROGRESS.md` để lấy trạng thái khóa, số liệu chính và giới hạn.
2. Mở `README.md` để hiểu bài toán quyết định, dữ liệu, lệnh tái lập và cấu trúc.
3. Mở `docs/GRADING_MAP.md` để đối chiếu rubric môn học.
4. Mở `reports/PIZZA_DSS_REPORT.pdf` để đọc bài nộp chính.
5. Mở `slides/PIZZA_DSS_SLIDE_DECK.pdf` để xem flow thuyết trình.

## 2. Đọc theo thứ tự làm việc

### Bước 0 - Bài toán DSS

- File đọc: `README.md`, `INSTRUCTOR.md`.
- Câu hỏi cần trả lời:
  - Hệ thống hỗ trợ quyết định gì?
  - Người dùng là ai?
  - Output quyết định là gì?
- Ý chính để trình bày: dự án không chỉ dự báo trễ, mà biến xác suất trễ thành
  Risk Score, Priority, Recommended Action và hàng đợi điều phối.

### Bước 1 - Dữ liệu và leakage

- Code chính: `src/pizza_dss/data_loader.py`.
- Notebook: `notebooks/01_data_audit_preprocessing.ipynb`.
- Artifact:
  - `reports/metrics/data_quality_summary.json`
  - `data/processed/train.csv`
  - `data/processed/dev.csv`
  - `data/processed/test.csv`
- Ý chính:
  - Dataset có 1.004 đơn, 25 cột gốc, 37 cột sau FE.
  - File chỉ có nhãn `is_delayed`; threshold trễ là kết quả suy luận từ
    duration grid, không phải SLA biết trước.
  - Không được dùng duration/delay/delivery_time/restaurant_avg_time làm feature.
  - Chronological last-20% không có delayed sample nên dùng stratified split cho
    bản học thuật.

### Bước 2 - Data forensics và dữ liệu synthetic

- Code chính:
  - `src/pizza_dss/business_analysis.py`
  - `src/pizza_dss/data_forensics.py`
- Artifact:
  - `reports/metrics/synthetic_data_audit.csv`
  - `reports/metrics/delay_threshold_inference.csv`
  - `reports/metrics/generator_deterministic_formulas.csv`
  - `reports/metrics/duration_model_recovery.csv`
  - `reports/metrics/feature_information_audit.csv`
  - `reports/metrics/uniformity_tests.csv`
  - `reports/metrics/brand_homogeneity_tests.csv`
  - `reports/metrics/brand_ablation.csv`
  - `reports/metrics/generator_reverse_engineering_summary.json`
- Hình nên dùng:
  - `reports/figures/synthetic_data_flags.png`
  - `reports/figures/feature_information_audit.png`
  - `reports/figures/brand_delay_rate_homogeneity.png`
- Ý chính:
  - Có 7 công thức/target tất định khớp tuyệt đối.
  - Duration nằm trên lưới 5 phút từ 15 đến 50.
  - Ranh giới nhãn quan sát nằm giữa 30 và 35 phút; `>30` và `>=35` tương
    đương vì duration chỉ có bội số 5.
  - Brand có artifact nhỏ; không nói brand hoàn toàn vô dụng hoặc khác biệt chất
    lượng thật.
  - Dữ liệu đủ dùng cho bài học DSS nhưng không nên overclaim kết quả kinh doanh.

### Bước 3 - EDA và insight

- Code chính: `src/pizza_dss/eda.py`.
- Notebook: `notebooks/02_eda.ipynb`.
- Artifact:
  - `reports/metrics/delay_rate_by_traffic_level.csv`
  - `reports/metrics/delay_rate_by_distance_band.csv`
  - `reports/metrics/association_rules.csv`
  - `reports/metrics/kmeans_silhouette.csv`
- Hình nên dùng:
  - `reports/figures/delay_distribution.png`
  - `reports/figures/delay_rate_by_traffic.png`
  - `reports/figures/kmeans_silhouette.png`
- Ý chính:
  - Delayed là lớp thiểu số, nên Accuracy một mình không đủ.
  - Traffic/distance dẫn dắt phần lớn rủi ro.
  - Association rules và clustering là insight/giải thích, không phải bằng chứng
    nhân quả.

### Bước 4 - Customer behavior, forecast và recommendation

- Code chính: `src/pizza_dss/business_analysis.py`.
- Notebook: `notebooks/05_business_forecasting_recommendation.ipynb`.
- Artifact:
  - `reports/metrics/size_mix.csv`
  - `reports/metrics/type_mix.csv`
  - `reports/metrics/top_restaurant_by_type.csv`
  - `reports/metrics/monthly_demand_forecast.csv`
  - `reports/metrics/preference_trend_forecast.csv`
  - `reports/metrics/hourly_staffing_plan.csv`
  - `reports/metrics/recommendation_rules.csv`
- Hình nên dùng:
  - `reports/figures/size_preference.png`
  - `reports/figures/monthly_demand_forecast.png`
  - `reports/figures/preference_size_share_forecast.png`
  - `reports/figures/preference_type_share_forecast.png`
  - `reports/figures/hourly_staffing_plan.png`
- Ý chính:
  - Top size là Medium, top type là Non-Veg.
  - Demand forecast có MAPE 45,7%, chỉ là demo planning.
  - Preference trend forecast trả lời được câu hỏi "dự đoán trend về sau?", nhưng
    phải kèm caveat synthetic.
  - Recommendation là popularity/context rules, chưa phải collaborative filtering.

### Bước 5 - Modeling

- Code chính: `src/pizza_dss/modeling.py`, `src/pizza_dss/features.py`.
- Notebook: `notebooks/03_modeling.ipynb`.
- Artifact:
  - `reports/metrics/baseline_dev_metrics.csv`
  - `reports/metrics/model_dev_comparison.csv`
  - `reports/metrics/feature_set_comparison.csv`
  - `reports/metrics/model_test_metrics.csv`
  - `reports/metrics/model_summary.json`
  - `models/best_delay_model.joblib`
- Ý chính:
  - Fit trên train, chọn bằng dev theo F2, báo test một lần.
  - Model chính là Logistic Regression trên compact feature set.
  - Test: Accuracy 0,9602; Balanced Accuracy 0,9661; F2 0,9491; MCC 0,8889.
  - Luôn đặt cạnh baseline always-on-time và always-delayed.

### Bước 6 - DSS, dashboard và tối ưu vận tải

- Code chính:
  - `src/pizza_dss/decision_rules.py`
  - `src/pizza_dss/dashboard_data.py`
  - `src/pizza_dss/transport_optimization.py`
  - `app/streamlit_app.py`
- Notebook: `notebooks/04_dss_optimization_powerbi.ipynb`.
- Artifact:
  - `data/processed/delay_priority_queue.csv`
  - `reports/metrics/transport_assignment.csv`
  - `reports/metrics/transport_assignment_summary.json`
  - `powerbi/`
- Ý chính:
  - DSS chuyển xác suất thành Risk Score, Priority và Recommended Action.
  - Transportation scenario dùng đơn thật nhưng driver/fleet là giả lập.
  - Power BI hiện là data pack CSV/DAX/spec, không phải file `.pbix`.

## 2b. Bản đồ Hình → Slide và Dữ liệu → Power BI

**Quy ước.** `[SLIDE]` = hình nên đưa vào slide thuyết trình; `[PBI]` = bảng dữ
liệu nạp vào Power BI. Tất cả notebook giờ đã **nhúng hình inline** (cuộn là
thấy biểu đồ ngay cạnh bảng), nguồn PNG dùng chung với slide và báo cáo nằm ở
`reports/figures/`.

### Hình → Slide

| Hình (`reports/figures/`) | Trạng thái slide | Notebook nguồn | Dùng để nói gì |
|---|---|---|---|
| `delay_distribution.png` | [SLIDE] đã có | 01/02 | Lớp trễ là thiểu số (~21%) |
| `duration_grid_by_delay.png` | [SLIDE] nên thêm | 02 | Ranh giới nhãn nằm giữa 30 và 35 phút |
| `delay_severity_distribution.png` | tùy chọn | 02 | Mức độ trễ theo duration bucket |
| `delay_rate_by_traffic.png` | [SLIDE] đã có | 02 | Traffic dẫn dắt rủi ro |
| `delay_rate_by_distance_band.png` | [SLIDE] nên thêm | 02 | Distance band và rủi ro |
| `top_pizza_types.png` | tùy chọn | 02 | Món/type phổ biến |
| `restaurant_delay_rate.png` | tùy chọn | 02 | Brand workload/risk, kèm caveat artifact |
| `correlation_heatmap.png` | [SLIDE] nên thêm | 02 | Redundancy distance↔estimated_duration |
| `association_lift_top.png` | tùy chọn | 02 | Điều kiện đi với delayed |
| `synthetic_data_flags.png` | [SLIDE] đã có | 05/06 | Dữ liệu synthetic |
| `delay_threshold_inference.png` | [SLIDE] nên thêm | 06 | Suy luận ngưỡng nhãn 30–35 |
| `generator_deterministic_formula_errors.png` | tùy chọn | 06 | 7 công thức tất định |
| `feature_information_audit.png` | [SLIDE] đã có | 06 | MI sau kiểm soát distance (kèm caveat bias) |
| `brand_delay_rate_homogeneity.png` | [SLIDE] nên thêm | 06 | Brand chồng lấp, không khác biệt thật |
| `model_comparison.png` | [SLIDE] nên thêm | 03 | So sánh 6 classifier theo F2/MCC |
| `confusion_matrix_test.png` | [SLIDE] nên thêm | 03 | Bắt gần hết đơn trễ trên test |
| `roc_curves.png` | tùy chọn | 03 | Tách lớp giữa các model |
| `model_coefficients.png` | tùy chọn | 03 | Distance/traffic đẩy odds trễ |
| `size_preference.png` | [SLIDE] đã có | 05 | Medium phổ biến nhất |
| `monthly_demand_forecast.png` | [SLIDE] đã có | 05 | Demand + seasonal-naive (demo) |
| `preference_size_share_forecast.png` | [SLIDE] đã có | 05 | Trend share size (gần phẳng) |
| `hourly_staffing_plan.png` | tùy chọn | 05 | Peak 19h cho staffing |
| `priority_distribution.png` | [SLIDE] nên thêm | 04 | Cơ cấu Low/Med/High |
| `risk_score_histogram.png` | tùy chọn | 04 | Phân bố risk score + ngưỡng |
| `transport_assignment_cost.png` | tùy chọn | 04 | Chi phí gán tài xế giả lập |

Slide hiện đã chèn 7 hình (`pizza_dss_slides.tex`). Các hình `[SLIDE] nên thêm`
là khuyến nghị mở rộng; chèn thì rebuild bằng `scripts.build_slides_pdf`.

### Dữ liệu → Power BI

Bảng nạp vào Power BI nằm ở `powerbi/` (xem `powerbi/manifest.json`). Nguồn sinh
trong `src/pizza_dss/powerbi.py::build_powerbi_pack` (notebook 04):

| Bảng `[PBI]` | Loại | Nguồn |
|---|---|---|
| `fact_orders.csv` | fact | bảng đơn đã chuẩn hóa từ `data_loader` |
| `fact_delay_queue.csv` | fact | `data/processed/delay_priority_queue.csv` |
| `fact_transport_assignment.csv` | fact | `reports/metrics/transport_assignment.csv` |
| `fact_monthly_demand_forecast.csv` | fact | `reports/metrics/monthly_demand_forecast.csv` |
| `fact_hourly_staffing_plan.csv` | fact | `reports/metrics/hourly_staffing_plan.csv` |
| `fact_recommendation_rules.csv` | fact | `reports/metrics/recommendation_rules.csv` |
| `fact_hypothesis_tests.csv` | fact | `reports/metrics/hypothesis_tests.csv` |
| `fact_data_realism_audit.csv` | fact | `reports/metrics/synthetic_data_audit.csv` |
| `fact_product_type_mix.csv` | fact | `reports/metrics/type_mix.csv` |
| `fact_product_size_mix.csv` | fact | `reports/metrics/size_mix.csv` |
| `fact_top_restaurant_by_type.csv` | fact | `reports/metrics/top_restaurant_by_type.csv` |
| `fact_redundant_feature_audit.csv` | fact | `reports/metrics/redundant_feature_audit.csv` |
| `dim_restaurant.csv` | dim | unique `restaurant_name` từ `fact_orders` |
| `dim_location.csv` | dim | unique `location` từ `fact_orders` |
| `dim_date.csv` | dim | `order_date` từ `fact_orders` |

Quan hệ chính (đã ghi trong `dashboard_spec.md`): `fact_orders[order_id]` →
`fact_delay_queue[order_id]` → `fact_transport_assignment[order_id]`; và
`fact_orders` nối `dim_restaurant`/`dim_location`/`dim_date`. Đo lường ở
`powerbi/measures.dax`.

## 3. Script chạy lại

```powershell
cd pizza_delivery_dss
.\.venv\Scripts\python.exe -m scripts.run_all
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

Kết quả khóa hiện tại:

- `scripts.run_all`: 18/18 bước pass.
- Unit tests: 26/26 pass.
- Streamlit AppTest: pass.

## 4. Outline thuyết trình 10-12 phút

1. Bài toán quyết định: vì sao cần DSS, output là gì.
2. Dữ liệu và leakage: target, cột cấm, split.
3. Data realism/forensics: dữ liệu synthetic, công thức tất định, duration grid.
4. EDA insight: lớp trễ, traffic/distance, complexity.
5. Behavior và forecast: size/type, demand forecast, trend share, staffing.
6. Modeling: feature set compact, model comparison, metric test và baseline.
7. DSS layer: risk score, priority, recommended action, delay queue.
8. Tối ưu vận tải và Power BI: assignment scenario, dashboard/data pack.
9. Giới hạn: synthetic data, forecast demo, driver giả lập, priority chưa tối ưu cost thật.
10. Kết luận: dự án nhỏ nhưng đủ quy trình DSS môn học.

## 5. Các câu dễ bị hỏi khi bảo vệ

| Câu hỏi | Cách trả lời ngắn |
|---|---|
| Vì sao không dùng `delivery_duration_min`? | Đây là thông tin sau giao hàng; audit cho thấy nó phục dựng nhãn `is_delayed` với 0 mismatch, dùng nó sẽ leakage. |
| Sao biết 30 phút là trễ? | Không biết trước từ SLA; nhóm suy luận từ nhãn `is_delayed`: on-time max 30, delayed min 35, threshold sweep mismatch 0. |
| Vì sao không split theo thời gian? | Last-20% theo thời gian có 0 delayed sample, nên bản học thuật dùng stratified split và ghi rõ giới hạn. |
| Forecast có dùng được thật không? | Không. MAPE 45,7% và dữ liệu synthetic nên chỉ minh họa quy trình planning. |
| Brand nào tốt hơn? | Không kết luận vậy. Brand có artifact trong dữ liệu sinh, không phải bằng chứng chất lượng thật. |
| Power BI có `.pbix` chưa? | Chưa, project cung cấp data pack CSV/DAX/spec để dựng `.pbix` trong Power BI Desktop. |
| Tối ưu vận tải có dữ liệu tài xế thật không? | Không. Đơn hàng là thật từ file Kaggle, còn driver/capacity là scenario giả lập. |
