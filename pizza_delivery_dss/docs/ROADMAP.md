# Roadmap - Pizza Delivery DSS

Tài liệu này là kế hoạch cho dự án con pizza delivery. Dự án CrisisMMD gốc được
giữ nguyên.

## GĐ0 - Dữ liệu và môi trường

- Tải file Excel Kaggle vào `data/raw/`.
- Audit schema, số dòng, missing value, duplicate order id.
- Kiểm tra leakage: suy luận xem `is_delayed` có thể phục dựng bằng luật ngưỡng
  trên `delivery_duration_min` hay không; không coi 30 phút là SLA biết trước.

Artifact:

- `reports/metrics/data_quality_summary.json`
- `data/processed/train.csv`
- `data/processed/dev.csv`
- `data/processed/test.csv`

## GĐ1 - Tiền xử lý và feature engineering

- Chuẩn hóa tên cột sang snake_case.
- Sửa giá trị nhà hàng bị lệch dấu apostrophe.
- Tách feature được phép dùng trước giao hàng và feature bị cấm do leakage.
- Tạo split stratified train/dev/test cố định.

Quy tắc:

- Fit encoder/scaler trên train.
- Chọn model bằng dev.
- Chỉ báo test sau khi khóa model.

## GĐ2 - EDA

- Phân bố delay, nhà hàng, traffic, distance, pizza size/type, payment.
- Delay rate theo traffic, distance band, size, hour, restaurant.
- K-Means để khám phá nhóm đơn hàng.
- Association analysis để tìm luật vận hành dễ hiểu.
- Customer behavior: size/type/location/restaurant/payment preference.
- Kiểm định giả thiết chi-square giữa biến categorical và delay.
- Audit dữ liệu synthetic/rác: timestamp, duration, formula-derived columns.
- Data forensics: truy ngược công thức tất định, duration grid, MI categorical,
  uniformity GOF và brand ablation.

Artifact:

- `reports/figures/delay_distribution.png`
- `reports/figures/delay_rate_by_traffic.png`
- `reports/figures/kmeans_silhouette.png`
- `reports/metrics/association_rules.csv`
- `reports/metrics/hypothesis_tests.csv`
- `reports/metrics/synthetic_data_audit.csv`
- `reports/metrics/generator_reverse_engineering_summary.json`
- `reports/metrics/feature_information_audit.csv`
- `reports/metrics/size_mix.csv`
- `reports/metrics/type_mix.csv`

## GĐ2b - Forecasting, staffing và recommendation

- Time-series monthly demand và seasonal-naive forecast.
- Preference trend forecast cho share size/type theo tháng.
- Hourly demand mix để đề xuất staffing scenario.
- Recommendation rules theo popularity/context: top type theo nhà hàng, top
  nhà hàng theo location, top nhà hàng cho cùng pizza type.
- Sales reporting theo size/type/restaurant/location.

Artifact:

- `reports/metrics/monthly_demand_forecast.csv`
- `reports/metrics/preference_trend_forecast.csv`
- `reports/metrics/hourly_staffing_plan.csv`
- `reports/metrics/recommendation_rules.csv`
- `reports/figures/monthly_demand_forecast.png`
- `reports/figures/hourly_staffing_plan.png`

## GĐ3 - Mô hình

So sánh tám classifier (để đáp ứng checklist 8 kiến trúc mô hình):

- Logistic Regression
- Decision Tree
- Gaussian Naive Bayes
- k-NN
- SVM
- Random Forest
- Gradient Boosting
- Multi-layer Perceptron (ANN)

Metric:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1
- F2
- MCC
- ROC-AUC

Baseline bắt buộc:

- Always on-time
- Always delayed

Mô hình chính dùng feature set compact để giảm cột deterministic/trùng thông
tin và dùng `pizza_size_score` cho Small/Medium/Large/XL. Full pre-dispatch
feature set được giữ để sensitivity analysis.

Model được chọn bằng F2 trên dev vì bỏ sót đơn trễ gây chi phí vận hành cao.
F2 không được báo một mình; phải đặt cạnh Balanced Accuracy và MCC.

## GĐ4 - Tầng DSS

- Chuyển xác suất delayed thành `delay_risk_score`.
- Xếp `priority`: Low, Medium, High.
- Sinh `recommended_action`.
- Hàng đợi priority được sắp theo risk giảm dần.

Priority là chính sách minh bạch, không phải ground truth thống kê.

## GĐ5 - Tối ưu vận tải/phân công

- Lấy nhóm đơn High priority từ hàng đợi DSS.
- Tạo kịch bản tài xế giả lập có capacity và hệ số chi phí.
- Tối thiểu hóa chi phí gán đơn/tài xế bằng Hungarian assignment nếu có
  `scipy`, hoặc greedy fallback để vẫn tái lập được.
- Báo rõ đây là kịch bản prescriptive DSS vì dataset không có bảng tài xế thật.

Artifact:

- `reports/metrics/transport_assignment.csv`
- `reports/metrics/transport_assignment_summary.json`

## GĐ6 - Dashboard và Power BI

Dashboard Streamlit gồm:

1. Overview
2. EDA
3. Customer Behavior
4. Forecast & Staffing
5. Model Evaluation
6. Single Order Demo
7. Delay Queue
8. Data Quality

Power BI pack gồm fact/dim CSV, DAX measures, manifest và dashboard spec để
nhập vào Power BI Desktop.

Artifact:

- `app/streamlit_app.py`
- `powerbi/fact_orders.csv`
- `powerbi/measures.dax`
- `powerbi/dashboard_spec.md`
- `powerbi/fact_monthly_demand_forecast.csv`
- `powerbi/fact_recommendation_rules.csv`
- `powerbi/fact_data_realism_audit.csv`

## GĐ7 - Báo cáo, slide và minh chứng

- Notebook module đã chạy:
  - `notebooks/01_data_audit_preprocessing.ipynb`
  - `notebooks/02_eda.ipynb`
  - `notebooks/03_modeling.ipynb`
  - `notebooks/04_dss_optimization_powerbi.ipynb`
  - `notebooks/05_business_forecasting_recommendation.ipynb`
  - `notebooks/06_data_forensics.ipynb`
- Metric CSV/JSON.
- Hình trong `reports/figures/`.
- README, DATA, PROGRESS, WORKFLOW_PRESENTATION_GUIDE, CLEANUP_PLAN.
- LaTeX report PDF và Beamer slide PDF.
- Screenshot dashboard nếu cần nộp.

Artifact:

- `reports/PIZZA_DSS_REPORT.pdf`
- `slides/PIZZA_DSS_SLIDE_DECK.pdf`

## Runbook

```powershell
cd pizza_delivery_dss
python -m scripts.run_all
python -m unittest discover -s tests -v
streamlit run app/streamlit_app.py
```

`scripts.run_all` chạy 18 bước từ audit dữ liệu đến build PDF/slide, Power BI
pack, kiểm thử và Streamlit smoke test.
