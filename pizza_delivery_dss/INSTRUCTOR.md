# Hướng dẫn chấm nhanh - Pizza Delivery DSS

## Thông tin bài nộp

- Học phần: Hệ hỗ trợ quyết định
- Đề tài: Hệ hỗ trợ quyết định điều phối giao pizza dựa trên dự báo đơn trễ
- Giảng viên/giảng viên giao đề tài: TS. Lê Hải Hà; TS. Trịnh Đình Thăng
- Dataset: Kaggle `akshaygaikwad448/pizza-delivery-data-with-enhanced-features`
- Báo cáo draft: `reports/final_report_draft.md`
- Báo cáo PDF: `reports/PIZZA_DSS_REPORT.pdf`
- Slide PDF: `slides/PIZZA_DSS_SLIDE_DECK.pdf`
- Power BI pack: `powerbi/`

## Lộ trình đọc

1. Đọc `reports/PIZZA_DSS_REPORT.pdf` hoặc source
   `reports/latex/main.tex` để nắm bài toán, dữ liệu, EDA, mô hình, DSS,
   customer behavior, forecasting, recommendation, tối ưu vận tải và giới hạn.
2. Mở lần lượt sáu notebook trong `notebooks/` để kiểm tra quy trình theo
   module: data/audit, EDA, modeling, DSS/optimization/Power BI, business,
   data forensics.
3. Đối chiếu metric trong `reports/metrics/`, đặc biệt nhóm reverse-engineering
   mới nếu cần kiểm tra phần data forensics.
4. Kiểm tra slide tại `slides/PIZZA_DSS_SLIDE_DECK.pdf`.
5. Kiểm tra Power BI-ready pack tại `powerbi/`.
6. Chạy dashboard:

```powershell
cd pizza_delivery_dss
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

## Bản đồ nội dung

| Nội dung | File kiểm chứng |
|---|---|
| Dữ liệu và audit leakage | `src/pizza_dss/data_loader.py`, `reports/metrics/data_quality_summary.json` |
| EDA và insight | `notebooks/02_eda.ipynb`, `src/pizza_dss/eda.py` |
| Synthetic/data audit | `src/pizza_dss/business_analysis.py`, `reports/metrics/synthetic_data_audit.csv` |
| Generator reverse-engineering | `src/pizza_dss/data_forensics.py`, `reports/metrics/generator_reverse_engineering_summary.json` |
| Deterministic formulas | `reports/metrics/generator_deterministic_formulas.csv`, `reports/figures/generator_deterministic_formula_errors.png` |
| Categorical artifact/noise audit | `reports/metrics/feature_information_audit.csv`, `reports/figures/feature_information_audit.png` |
| Brand gộp chuỗi/ablation | `reports/metrics/brand_homogeneity_tests.csv`, `reports/metrics/brand_ablation.csv` |
| Hypothesis tests | `reports/metrics/hypothesis_tests.csv` |
| Customer behavior | `reports/metrics/size_mix.csv`, `reports/metrics/type_mix.csv`, `reports/metrics/top_restaurant_by_type.csv` |
| Forecasting/staffing | `reports/metrics/monthly_demand_forecast.csv`, `reports/metrics/hourly_staffing_plan.csv`, `reports/metrics/preference_trend_forecast.csv` |
| Recommendation | `reports/metrics/recommendation_rules.csv` |
| Sáu classifier | `src/pizza_dss/modeling.py`, `reports/metrics/model_dev_comparison.csv` |
| Feature set comparison | `reports/metrics/feature_set_comparison.csv` |
| Baseline | `reports/metrics/baseline_test_metrics.csv` |
| DSS Risk/Priority/Action | `src/pizza_dss/decision_rules.py` |
| Tối ưu vận tải | `src/pizza_dss/transport_optimization.py`, `reports/metrics/transport_assignment.csv` |
| Dashboard queue | `data/processed/delay_priority_queue.csv` |
| Power BI pack | `powerbi/fact_orders.csv`, `powerbi/measures.dax`, `powerbi/dashboard_spec.md` |
| Report/slide | `reports/PIZZA_DSS_REPORT.pdf`, `slides/PIZZA_DSS_SLIDE_DECK.pdf` |
| Kiểm thử | `tests/test_pipeline.py` |

## Giao thức đánh giá

- Không dùng các cột sau-giao-hàng làm feature: delivery duration, delivery
  time, delay, delivery efficiency, restaurant average time.
- Fit preprocessing và model trên train.
- Chọn model bằng dev theo F2.
- Báo test sau khi khóa model.
- Luôn đặt model cạnh baseline always-on-time và always-delayed.
- Risk Score và Priority là chính sách hỗ trợ quyết định, không phải nhãn tối
  ưu thống kê.
- Tối ưu vận tải dùng đơn hàng thật nhưng fleet/tài xế giả lập vì dataset không
  có bảng tài xế thật.
- Dataset có nhiều dấu hiệu synthetic; forecast/recommendation là demo quy
  trình DSS, không phải claim sản xuất.
- Reverse-engineering cho thấy nhiều cột là công thức tất định; brand có artifact
  nhỏ trong model nhưng không nên diễn giải là khác biệt vận hành thật.

## Kết quả khóa bản hoàn thiện

| Chỉ số | Logistic Regression test |
|---|---:|
| Accuracy | 0,9602 |
| Balanced Accuracy | 0,9661 |
| Precision | 0,8542 |
| Recall | 0,9762 |
| F1 | 0,9111 |
| F2 | 0,9491 |
| MCC | 0,8889 |
| ROC-AUC | 0,9961 |

Baseline always-on-time có Accuracy 0,7910 nhưng F2 bằng 0 vì không phát hiện
đơn trễ nào. Baseline always-delayed có Recall 1,0 nhưng Accuracy 0,2090 và MCC
0.

## Giới hạn cần lưu ý

1. Dataset nhỏ và có tính mô phỏng; chưa đại diện cho mọi chuỗi pizza thật.
2. File chỉ cung cấp nhãn `is_delayed`; audit suy luận ranh giới nhãn nằm giữa
   30 và 35 phút. Kết quả sẽ sai lệch nếu dùng nhầm cột duration/delay làm
   feature.
3. Chronological last-20% không có đơn trễ; dự án dùng stratified split để phục
   vụ học thuật và ghi rõ giới hạn này.
4. Feature compact được chọn vì dữ liệu có nhiều cột deterministic/trùng thông
   tin; full feature set chỉ dùng để so sánh.
5. Forecast seasonal-naive có MAPE cao, chỉ dùng minh họa planning.
6. Priority là rule minh bạch cho dashboard, chưa được tối ưu bằng dữ liệu chi
   phí kinh doanh thật.
7. Preference trend và brand analysis là bằng chứng về generator/data artifact,
   không phải kết luận thị trường pizza thật.
