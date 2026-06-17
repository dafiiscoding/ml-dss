# Progress - Pizza Delivery DSS

> Cập nhật kiểm tra: 17/06/2026.

## Trạng thái

| Giai đoạn | Trạng thái | Bằng chứng |
|---|---|---|
| GĐ0 - Dữ liệu | Hoàn thành bản mở rộng | 1.004 dòng, 25 cột gốc, 37 cột sau FE; audit không missing/duplicate |
| GĐ1 - Tiền xử lý | Hoàn thành | Split train/dev/test 602/201/201, stratified theo `is_delayed` |
| GĐ2 - EDA/Business/Forensics | Hoàn thành bản mở rộng | Delay severity, duration grid, preference, brand/location/state dependency, association rules, K-Means, hypothesis tests, forecast, recommendation, reverse-engineering generator |
| GĐ3 - Mô hình | Hoàn thành bản mở rộng | Compact feature set, so sánh sáu classifier, chọn trên dev, báo test một lần |
| GĐ4 - DSS | Hoàn thành | Delay Risk Score, Priority, Recommended Action, queue CSV |
| GĐ5 - Tối ưu vận tải | Hoàn thành | Gán tài xế giả lập cho đơn High priority, cost matrix, assignment CSV |
| GĐ6 - Dashboard/BI | Hoàn thành bản mở rộng | Streamlit tám tab, Power BI data pack mở rộng, AppTest pass |
| GĐ7 - Minh chứng | Hoàn thành bản mở rộng | README, DATA, ROADMAP, Instructor guide, report draft/PDF chuẩn form học thuật DSS, slides, metrics, figures, notebooks, tests |

## Protocol khóa

1. Dự án pizza là dự án con, không thay thế CrisisMMD.
2. Chỉ dùng feature biết được trước hoặc tại thời điểm nhận đơn.
3. Không dùng `delivery_duration_min`, `delivery_time`, `delay_min`,
   `delivery_efficiency_min_per_km`, `restaurant_avg_time` làm feature dự báo.
4. Fit preprocessing trên train, chọn model trên dev, báo test sau khi khóa.
5. Luôn báo baseline always-on-time và always-delayed.
6. Priority/Risk là chính sách DSS minh bạch, không khẳng định tối ưu thống kê.
7. Bài toán vận tải dùng đơn thật nhưng fleet/tài xế là kịch bản giả lập vì
   dataset Kaggle không có bảng tài xế, depot hoặc capacity thật.
8. Dataset có nhiều dấu hiệu synthetic/rác; báo cáo phải nêu rõ data realism
   audit và không overclaim forecast/recommendation.
9. Phần reverse-engineering phải theo số liệu thật: brand có thể gộp để báo cáo
   cấp chuỗi, nhưng kiểm định homogeneity không ủng hộ kết luận các brand đồng
   nhất thống kê.

## Kết quả bản hoàn thiện

- Corpus: 1.004 đơn hàng.
- Target: 210 delayed, 794 on-time; delayed rate 20,92%.
- Audit leakage: source chỉ có nhãn `is_delayed`; suy luận từ dữ liệu cho thấy
  ranh giới hiệu dụng nằm giữa 30 và 35 phút. Trên lưới duration bội số 5,
  `delivery_duration_min > 30` và `delivery_duration_min >= 35` đều khớp nhãn
  0 mismatch.
- Chronological last-20% có 0 delayed, nên bản học thuật dùng split stratified
  cố định thay vì time holdout.
- Split: train 602, dev 201, test 201.
- Active feature set: `compact_nonredundant` gồm 12 feature, dùng
  `pizza_size_score` và bỏ các cột deterministic/trùng thông tin.
- Feature comparison dev: compact F2 0,9434; full pre-dispatch F2 0,9286.
- Model chọn theo dev F2: Logistic Regression.
- Test: Accuracy 0,9602; Balanced Accuracy 0,9661; Precision 0,8542;
  Recall 0,9762; F1 0,9111; F2 0,9491; MCC 0,8889.
- Baseline always-on-time: Accuracy 0,7910; F2 0; MCC 0.
- Baseline always-delayed: Accuracy 0,2090; F2 0,5691; MCC 0.
- Synthetic/data realism audit: 12 warning/critical flags.
- Generator reverse-engineering: 7 công thức/target tất định khớp tuyệt đối;
  duration nằm trên lưới 5 phút từ 15 đến 50; best OLS duration recovery R2
  0,8754.
- Information audit sau distance-band control: location và pizza_type còn
  strong artifact signal; payment_method, restaurant_name và traffic_level là
  weak artifact/sampling signal; pizza_size và is_weekend gần như noise.
- Uniformity GOF reject location, pizza_type, pizza_size và toppings_count;
  toppings_count chỉ có support 1-5, tập trung chủ yếu 2-5.
- Brand homogeneity: count mỗi brand 194-212 nhưng KS/chi-square reject đồng
  nhất; bỏ `restaurant_name` làm dev F2 giảm 0,0343, nên đây là artifact nhỏ
  nhưng không nên diễn giải là chất lượng thương hiệu thật.
- Hypothesis tests significant at 0,05: 8 categorical variables.
- Customer behavior: top size Medium, top type Non-Veg.
- Demand forecast: seasonal-naive prior-year, 19 backtest months, MAE 12,211,
  MAPE 45,7%, chỉ dùng làm demo planning.
- Preference trend forecast: dự báo share size/type theo tháng trong 6 tháng
  tới; có 9 trend p-value dưới 0,05 nhưng phải xem là artifact của dữ liệu sinh,
  không phải thị hiếu thật.
- Staffing scenario: peak hour 19h.
- Recommendation rules: 60 rule-based recommendations.
- Transportation scenario: 12 đơn High priority được gán cho 6 tài xế giả lập;
  mean assignment cost 32,84.
- Power BI pack: `powerbi/` gồm fact/dim CSV, `measures.dax`, dashboard spec và
  manifest.
- Detailed EDA artifacts: `duration_delay_profile.csv`,
  `duration_grid_by_delay.csv`, `delay_severity_distribution.csv`,
  `favorite_item_summary.csv`, `restaurant_dependency_summary.csv`,
  `location_dependency_summary.csv`, `state_dependency_summary.csv`,
  `restaurant_pizza_type_mix.csv`, `restaurant_pizza_size_mix.csv`,
  `state_pizza_type_mix.csv`, `state_pizza_size_mix.csv`.
- Forensics artifacts: `delay_threshold_inference.csv`,
  `generator_deterministic_formulas.csv`,
  `duration_model_recovery.csv`, `feature_information_audit.csv`,
  `uniformity_tests.csv`, `brand_homogeneity_tests.csv`, `brand_ablation.csv`,
  `pooled_chain_summary.csv`, `preference_trend_forecast.csv` và
  `generator_reverse_engineering_summary.json`.
- Detailed EDA figures: `duration_grid_by_delay.png`,
  `delay_severity_distribution.png`, `delay_rate_by_distance_band.png`,
  `delay_rate_by_complexity_band.png`, `top_pizza_types.png`,
  `pizza_type_size_heatmap.png`, `restaurant_delay_rate.png`,
  `restaurant_pizza_type_mix_heatmap.png`,
  `restaurant_pizza_size_mix_heatmap.png`, `state_delay_rate_top15.png`.
- Forensics figures: `generator_deterministic_formula_errors.png`,
  `duration_model_residual_histogram.png`, `feature_information_audit.png`,
  `uniformity_tests.png`, `brand_delay_rate_homogeneity.png`,
  `preference_type_share_forecast.png`, `preference_size_share_forecast.png`.
- Report PDF: `reports/PIZZA_DSS_REPORT.pdf`, đã viết lại theo form báo cáo
  học thuật DSS, 34 trang, có bìa, lời mở đầu, tóm tắt, thuật ngữ, chương/phụ
  lục và tài liệu tham khảo.
- Slide PDF: `slides/PIZZA_DSS_SLIDE_DECK.pdf`.
- Executed notebooks: `notebooks/01_data_audit_preprocessing.ipynb`,
  `notebooks/02_eda.ipynb`, `notebooks/03_modeling.ipynb`,
  `notebooks/04_dss_optimization_powerbi.ipynb`,
  `notebooks/05_business_forecasting_recommendation.ipynb`,
  `notebooks/06_data_forensics.ipynb`.
- Unit tests: 26/26 pass.
- Streamlit smoke test: pass.
- Full runbook: `python -m scripts.run_all` pass 18/18.
- Môi trường verify cuối: Python 3.12 trong `.venv`.

## Việc tùy chọn trước khi nộp

Không còn việc bắt buộc còn mở trong scope dự án con. Các việc dưới đây chỉ
cần làm nếu lớp yêu cầu thêm định dạng nộp cụ thể:

- Chụp screenshot dashboard nếu muốn đưa minh họa giao diện vào phần trình bày.
- Dựng file `.pbix` trong Power BI Desktop bằng bộ `powerbi/` nếu giảng viên yêu
  cầu nộp dashboard native.
- Xuất PPTX từ slide PDF/Beamer nếu lớp yêu cầu đúng định dạng PowerPoint.
