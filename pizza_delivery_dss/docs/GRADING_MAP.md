# Bản đồ tiêu chí chấm - Pizza Delivery DSS

Tài liệu này giúp bám rubric giống dự án CrisisMMD nhưng ở phạm vi nhỏ hơn.

| Tiêu chí | Cách đáp ứng | Artifact |
|---|---|---|
| Hiểu vấn đề và quyết định | Quản lý cần biết đơn nào có nguy cơ trễ để ưu tiên theo dõi/điều phối | `reports/final_report_draft.md` Chương 1 |
| Dữ liệu và tiền xử lý | Kaggle Excel thật, audit schema, missing, duplicate, leakage, split cố định | `docs/DATA.md`, `data_loader.py`, `data_quality_summary.json` |
| Phân tích và insight | Delay rate, synthetic audit, K-Means, association rules, hypothesis tests | `notebooks/02_eda.ipynb`, `synthetic_data_audit.csv`, `hypothesis_tests.csv` |
| Data forensics / reverse-engineering | Truy ngược công thức tất định, lượng tử hóa duration, MI categorical, uniformity GOF, brand ablation | `src/pizza_dss/data_forensics.py`, `generator_reverse_engineering_summary.json` |
| Customer behavior | Size/type/location/restaurant preference, same-type brand preference | `notebooks/05_business_forecasting_recommendation.ipynb`, `size_mix.csv`, `type_mix.csv` |
| Forecasting/staffing | Monthly demand forecast, hourly staffing scenario, preference share trend forecast | `monthly_demand_forecast.csv`, `hourly_staffing_plan.csv`, `preference_trend_forecast.csv` |
| Recommendation | Rule-based recommendation theo context/popularity | `recommendation_rules.csv` |
| Mô hình/hệ thống | Compact feature set, so sánh sáu classifier, chọn dev, tune LR, F-beta threshold audit, 100-run stability, báo test, lưu model | `notebooks/03_modeling.ipynb`, `model_dev_comparison.csv`, `fbeta_threshold_analysis.csv`, `model_stability_100runs.csv` |
| DSS/prescriptive layer | Delay Risk Score, Priority, Recommended Action, Delay Queue, Risk Score component breakdown | `decision_rules.py`, `delay_priority_queue.csv`, `risk_component_policy_spec.csv` |
| Tối ưu vận tải | Gán tài xế giả lập cho nhóm đơn High priority theo cost policy minh bạch | `notebooks/04_dss_optimization_powerbi.ipynb`, `transport_assignment.csv`, `transport_cost_policy_spec.csv` |
| Dashboard/BI | Streamlit tám tab và Power BI-ready data pack mở rộng, gồm Risk/transport policy tables | `app/streamlit_app.py`, `powerbi/`, `powerbi/POWERBI_BUILD_GUIDE.md` |
| Báo cáo/slide | LaTeX report PDF và slide PDF | `reports/PIZZA_DSS_REPORT.pdf`, `reports/REPORT_GUIDE.md`, `slides/PIZZA_DSS_SLIDE_DECK.pdf`, `slides/SLIDE_GUIDE.md` |
| Minh chứng tái lập | `run_all`, notebooks executed, tests pass | `scripts/run_all.py`, `tests/test_pipeline.py` |
| Giới hạn và khuyến nghị | Nêu leakage, split, target definition, Priority chưa tối ưu thống kê | `reports/final_report_draft.md` |

## Mapping kiến thức môn học

| Kiến thức | Trong dự án |
|---|---|
| DSS | Tầng dự báo + tầng quyết định + dashboard |
| Decision making | Chọn đơn cần theo dõi trước, phân mức ưu tiên, gợi ý hành động |
| Supervised learning | Dự báo `is_delayed` |
| Decision Tree | Một classifier so sánh |
| Naive Bayes | GaussianNB trên feature đã transform |
| k-NN | Một classifier so sánh |
| SVM | SVC có probability để dùng xác suất delay |
| Ensemble | Random Forest |
| Clustering | K-Means trên feature trước-giao-hàng |
| Association rules | Luật traffic/distance/size/peak hour dẫn tới delayed |
| Hypothesis testing | Chi-square cho payment, size, type, location, traffic |
| Time series | Monthly demand forecast và hourly staffing |
| Preference trend forecasting | Dự báo share size/type theo tháng, kèm caveat synthetic |
| Recommendation | Popularity-based recommendation rules |
| Feature engineering | Size score 1-4, compact feature set, redundant feature audit |
| Hyperparameter tuning | Tune Logistic Regression bằng GridSearchCV trên train, sau đó so default/tuned trên dev |
| Threshold optimization | Quét F1/F2/F3 trên dev và kiểm tra transfer sang test, không chọn bằng test |
| Stability audit | 100 lần repeated stratified holdout trên train+dev để kiểm soát split may mắn |
| Data quality | Synthetic/data realism audit, generator reverse-engineering, uniformity tests |
| Transportation optimization | Bài toán phân công/vận tải dạng chi phí tối thiểu cho tài xế giả lập |
| Weighted scoring model | Chuẩn hóa các component về 0-100 rồi cộng có trọng số thành Risk Score |
| BI/Dashboard | Streamlit trình bày KPI và hàng đợi, Power BI pack để dựng báo cáo |

## Checklist trước khi nộp

- [x] `.\.venv\Scripts\python.exe -m scripts.run_all` chạy pass 18/18.
- [x] `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` chạy pass 30/30.
- [x] Sáu notebook module đã execute.
- [x] Báo cáo PDF LaTeX đã build.
- [x] Slide PDF đã build.
- [x] Power BI data pack đã sinh trong `powerbi/`.
- [x] Báo cáo không dùng số liệu ngoài `reports/metrics/`.
- [x] Báo cáo nêu rõ leakage columns bị cấm.
- [x] Dashboard smoke test pass.
