# Checklist nộp bài - Pizza Delivery DSS

## Thư mục chính

Nộp toàn bộ thư mục `pizza_delivery_dss/` để giữ đủ code, dữ liệu, notebook,
artifact, report, slide và Power BI data pack.

## File nên mở trước

| Mục đích | File |
|---|---|
| Báo cáo chính | `reports/PIZZA_DSS_REPORT.pdf` |
| Slide thuyết trình | `slides/PIZZA_DSS_SLIDE_DECK.pdf` |
| Hướng dẫn chấm nhanh | `INSTRUCTOR.md` |
| Hướng dẫn đọc hiểu/trình bày | `docs/WORKFLOW_PRESENTATION_GUIDE.md` |
| Plan dọn sạch | `docs/CLEANUP_PLAN.md` |
| Mapping rubric | `docs/GRADING_MAP.md` |
| Tiến độ và kết quả khóa | `docs/PROGRESS.md` |
| Reverse-engineering dữ liệu | `reports/metrics/generator_reverse_engineering_summary.json` |
| Power BI pack | `powerbi/README.md` |

## Notebook theo module

1. `notebooks/01_data_audit_preprocessing.ipynb`
2. `notebooks/02_eda.ipynb`
3. `notebooks/03_modeling.ipynb`
4. `notebooks/04_dss_optimization_powerbi.ipynb`
5. `notebooks/05_business_forecasting_recommendation.ipynb`
6. `notebooks/06_data_forensics.ipynb`

## Lệnh kiểm tra tái lập

```powershell
cd pizza_delivery_dss
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.run_all
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

## Kết quả khóa

- Full runbook: `18/18` bước pass.
- Unit tests: `17/17` pass.
- Streamlit smoke test: pass.
- Model chính: Logistic Regression trên compact feature set.
- Test: Accuracy `0,9602`, F2 `0,9491`, MCC `0,8889`.
- Forensics: 7 công thức/target tất định khớp tuyệt đối, duration nằm trên
  lưới 5 phút, brand ablation dev F2 giảm `0,0343` khi bỏ restaurant.
- Threshold trễ là kết quả suy luận từ nhãn: ranh giới quan sát giữa 30 và 35
  phút, không phải SLA biết trước.
- Trend sở thích: có `preference_trend_forecast.csv` và hai hình share forecast
  cho pizza size/type; kết luận phải ghi là minh họa phương pháp vì dữ liệu sinh.
- Power BI-ready pack đã có fact/dim CSV, DAX measures và dashboard spec.

## Lưu ý khi bảo vệ

- Dataset có nhiều dấu hiệu synthetic; dự án đã audit và ghi rõ giới hạn.
- Không nói brand hoàn toàn vô dụng: kiểm định homogeneity reject đồng nhất,
  nên chỉ gộp 5 brand thành một chuỗi cho báo cáo cấp chuỗi, không diễn giải
  thành chất lượng vận hành thật.
- Không dùng `delivery_duration_min`, `delay_min`, `delivery_time`,
  `delivery_efficiency_min_per_km`, `restaurant_avg_time` làm feature dự báo.
- Forecast/recommendation/tối ưu vận tải là prototype học thuật, không phải
  claim sản xuất.
- Power BI hiện là data pack để dựng dashboard; nếu lớp yêu cầu `.pbix`, mở
  Power BI Desktop và import các file trong `powerbi/`.
