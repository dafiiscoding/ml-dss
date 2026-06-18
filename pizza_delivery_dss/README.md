# Pizza Delivery Decision Support System

> 👉 **Mới vào / trái ngành?** Đọc [`START_HERE.md`](START_HERE.md) trước — kể
> dự án bằng lời thường, chỉ thứ tự đọc và cách hiểu các con số.

Dự án con này được tạo song song với dự án CrisisMMD hiện tại. Mục tiêu là giữ
quy trình học thuật đầy đủ của môn Hệ hỗ trợ quyết định nhưng giảm phạm vi về
một bài toán tabular nhỏ: dự báo và ưu tiên đơn giao pizza có nguy cơ trễ.

## Bài toán quyết định

Đầu vào là dữ liệu đơn hàng pizza từ Kaggle. Hệ thống hỗ trợ quản lý vận hành:

- dự báo đơn hàng có khả năng giao trễ trước khi giao;
- xếp mức ưu tiên Low, Medium, High;
- gợi ý hành động như theo dõi đơn, chuẩn bị tài xế dự phòng hoặc thông báo
  chăm sóc khách hàng;
- phân tích customer behavior: size/type/location/restaurant/payment;
- dự báo nhu cầu theo tháng và gợi ý staffing theo giờ;
- dự báo trend sở thích size/type theo share tháng, kèm caveat dữ liệu sinh;
- tạo recommendation rules theo popularity/context;
- truy ngược thuật toán sinh dữ liệu: công thức tất định, lượng tử hóa
  duration, kiểm định categorical noise/artifact, uniformity và brand ablation;
- tối ưu gán tài xế giả lập cho nhóm đơn High priority theo tinh thần bài toán
  vận tải/phân công;
- trình bày KPI, EDA, kết quả mô hình và hàng đợi ưu tiên trong Streamlit;
- chuẩn bị bộ CSV/DAX/spec để dựng dashboard trong Power BI.

## Dữ liệu

- Nguồn: `akshaygaikwad448/pizza-delivery-data-with-enhanced-features`
- File gốc: `Enhanced_pizza_sell_data_2024-25.xlsx`
- Kích thước hiện tại: 1.004 dòng, 25 cột gốc, 37 cột sau feature engineering.
- License Kaggle: CC0 Public Domain.

Dữ liệu gốc đã được tải vào:

```text
data/raw/Enhanced_pizza_sell_data_2024-25.xlsx
```

Nếu cần tải lại:

```powershell
cd pizza_delivery_dss
python -m scripts.download_data
```

## Quy tắc chống leakage

Mục tiêu mô hình là `is_delayed`. Vì đây là bài toán dự báo trước khi đơn hoàn
tất, pipeline cấm dùng các cột biết sau giao hàng làm feature:

- `delivery_time`
- `delivery_duration_min`
- `delivery_efficiency_min_per_km`
- `delay_min`
- `restaurant_avg_time`

File chỉ cung cấp nhãn `is_delayed`, không cung cấp tài liệu SLA. Audit suy
luận ranh giới nhãn nằm giữa 30 và 35 phút: trên lưới duration bội số 5,
`delivery_duration_min > 30` và `delivery_duration_min >= 35` đều khớp nhãn
0 mismatch. Vì vậy đưa `delivery_duration_min` hoặc `delay_min` vào mô hình sẽ
tạo kết quả ảo.

## Quy trình tái lập

```powershell
cd pizza_delivery_dss
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.run_all
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

Lệnh `.\.venv\Scripts\python.exe -m scripts.run_all` hiện chạy 18 bước:
audit data/split, EDA artifacts,
business/forecasting/synthetic-data artifacts, model training/evaluation,
data forensics/generator reverse-engineering artifacts, dashboard queue,
transportation optimization, Power BI data pack, sáu notebook đã execute,
LaTeX report PDF, Beamer slide PDF, unit tests và Streamlit smoke test.

Ghi chú môi trường: bản hoàn thiện đã được verify bằng Python 3.12 trong
`.venv`. Trên máy hiện tại không nên dùng trực tiếp lệnh `python` nếu nó trỏ
sang Python 3.14.

## Cấu trúc

```text
app/                  Dashboard Streamlit
data/raw/             Excel Kaggle gốc
data/processed/       Train/dev/test đã chuẩn hóa
docs/                 Roadmap, tiến độ, mô tả dữ liệu
models/               Mô hình đã huấn luyện
powerbi/              CSV, DAX measures và spec dựng Power BI
reports/figures/      Hình EDA và đánh giá
reports/metrics/      CSV/JSON kết quả
reports/latex/        Mã nguồn LaTeX báo cáo
scripts/              Runbook tái lập
slides/               Mã nguồn và PDF slide Beamer
src/pizza_dss/        Loader, EDA, model, DSS rules
tests/                Kiểm thử pipeline
```

## Độ phủ kiến thức môn học

| Nội dung học phần | Thể hiện trong dự án |
|---|---|
| DSS và quy trình ra quyết định | Delay Risk Score, Priority, Recommended Action |
| Supervised learning | Dự báo `is_delayed` |
| Decision Tree, Naive Bayes, k-NN, SVM, Ensemble | So sánh sáu classifier |
| Đánh giá mô hình | Train/dev/test, baseline, Accuracy, Balanced Accuracy, F1, F2, MCC |
| Clustering | K-Means phân nhóm đơn hàng theo đặc trưng trước giao |
| Association Analysis | Luật đồng xuất hiện giữa traffic, distance, size, peak hour và delay |
| Hypothesis Testing | Chi-square cho payment, size, type, location, traffic |
| Customer Behavior | Size/type/location/restaurant preference, same-type brand preference |
| Time Series | Monthly demand forecast, hourly staffing scenario, preference share forecast |
| Recommendation | Rule-based recommendation theo context và popularity |
| Data Quality | Synthetic/data realism audit, redundant feature audit, generator reverse-engineering |
| Optimization/Transportation | Gán tài xế giả lập cho đơn High priority bằng ma trận chi phí |
| BI/Dashboard | Streamlit dashboard, Power BI-ready pack và hàng đợi ưu tiên |

## Tài liệu nộp bản hoàn thiện

| Nội dung | File |
|---|---|
| Checklist nộp bài | [docs/SUBMISSION_CHECKLIST.md](docs/SUBMISSION_CHECKLIST.md) |
| Hướng dẫn đọc hiểu/trình bày | [docs/WORKFLOW_PRESENTATION_GUIDE.md](docs/WORKFLOW_PRESENTATION_GUIDE.md) |
| Hướng dẫn nội dung báo cáo PDF | [reports/REPORT_GUIDE.md](reports/REPORT_GUIDE.md) |
| Hướng dẫn nội dung slide | [slides/SLIDE_GUIDE.md](slides/SLIDE_GUIDE.md) |
| Hướng dẫn dựng Power BI | [powerbi/POWERBI_BUILD_GUIDE.md](powerbi/POWERBI_BUILD_GUIDE.md) |
| Bảng giảm tải scope report/slide | [docs/SCOPE_PRIORITIZATION.md](docs/SCOPE_PRIORITIZATION.md) |
| Plan dọn sạch project | [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md) |
| Hướng dẫn chấm | [INSTRUCTOR.md](INSTRUCTOR.md) |
| Bản đồ rubric | [docs/GRADING_MAP.md](docs/GRADING_MAP.md) |
| Báo cáo draft | [reports/final_report_draft.md](reports/final_report_draft.md) |
| Báo cáo PDF LaTeX | [reports/PIZZA_DSS_REPORT.pdf](reports/PIZZA_DSS_REPORT.pdf) |
| Slide PDF | [slides/PIZZA_DSS_SLIDE_DECK.pdf](slides/PIZZA_DSS_SLIDE_DECK.pdf) |
| Notebook data/audit | [notebooks/01_data_audit_preprocessing.ipynb](notebooks/01_data_audit_preprocessing.ipynb) |
| Notebook EDA | [notebooks/02_eda.ipynb](notebooks/02_eda.ipynb) |
| Notebook modeling | [notebooks/03_modeling.ipynb](notebooks/03_modeling.ipynb) |
| Notebook DSS/Optimization/Power BI | [notebooks/04_dss_optimization_powerbi.ipynb](notebooks/04_dss_optimization_powerbi.ipynb) |
| Notebook business/forecast/recommendation | [notebooks/05_business_forecasting_recommendation.ipynb](notebooks/05_business_forecasting_recommendation.ipynb) |
| Notebook data forensics | [notebooks/06_data_forensics.ipynb](notebooks/06_data_forensics.ipynb) |
| Data forensics artifacts | [reports/metrics/generator_reverse_engineering_summary.json](reports/metrics/generator_reverse_engineering_summary.json) |
| Hình forensics/trend | [reports/figures/feature_information_audit.png](reports/figures/feature_information_audit.png) |
| Power BI data pack | [powerbi/README.md](powerbi/README.md), [powerbi/POWERBI_BUILD_GUIDE.md](powerbi/POWERBI_BUILD_GUIDE.md) |

## Thông tin nhóm

| Thành viên | MSSV | Phân công chính | Tỷ lệ |
|---|---|---|---:|
| Đoàn Danh Long | 20237354 | Pipeline DSS, mô hình hóa, báo cáo | 25% |
| Vũ Quang Vinh | 20237408 | EDA, customer behavior, kiểm định giả thiết | 25% |
| Nguyễn Đăng Đức | 20237312 | Feature engineering, forecast, recommendation, Power BI | 25% |
| Nguyễn Tuấn Anh | 20237293 | Dashboard, tối ưu vận tải, kiểm thử, slide | 25% |

## Phạm vi

Đây là prototype học thuật trên dataset có nhiều dấu hiệu synthetic. Hệ thống
không điều phối tài xế thật, không gửi tin nhắn thật cho khách hàng và không
chứng minh chính sách priority/forecast/recommendation là tối ưu kinh doanh.
