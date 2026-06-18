# Hướng dẫn nội dung báo cáo PDF

File này giải thích báo cáo `PIZZA_DSS_REPORT.pdf` dùng để làm gì, đọc theo thứ
tự nào, mỗi chương cần chứng minh điều gì và trước khi nộp cần kiểm tra phần
nào. Source chính nằm ở `reports/latex/main.tex`.

Trước khi thêm nội dung mới vào report, đọc `docs/SCOPE_PRIORITIZATION.md` để
biết phần nào là lõi phải trình bày và phần nào chỉ nên để phụ lục/notebook.

## 1. Vai trò của báo cáo

Báo cáo là tài liệu nộp chính cho học phần. Mục tiêu không chỉ khoe điểm mô
hình, mà chứng minh nhóm làm đủ quy trình DSS:

- xác định bài toán quyết định và người dùng;
- kiểm soát dữ liệu, leakage và target;
- phân tích EDA, customer behavior, hypothesis testing, forecasting;
- xây mô hình supervised learning, chọn/tune đúng protocol;
- biến dự báo thành Risk Score, Priority, Recommended Action;
- minh họa tối ưu vận tải và dashboard;
- nêu rõ giới hạn vì dataset synthetic.

## 2. Cấu trúc nội dung

| Phần trong PDF | Nội dung phải đọc/kiểm tra | Minh chứng chính |
|---|---|---|
| Bảng phân công | Ai làm phần nào, tỷ lệ đóng góp | `docs/team_info.json`, phụ lục tự chấm |
| Lời mở đầu + Tóm tắt | Vấn đề, dữ liệu, model khóa, kết quả chính, caveat synthetic | `reports/metrics/model_summary.json` |
| Tám câu hỏi bắt buộc | Trả lời trực diện đề tài làm gì, ai dùng, quyết định gì, dữ liệu nào | Chương 1-6 |
| Bảng thuật ngữ | Giải thích nhanh DSS, leakage, F2, MCC, CI, Risk Score | `docs/GLOSSARY.md` |
| Chương 1 | Bối cảnh, phạm vi và câu hỏi nghiên cứu | `README.md`, `GOAL.md` |
| Chương 2 | Cơ sở lý thuyết DSS, pipeline, kỹ thuật dùng trong môn | `docs/GRADING_MAP.md` |
| Chương 3 | Dữ liệu, target, leakage, reverse-engineering, split | NB01, NB06, `reports/metrics/*forensics*` |
| Chương 4 | EDA, behavior, brand/location/state, kiểm định, forecast | NB02, NB05 |
| Chương 5 | Modeling, feature set, 6 classifier, tuning LR, F-beta threshold, 100-run stability, test và baseline | NB03, `model_*metrics.csv`, `fbeta_threshold_analysis.csv`, `model_stability_summary.json` |
| Chương 6 | Risk Score, component breakdown, queue, assignment, Streamlit, Power BI | NB04, `decision_rules.py`, `risk_component_policy_spec.csv`, `transport_cost_policy_spec.csv`, `powerbi/` |
| Chương 7 | Tái lập, đối chiếu rubric, khuyến nghị và giới hạn | `scripts.run_all`, tests |
| Kết luận | Kết quả đạt được và hướng phát triển | Toàn bộ artifact |
| Phụ lục | Artifact, tự chấm, khai báo AI | Nhóm cần rà trước khi nộp |

## 3. Các luận điểm phải nói đúng

- Không nói "30 phút là SLA". Báo cáo chỉ suy luận từ nhãn: duration on-time lớn
  nhất là 30, delayed nhỏ nhất là 35; vì duration nằm trên lưới 5 phút nên
  `>30` và `>=35` tương đương trên dữ liệu quan sát.
- Không dùng `delivery_duration_min`, `delivery_time`, `delay_min`,
  `delivery_efficiency_min_per_km`, `restaurant_avg_time` làm feature dự báo.
- Điểm model cao vì data gần tất định/synthetic; không claim hiệu năng sản xuất.
- Forecast, recommendation và assignment là prototype học thuật.
- Power BI trong repo là data pack CSV/DAX/spec, chưa phải file `.pbix`.
- Tuning chỉ áp dụng cho Logistic Regression sau khi đã chọn model; tuned
  `C=0.3` kém default `C=1.0` trên dev nên giữ default.
- F-beta threshold đã được thử trên dev: dev-best F2 threshold giảm FN nhưng khi
  transfer sang test thì FP tăng và F2 thấp hơn default; không dùng test để chọn
  lại threshold.
- 100-run stability chỉ dùng train+dev pool, không dùng test; mục đích là kiểm
  tra điểm cao có ổn định hay do một split may mắn.
- Forecast MAPE 45,7% không phải lỗi cần che hoặc cố tune; đó là bằng chứng
  forecast chỉ phù hợp minh họa planning trên data synthetic/partial.
- Recommendation chỉ là rule/popularity/context heuristic, không phải hệ
  recommender cá nhân hóa.
- Risk Score là weighted scoring policy minh bạch: từng component được chuẩn hóa
  về 0-100 rồi cộng trọng số; ngưỡng 35/65 là heuristic vận hành, không phải
  ngưỡng tối ưu thống kê tuyệt đối.
- Assignment dùng đơn thật trong queue nhưng driver/fleet/capacity là giả lập.
  Cost policy đã được tách thành bảng riêng để tránh giấu công thức trong code.

## 4. Phần cần nhóm kiểm tra trước khi nộp

1. Bảng phân công: tên, MSSV, tỷ lệ, mô tả công việc.
2. Phụ lục tự chấm: điền điểm tự chấm thật theo barem lớp.
3. Phụ lục khai báo AI: sửa đúng công cụ nhóm thật sự dùng.
4. Nếu lớp yêu cầu `.docx`, xuất PDF/LaTeX sang Word thủ công hoặc dùng công cụ
   chuyển đổi riêng.
5. Nếu lớp yêu cầu ảnh dashboard, chụp Streamlit hoặc Power BI Desktop rồi thêm
   vào phụ lục hoặc slide.

## 5. Cách rebuild

```powershell
cd pizza_delivery_dss
.\.venv\Scripts\python.exe -m scripts.build_report_pdf
```

Full verification:

```powershell
cd pizza_delivery_dss
.\.venv\Scripts\python.exe -m scripts.run_all
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Khi build xong, file nộp chính là `reports/PIZZA_DSS_REPORT.pdf`.
