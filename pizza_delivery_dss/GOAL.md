# GOAL — Pizza Delivery DSS

> File goal **chuẩn** cho mọi agent. Đọc file này đầu tiên để biết mục tiêu,
> tiêu chí đạt, protocol, kết quả khóa, khoảng trống còn lại và thứ tự sửa.
> Pizza là project con trong repo `ml-dss`, không thay thế project CrisisMMD.
>
> Đọc kèm: `START_HERE.md`, `README.md`, `docs/ROADMAP.md`,
> `docs/PROGRESS.md`, `docs/GRADING_MAP.md`, `docs/SCOPE_PRIORITIZATION.md`,
> `docs/WORKFLOW_PRESENTATION_GUIDE.md`, `reports/REPORT_GUIDE.md`,
> `slides/SLIDE_GUIDE.md`, `powerbi/POWERBI_BUILD_GUIDE.md`.
>
> Cập nhật: 18/06/2026.

---

## 0. Goal Tổng

Xây prototype **Hệ hỗ trợ quyết định điều phối giao pizza**: dự báo đơn có nguy
cơ giao trễ **trước khi giao**, giải thích rủi ro, chuyển dự báo thành **Risk
Score → Priority → Recommended Action → Queue/Assignment**, rồi trình bày bằng
report, slide, Streamlit và Power BI pack theo chuẩn bài học DSS.

Điểm bắt buộc của project: dataset nhỏ và có nhiều dấu hiệu synthetic, nên mục
tiêu không phải chứng minh năng lực sản xuất ngoài đời. Mục tiêu là làm đủ quy
trình, kiểm soát leakage, phân tích trung thực, và biến kết quả thành quyết định
minh bạch.

---

## 1. Legend Trạng Thái

- ✅ = đã có artifact và đã verify.
- 🟡 = đã có một phần nhưng cần trình bày/giải thích/đo thêm để đạt chuẩn mới.
- ⬜ = chưa làm hoặc cần bổ sung ở pha sau.
- ⚠️ = caveat bắt buộc phải nói khi trình bày.

Verification gần nhất: `python -m scripts.run_all` pass **18/18**,
`python -m unittest discover -s tests -v` pass **30/30**.

---

## 2. Goal Chi Tiết Theo Tiêu Chí Đo Được

### G1 — Bài Toán Quyết Định Và Phạm Vi DSS

| Tiêu chí đo được | Trạng thái |
|---|---|
| Nêu rõ người dùng: quản lý/điều phối vận hành giao pizza. | ✅ |
| Nêu rõ quyết định cần hỗ trợ: đơn nào có nguy cơ trễ, ưu tiên mức nào, cần hành động gì trước khi trễ. | ✅ |
| Output DSS bắt buộc: `delayed_probability`, `delay_risk_score`, `priority`, `recommended_action`, `delay_priority_queue`. | ✅ |
| Phân biệt rõ predictive DSS (dự báo), diagnostic/EDA (giải thích), prescriptive DSS (assignment/queue). | ✅ |
| Không trình bày project như bài ML thuần; report/slide phải nhấn mạnh ra quyết định. | ✅ |

### G2 — Dữ Liệu, Target, Feature Engineering Và Leakage

| Tiêu chí đo được | Trạng thái |
|---|---|
| Audit dữ liệu gốc: số dòng/cột, missing, duplicate, date range, class balance. | ✅ |
| Target `is_delayed` được dùng từ file, không tự tạo nhãn để train. | ✅ |
| Ngưỡng trễ được **suy luận từ dữ liệu**: duration on-time max 30, delayed min 35; `>30` và `>=35` tương đương vì duration nằm trên lưới 5 phút. | ✅ |
| Cấm feature hậu nghiệm: `delivery_duration_min`, `delivery_time`, `delay_min`, `delivery_efficiency_min_per_km`, `restaurant_avg_time`. | ✅ |
| Mỗi FE có lý do: `pizza_size_score`, `time_segment`, `complexity_band`, `distance_band`, `order_period`, `order_weekday`. | ✅ |
| Compact feature set bỏ các cột deterministic/trùng thông tin và giải thích bằng evidence audit. | ✅ |

### G3 — Modeling, Bayes, Tuning Và F-Beta

| Tiêu chí đo được | Trạng thái |
|---|---|
| So sánh 6 classifier: Logistic Regression, Decision Tree, Gaussian Naive Bayes, k-NN, SVM, Random Forest. | ✅ |
| Báo rõ Naive Bayes **đã thử** nhưng không chọn vì dev F2/MCC kém và giả định độc lập feature không phù hợp dữ liệu này. | ✅ |
| Chọn model trên dev theo F2; báo test đúng một lần sau khi khóa. | ✅ |
| Luôn đặt cạnh baseline always-on-time và always-delayed. | ✅ |
| Cross-validation 5-fold có mean±std để xem ổn định. | ✅ |
| Hyperparameter tuning theo nguyên tắc select-then-tune: tune chỉ Logistic Regression sau khi LR thắng vòng chọn model. | ✅ |
| Tuning LR: GridSearchCV trên train, scoring F2, lưới `C ∈ {0.1, 0.3, 1, 3, 10}`; CV chọn `C=0.3` nhưng dev kém default nên giữ `C=1.0`. | ✅ |
| **F-beta/threshold analysis**: kiểm tra F1/F2/F3 hoặc PR threshold trên dev để giải thích vì sao F2/ngưỡng hiện tại hợp lý. Artifact: `fbeta_threshold_analysis.csv`, `fbeta_threshold_policy_transfer.csv`, `fbeta_threshold_curve.png`. | ✅ |
| **100-run stability audit**: repeated split/CV 100 lần trên train/dev pool để xem điểm cao có ổn định hay ăn may một split. Không dùng test trong audit này. Artifact: `model_stability_100runs.csv`, `model_stability_summary.json`, `model_stability_f2_distribution.png`. | ✅ |
| Report/slide phải có câu trả lời trực diện: “sao không dùng Bayes?”, “sao không optimize F-beta sâu hơn?”, “vì sao điểm cao?”. | ✅ |

### G4 — Risk Score 0-100 Và Priority Policy

| Tiêu chí đo được | Trạng thái |
|---|---|
| Công thức Risk Score được nêu rõ: `0.55 model + 0.15 traffic + 0.12 distance + 0.08 peak + 0.06 complexity + 0.04 weekend`. | ✅ |
| Giải thích từng thành phần `S_model`, `S_traffic`, `S_distance`, `S_peak`, `S_complexity`, `S_weekend` được chuẩn hóa về 0-100 như thế nào. | ✅ |
| Giải thích vì sao trọng số như vậy: model là bằng chứng chính, phần còn lại là áp lực vận hành minh bạch. | ✅ |
| Ngưỡng priority 35/65 được gọi đúng là **policy heuristic**, không phải tối ưu thống kê. | ✅ |
| Có calibration theo risk band: risk cao phải có observed delay rate cao hơn. | ✅ |
| Có sensitivity ngưỡng 35/65: đổi ngưỡng thì số đơn High/captured delayed thay đổi thế nào. | ✅ |
| Có `risk_component_policy_spec.csv`, `risk_component_breakdown.csv`, `risk_component_breakdown.png`, `risk_calibration.csv`, `priority_threshold_sensitivity.csv` và report/Streamlit/slide đọc được breakdown, không chỉ hiện score cuối. | ✅ |
| Single Order Demo phải hiển thị probability, từng component, Risk Score, Priority và Recommended Action. | ✅ |

### G5 — Prescriptive DSS: Assignment/Transportation Scenario

| Tiêu chí đo được | Trạng thái |
|---|---|
| Tách rõ assignment là tầng prescriptive riêng, không trộn với model training. | ✅ |
| Input thật: các đơn High priority trong queue/test orders. | ✅ |
| Input giả lập: 6 tài xế/slot/capacity vì Kaggle không có bảng tài xế thật. | ✅ |
| Cost formula phải minh bạch: chi phí gán phụ thuộc vào risk/distance/proxy/capacity hoặc quy tắc đã định. | ✅ |
| Algorithm: Hungarian nếu thư viện khả dụng, greedy fallback nếu không. | ✅ |
| Output bắt buộc: `transport_driver_scenario.csv`, `transport_assignment.csv`, `transport_assignment_summary.json`, `transport_cost_policy_spec.csv`. | ✅ |
| Report/slide phải nói rõ đây là demo transportation/assignment, không claim điều phối tài xế thật. | ✅ |
| Figure/table trình bày assignment cost và kết quả gán trong report/slide/dashboard: `transport_assignment_cost.png`, assignment table, summary KPI. | ✅ |

### G6 — EDA, Business Analysis Và Data Forensics

| Tiêu chí đo được | Trạng thái |
|---|---|
| EDA có delay distribution, traffic, distance, complexity, duration grid, severity. | ✅ |
| Customer behavior có size/type/location/restaurant/payment, same-type preferred restaurant. | ✅ |
| Hypothesis testing có p-value/effect size/caveat nhóm nhỏ. | ✅ |
| Forecast demand và staffing có backtest/MAE/MAPE/caveat. | ✅ |
| Trend preference size/type có forecast share và caveat synthetic. | ✅ |
| Recommendation là rule/popularity/context-based và ghi rõ giới hạn. | ✅ |
| Forensics chứng minh ≥7 công thức/target deterministic với `max_abs_error ≈ 0`. | ✅ |
| Permutation-MI phân biệt signal/noise/artifact sau distance control. | ✅ |
| Brand homogeneity + ablation + bootstrap ΔF2, không overclaim brand thật. | ✅ |
| Report/slide phải biến “data rác” thành điểm mạnh học thuật: biết audit, biết caveat, biết không overclaim. | ✅ |

### G7 — Streamlit Dashboard

Streamlit không chỉ “pass AppTest”. Nó là demo DSS cho quản lý vận hành.

| Tab/Yêu cầu | Vai trò | Trạng thái |
|---|---|---|
| Overview | KPI tổng quan: orders, delayed rate, restaurants, high priority; hình duration/traffic. | ✅ |
| EDA | Cho chọn group-by và xem delay rate theo nhóm. | ✅ |
| Customer Behavior | Size/type preference, same-type restaurant, recommendation rules. | ✅ |
| Forecast & Staffing | Demand forecast, MAE/MAPE, hourly staffing scenario. | ✅ |
| Model Evaluation | So sánh model, test metrics, baselines, tuning LR, F-beta threshold transfer và 100-run stability. | ✅ |
| Single Order Demo | Người dùng nhập đơn giả định và xem probability/risk/priority/action kèm breakdown component. | ✅ |
| Delay Queue | Filter priority/traffic, sort theo risk, download CSV. | ✅ |
| Data Quality | Synthetic audit + hypothesis tests kèm caveat dữ liệu synthetic/rác. | ✅ |
| UI role | Dashboard show vai trò ra quyết định ở từng tab, có filter/tương tác và ưu tiên queue/action. | ✅ |

### G8 — Report PDF, Slide Beamer Và Power BI

Report/slide là nơi thầy và người xem nhìn trực tiếp nhất. Notebook là minh
chứng quy trình, không phải deliverable kể chuyện chính.

| Deliverable | Tiêu chí đo được | Trạng thái |
|---|---|---|
| Report PDF | Có flow học thuật: bài toán → dữ liệu → forensics → EDA/business → modeling/tuning → DSS → assignment/dashboard → giới hạn. | ✅ |
| Report PDF | Mỗi phần quan trọng phải trả lời: làm gì, vì sao, kỹ thuật, bằng chứng, quyết định. | ✅ |
| Report PDF | Có mục riêng hoặc đoạn rõ cho Bayes, F-beta, 100-run stability, Risk Score breakdown, transport cost. | ✅ |
| Report PDF | Phụ lục phân công, tự chấm, khai báo AI cần nhóm điền/xác nhận thật. | 🟡 |
| Slide Beamer | Source `.tex` build được, không sửa tay PDF. | ✅ |
| Slide Beamer | Flow thuyết trình chuẩn: problem → data → forensics → EDA → modeling → tuning/stability → DSS → transport → dashboard → caveat. | ✅ |
| Slide Beamer | Có slide riêng cho Risk Score formula, model+tuning, stability, transport assignment, Streamlit/Power BI role. | ✅ |
| Slide Beamer | Chuẩn trình bày: mỗi slide có một thông điệp chính, bảng/hình đủ đọc, không dump notebook, có kết luận/caveat ngay trên slide. Speaking notes trong `slides/SLIDE_GUIDE.md` phải khớp nội dung slide. | ✅ |
| Power BI | Data pack có fact/dim CSV, DAX, manifest, dashboard spec, build guide. | ✅ |
| Power BI | Nếu lớp yêu cầu `.pbix`, dựng thủ công trong Power BI Desktop theo guide và ghi rõ không sinh tự động. | 🟡 |

### G9 — Reproducibility Và Kiểm Thử

| Tiêu chí đo được | Trạng thái |
|---|---|
| `scripts.run_all` chạy full pipeline 18/18. | ✅ |
| Unit tests pass 30/30. | ✅ |
| 6 notebook execute 0 lỗi và khớp `reports/metrics/*`. | ✅ |
| Report PDF + slide PDF build được. | ✅ |
| Streamlit AppTest pass. | ✅ |
| Power BI pack rebuild không ghi đè mất guide/link. | ✅ |

---

## 3. Protocol Không Thương Lượng

1. Pizza là **project con**, không sửa/đổi mục tiêu CrisisMMD.
2. Chỉ dùng feature biết trước/tại lúc nhận đơn.
3. Cấm tuyệt đối feature hậu nghiệm:
   `delivery_duration_min`, `delivery_time`, `delay_min`,
   `delivery_efficiency_min_per_km`, `restaurant_avg_time`.
4. Fit preprocessing/model trên train; chọn model/tune trên dev hoặc CV trong
   train; báo test một lần sau khi khóa.
5. F-beta/threshold analysis và 100-run stability audit chỉ dùng train/dev
   hoặc repeated CV; không dùng test để chọn lại model/ngưỡng.
6. Luôn báo baseline always-on-time và always-delayed.
7. F2 là metric chính vì FN đắt hơn FP, nhưng phải báo kèm Accuracy, Balanced
   Accuracy, F1, MCC, confusion matrix và CI.
8. Risk/Priority là chính sách minh bạch, không phải tối ưu thống kê tuyệt đối.
9. Assignment dùng đơn thật nhưng driver/fleet/capacity là giả lập.
10. Dataset synthetic/rác: mọi forecast/recommendation/brand conclusion phải có
   caveat, không claim thị trường thật.
11. Nếu 100-run stability vẫn cho điểm rất cao, phải giải thích là do generator/
    target gần tất định, không phải bằng chứng sản xuất ngoài đời.
12. Artifact sinh ở script/module; notebook/report/slide đọc artifact để trình
    bày, tránh mỗi nơi tính một kiểu.

---

## 4. Kết Quả Khóa Hiện Tại

- Corpus: 1.004 đơn; 210 delayed / 794 on-time; delayed rate 20,92%.
- Target audit: `is_delayed` khớp `delivery_duration_min > 30` với 0 mismatch;
  vì duration là bội số 5, `>30` và `>=35` tương đương trên data quan sát.
- Chronological last-20% có 0 delayed ⇒ dùng stratified split cho bản học thuật.
- Split: train/dev/test = 602/201/201.
- Active feature set: `compact_nonredundant`, 12 feature.
- Model khóa: Logistic Regression default `C=1.0`.
- Tuning: GridSearchCV chọn `C=0.3` theo CV F2, nhưng dev F2 giảm từ 0,9434
  xuống 0,8894 ⇒ giữ default.
- Test locked model: Accuracy 0,9602; Balanced Accuracy 0,9661; Precision
  0,8542; Recall 0,9762; F1 0,9111; F2 0,9491; MCC 0,8889.
- Baseline: always-on-time F2 0, MCC 0; always-delayed F2 0,5691, MCC 0.
- CV 5-fold: SVM F2 0,945±0,043; Logistic Regression F2 0,939±0,032.
- Risk policy hiện tại: 35/65 thresholds; calibration và sensitivity đã có.
- Transport scenario hiện tại: 12 đơn High priority, 6 driver giả lập, mean cost
  32,84.

---

## 5. Khoảng Trống Còn Lại Theo Goal Mới

Các mục này không phủ nhận việc pipeline hiện đã chạy được; chúng là phần cần
làm để project “chắc điểm” hơn sau review.

1. **Submission metadata**: nhóm điền thật phụ lục tự chấm và khai báo AI.
2. **Scope control**: dùng `docs/SCOPE_PRIORITIZATION.md` để giữ report/slide
   tập trung vào tầng A; các phần forecast, recommendation, K-Means,
   association, brand/state detail chỉ trình bày ngắn hoặc đưa phụ lục.

---

## 6. Definition Of Done Sau Review Mới

Project coi là hoàn chỉnh theo goal mới khi:

- G1-G9 đều ✅ hoặc có caveat được nhóm chấp nhận rõ ràng.
- `scripts.run_all` pass 18/18 sau mọi sửa.
- Unit tests pass.
- Report PDF và slide PDF build được.
- Streamlit AppTest pass.
- Report/slide/dashboard không overclaim data synthetic.
- Từng câu hỏi khó đều trả lời được:
  - Vì sao không dùng duration/delay làm feature?
  - Sao biết ngưỡng trễ nằm giữa 30 và 35?
  - Sao không chọn Naive Bayes?
  - F2/F-beta được chọn/tối ưu thế nào?
  - Điểm cao có phải do may mắn một split không?
  - Risk Score 0-100 tính ra sao?
  - Assignment dùng dữ liệu thật và giả lập phần nào?
  - Streamlit/Power BI hỗ trợ quyết định gì?

---

## 7. Pha Sửa Tiếp Theo Đề Xuất

1. **Pha 6 — Final verification**: run_all, tests, AppTest, cleanup, commit.

Trước khi làm Pha 2, đọc `docs/SCOPE_PRIORITIZATION.md` để quyết định nội dung
nào đưa vào report/slide chính và nội dung nào chỉ giữ ở notebook/phụ lục.
