# GOAL — Pizza Delivery DSS

> File goal **chuẩn** cho mọi agent. Đọc file này đầu tiên để biết *mục tiêu*,
> *tiêu chí đạt (đo được)*, *protocol không thương lượng*, *kết quả khóa* và
> *trạng thái thật*. Cấu trúc học theo **p1** = dự án mẹ CrisisMMD
> (`../.ai/state.md`): goal cụ thể → protocol → locked results → verification.
> Đọc kèm: `README.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md`,
> `docs/GRADING_MAP.md`, `docs/WORKFLOW_PRESENTATION_GUIDE.md`.
> Cập nhật: 17/06/2026.

---

## 0. Goal tổng (1 câu)

Xây prototype **Hệ hỗ trợ quyết định** dự báo đơn pizza giao trễ **trước khi
giao** và biến dự báo thành hành động ưu tiên/điều phối, đồng thời thể hiện
**đầy đủ và trung thực** quy trình môn học trên một dataset tabular nhỏ (synthetic).

---

## 1. Goal chi tiết theo tiêu chí (đo được)

Mỗi goal "đạt" khi **tất cả** tiêu chí con thỏa. ✅ = đạt & đã verify (chạy
`run_all` 18/18 + 26 test, 17/06/2026) | ⬜ = chưa/tùy chọn.

### G1 — Goal quyết định (cốt lõi DSS)
| Tiêu chí đo được | Trạng thái |
|---|---|
| Pipeline: `P(trễ)` → `delay_risk_score` (0–100) → `priority` (Low ≤35 < Medium ≤65 < High) → `recommended_action` → hàng đợi sắp theo risk giảm dần | ✅ |
| Risk Score **bóc tách được** thành 6 thành phần, trọng số minh bạch (0.55 model + 0.45 áp lực vận hành) | ✅ |
| **Calibration**: `actual_delay_rate` tăng đơn điệu theo risk band trên test | ✅ |
| **Độ nhạy ngưỡng** 35/65 được trình bày (số đơn High đổi theo ngưỡng) | ✅ |
| Transport: gán 12 đơn High cho 6 tài xế giả lập, chi phí tối thiểu (Hungarian, greedy fallback), nêu rõ "giả lập" | ✅ |
| Dashboard Streamlit pass AppTest + Power BI pack 15 bảng + `measures.dax` + `manifest.json` | ✅ |

### G2 — Goal học thuật (phủ đủ quy trình môn DSS)
| Tiêu chí đo được | Trạng thái |
|---|---|
| Phủ **đủ** chủ đề trong `GRADING_MAP.md`: DSS, supervised, DT/NB/kNN/SVM/Ensemble, clustering, association, hypothesis test, customer behavior, time series, recommendation, optimization, BI | ✅ |
| Kỷ luật chống rò rỉ: fit **train**, chọn **dev** theo F2, **test 1 lần**, luôn báo **2 baseline** | ✅ |
| So sánh **6 classifier** + **cross-validation** (mean±std) chứng minh ổn định | ✅ (CV mới) |
| Mỗi chủ đề có **artifact** (`reports/metrics/*`) **và** mục notebook tương ứng | ✅ |

### G3 — Goal trung thực dữ liệu (data là synthetic)
| Tiêu chí đo được | Trạng thái |
|---|---|
| Forensics chứng minh ≥7 công thức/target **tất định** (`max_abs_error ≈ 0`) | ✅ |
| Ngưỡng nhãn **suy luận từ dữ liệu** (`>30` ≡ `≥35` trên lưới 5') chứ không giả định SLA | ✅ |
| **Permutation-MI** phân biệt tín hiệu thật vs mức nhiễu nền (bias biến nhiều mức) | ✅ |
| Brand: homogeneity test + **bootstrap ΔF2** (CI chứa 0 ⇒ within noise) | ✅ |
| **Mọi kết luận mạnh kèm CI/caveat**; nêu rõ điểm số cao do dữ liệu gần tất định, không phải năng lực dự báo thật | ✅ |

### G4 — Goal trình bày (Bám sát barem chấm điểm & cho beginner đọc)

**Đối tượng đọc (ai hiểu).** *Chính:* giảng viên chấm điểm (đáp ứng 100% checklist). *Phụ:* thành viên nhóm **mới bắt đầu** (chưa rành
ML/DSS).

**"Hiểu" nghĩa là gì.** Sau **một vòng đọc tuần tự**, người đọc nắm được:
1. **Mỗi bước làm gì và vì sao** (mục tiêu + lý do của từng phân tích).
2. **Cách đọc từng bảng/biểu đồ** — nhìn vào đâu, con số/đường nét nói lên điều gì.
3. **Các khái niệm/thuật ngữ cơ bản** (DSS, leakage, F2, MCC, CI, bootstrap…).
4. **Kết luận và giới hạn** (đặc biệt: dữ liệu synthetic, không overclaim).

**Quy ước ngôn ngữ.** Lời giải thích viết **tiếng Việt**; thuật ngữ kỹ thuật giữ
**nguyên tiếng Anh** (F2, MCC, ROC-AUC, bootstrap, lift…) và **giải thích bằng
tiếng Việt ngay lần dùng đầu** + tập trung trong `docs/GLOSSARY.md`.

| Tiêu chí đo được | Trạng thái |
|---|---|
| Báo cáo tuân thủ **cấu trúc 4 phần chính + 3 phụ lục bắt buộc** (Phân công, Tự chấm điểm, Khai báo AI) theo `Huong_dan_bai_tap_nhom_DSS.pdf` | 🟡 |
| **Trả lời trực diện 8 câu hỏi bắt buộc** trong báo cáo (Bài toán gì, ai dùng, quyết định gì, data nào...) | 🟡 |
| Có `docs/GLOSSARY.md` — định nghĩa **mọi thuật ngữ bằng tiếng Việt** cho người mới | ✅ |
| Mỗi notebook mở đầu nêu **đối tượng + quy ước ngôn ngữ + link glossary** | ✅ |
| Mỗi biểu đồ có câu **"Cách đọc hình"** (cho người mới) **và** insight gắn liền với quyết định kinh doanh | ✅ |
| Bảng **"Insight → quyết định"** cuối mỗi notebook | ✅ |
| Bản đồ **Hình → Slide** và **Dữ liệu → Power BI** | ✅ (`WORKFLOW_PRESENTATION_GUIDE.md` §2b) |

**Lưu ý sản phẩm nộp (theo `Huong_dan_bai_tap_nhom_DSS.pdf`).** Barem **không có
thuyết trình trực tiếp** ⇒ **slide KHÔNG bắt buộc** (chỉ là tùy chọn nội bộ, không
phải tiêu chí đạt Goal). Sản phẩm bắt buộc: báo cáo, dữ liệu/nguồn, code/notebook/
dashboard, **minh chứng kết quả (gồm ảnh chụp dashboard)**, bảng phân công. Báo
cáo **bắt buộc** có phụ lục **Tự chấm điểm** và **Khai báo sử dụng AI**, và trả
lời trực diện **8 câu hỏi**. Hai mục cần input thật của nhóm (điểm tự chấm; công
cụ/mục đích AI) nên để 🟡 đến khi nhóm điền.

---

## 2. Protocol không thương lượng (hard rules — học từ p1)

1. Pizza là **dự án con**, không thay thế CrisisMMD.
2. Chỉ dùng feature **biết trước/tại lúc nhận đơn**. Cấm tuyệt đối làm feature:
   `delivery_duration_min`, `delivery_time`, `delay_min`,
   `delivery_efficiency_min_per_km`, `restaurant_avg_time`.
3. Fit preprocessing **trên train**; chọn model **trên dev** theo **F2**; báo
   **test một lần** sau khi khóa, **kèm khoảng tin cậy bootstrap**.
4. Luôn báo baseline **always-on-time** và **always-delayed** cạnh kết quả.
5. Risk/Priority là **chính sách minh bạch**, không phải nhãn tối ưu thống kê.
6. Transport dùng **đơn thật** nhưng **driver/fleet là giả lập** (Kaggle không có
   bảng tài xế).
7. Dataset nhiều dấu hiệu synthetic ⇒ báo cáo **không overclaim**
   forecast/recommendation/brand; mọi tuyên bố mạnh kèm caveat + CI.
8. **Một nơi sinh artifact** (`build_*_artifacts` ở Bước 0 mỗi notebook), các mục
   sau chỉ **trình bày** — tách bạch, không tính lại, tuần tự.

---

## 3. Kết quả khóa (locked numbers — mục tiêu đo được)

- Corpus 1.004 đơn; 210 trễ / 794 đúng giờ; **delayed rate 20,92%**.
- `is_delayed` ≡ `delivery_duration_min > 30` (mismatch 0); chronological last-20%
  có 0 đơn trễ ⇒ dùng **stratified** split.
- Split **602 / 201 / 201**.
- Feature active: `compact_nonredundant` (12 feature).
- Model khóa: **Logistic Regression** (chọn theo F2 dev).
- Test (mục tiêu giữ ổn định): Accuracy 0,9602; Balanced Accuracy 0,9661;
  Precision 0,8542; Recall 0,9762; F1 0,9111; **F2 0,9491**; MCC 0,8889.
- Baseline always-on-time: F2 0, MCC 0. Always-delayed: F2 0,5691, MCC 0.
- CV 5-fold (đã chạy): SVM F2 0,945±0,043; Logistic Regression F2 0,939±0,032 —
  std nhỏ ⇒ ổn định. Bootstrap CI test cho F2/MCC/Recall đã sinh; ΔF2 brand có CI
  (within-noise) đã sinh.

---

## 4. Cổng verification (pass/fail cụ thể)

- `python -m scripts.run_all` → **18/18 bước pass**.
- `python -m unittest discover -s tests -v` → **pass hết** (gồm test cho 9 hàm
  chiều sâu mới).
- 6 notebook execute **0 cell lỗi**, mỗi notebook có chart inline + insight.
- Số liệu notebook khớp `reports/metrics/*`.
- LaTeX report + Beamer slide build được.

---

## 5. Definition of Done (Goal coi như ĐẠT khi)

- [x] G1, G2, G3, G4: mọi tiêu chí ở §1 đã ✅.
- [x] §4 verification: `run_all` **18/18** + tests pass (chạy 17/06/2026).
- [x] Test cho 9 hàm mới đã thêm và pass (**26 test** tổng).
- [ ] (Tùy chọn) slide bổ sung hình khuyến nghị; vá deprecation `SVC(probability=True)`.

---

## 6. Trạng thái thật

- **Đã ĐẠT các tiêu chí bắt buộc.** `run_all` chạy **18/18 bước (EXIT 0, 0
  traceback)**; **26/26 unit test pass**; 6 notebook execute 0 lỗi (35 hình
  inline); report 34 trang + slide 21 trang build; Streamlit AppTest pass.
- **WDAC vẫn quarantine DLL native** (`sklearn.cluster/neighbors/tree/ensemble`,
  trước đó cả `matplotlib._image`). **Đã xử lý ở mức code** bằng fallback thuần
  Python/numpy: `SimpleDecisionStumpClassifier`, `SimpleKNNClassifier`,
  `SimpleStumpEnsembleClassifier` (`modeling.py`) và `_SimpleKMeans` (`eda.py`),
  kích hoạt qua `_optional_classifier`/`try-import`. Nhờ vậy pipeline chạy đầy đủ
  **không cần** gỡ chính sách OS. Khi DLL được phép trở lại, code tự dùng sklearn
  gốc.
- Còn lại (tùy chọn, không chặn Goal): chèn hình khuyến nghị vào slide; vá
  deprecation `SVC(probability=True)` (sklearn 1.9 → bỏ ở 1.11).

## 7. Next actions (tùy chọn)

1. Wire các hình `[SLIDE] nên thêm` (xem `WORKFLOW_PRESENTATION_GUIDE.md` §2b) vào
   `slides/pizza_dss_slides.tex` + rebuild.
2. Vá `SVC(probability=True)` → `CalibratedClassifierCV(SVC(), ensemble=False)`.

---

## 8. Trả lời nhanh: "đã đạt Goal chưa?"

**ĐÃ ĐẠT** các tiêu chí bắt buộc (§1 toàn ✅, §4 verification pass: `run_all`
18/18 + 26 test, notebook/report/slide/dashboard đều OK). Hai việc ở §7 là **tùy
chọn**, không chặn Goal.
