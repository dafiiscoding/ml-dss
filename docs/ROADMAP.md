# LỘ TRÌNH CHUẨN HÓA & HOÀN THIỆN DỰ ÁN DSS

> **TÀI LIỆU LỊCH SỬ, KHÔNG DÙNG ĐỂ KẾT LUẬN TIẾN ĐỘ.**
> Kế hoạch này được viết trước khi dữ liệu/ảnh hoàn tất. Trạng thái, protocol và
> số liệu đã kiểm toán nằm tại [`docs/PROGRESS.md`](PROGRESS.md).

> **Đề tài:** Hệ hỗ trợ quyết định ứng phó thảm họa dùng văn bản & hình ảnh mạng xã hội (Multimodal Disaster Response DSS)
> **Dataset:** CrisisMMD v2.0 (thật) — QCRI/CrisisNLP
> **Mục tiêu:** Đưa dự án từ "chạy được trên dữ liệu giả" → "bám sát kiến thức trọng tâm môn học, có minh chứng phân tích thật, đạt điểm cao theo rubric".

---

## 0. Bản đồ: Kiến thức môn học ↔ Thành phần dự án

Bảng này là "kim chỉ nam" — mỗi thành phần phải gắn được vào ít nhất một slide trọng tâm. Sẽ đưa vào báo cáo để chứng minh đúng trọng tâm môn.

| Kiến thức môn học (slide) | Thành phần trong dự án |
| :--- | :--- |
| DSS0–1: Tổng quan DSS, Tác tử thông minh | Tầng quyết định (Prescriptive Layer): Risk Score, Routing, Manual Review |
| DSS05: Machine Learning (tổng quan) | Quy trình train/val/test, cross-validation, đánh giá, chống overfitting |
| DSS06: Decision Tree | Mô hình so sánh #1 |
| DSS07: Rule Induction, k-NN, Naive Bayes | Mô hình so sánh #2, #3, #4 |
| DSS08: ANN, SVM, Ensemble | SVM + Random Forest/Gradient Boosting (mô hình so sánh #5, #6) |
| DSS09a: Association Analysis | Apriori/FP-Growth trên đồng xuất hiện keyword/hashtag (EDA nâng cao) |
| DSS09c: Clustering | K-Means trên TF-IDF/embedding → phát hiện cụm chủ đề (EDA nâng cao) |
| DSS10–10b: Text & Web Mining | Làm sạch text, tokenize, stopwords, TF-IDF, n-grams, keyword analysis |
| DSS11: Social Media Mining | Phân tích tweet thảm họa (chủ đề chính của dataset) |
| DSS12/12b: DW & OLAP, PowerBI | Vai trò BI dashboard (Streamlit) — theo dõi KPI cứu trợ |
| CLIP (ngoài chương trình) | **Phụ trợ:** trích đặc trưng ảnh pretrained → minh chứng pipeline đa phương thức |

---

## Các giai đoạn (theo dõi tiến độ ở đây)

### 🟦 GĐ 0 — Chuẩn bị & tải dữ liệu thật  `[ ]`
**Rubric:** nền tảng cho mục 2 (Dữ liệu).
- [ ] Kiểm tra môi trường: Python, có cài được `torch`/`transformers` cho CLIP không (nếu không → fallback đặc trưng ảnh cổ điển).
- [ ] Tải **CrisisMMD v2.0**: annotations (file TSV, nhẹ) + ảnh (tarball ~2.5GB).
  - Nguồn: https://crisisnlp.qcri.org/crisismmd → "CrisisMMD_v2.0.tar.gz" + "crisismmd_datasplit_all".
  - ⚠️ Ảnh rất nặng → quyết định **subset** (vd 2–3 sự kiện, hoặc N≈2000 mẫu cân bằng) để CLIP xử lý khả thi.
- [ ] Đặt dữ liệu vào `data/raw/` theo đúng cấu trúc gốc.
**Đầu ra:** dữ liệu thật nằm trong repo (hoặc mô tả cách tải nếu quá nặng để commit).

### 🟦 GĐ 1 — Dữ liệu & Tiền xử lý  `[ ]`
**Rubric:** Mục 2 — Dữ liệu & tiền xử lý (15đ). **Slide:** DSS10 Text Mining.
- [ ] Viết lại `data_loader.py`: đọc annotation thật CrisisMMD (gộp task informative + humanitarian), map cột về schema chuẩn, ghép đường dẫn ảnh.
- [ ] Kiểm tra chất lượng: thiếu/trùng (`tweet_id`), ngoại lệ, kiểu dữ liệu, **phân bố mất cân bằng lớp** (báo cáo rõ).
- [ ] `text_preprocessing.py`: clean (regex), tokenize, stopwords, (lemmatization), TF-IDF + n-grams.
- [ ] `image_preprocessing.py`: CLIP embedding cho subset, **cache ra `.npy`** (chỉ chạy 1 lần).
- [ ] Lưu splits cố định vào `data/processed/`.
**Đầu ra:** `data/processed/` + embedding cache; bảng thống kê tiền xử lý.

### 🟦 GĐ 2 — EDA & Phát hiện Insight  `[ ]`  → **notebook `notebooks/02_eda.ipynb`**
**Rubric:** Mục 3 — Phân tích & insight (20đ, điểm cao nhất). **Slide:** DSS09a, DSS09c, DSS10, DSS11.
- [ ] Thống kê mô tả: phân bố lớp, theo `event_name`, độ dài tweet, tỉ lệ informative.
- [ ] Trực quan hóa: distribution charts, top keywords/wordcloud theo lớp, độ dài theo lớp.
- [ ] **Clustering (K-Means)** trên TF-IDF/CLIP → phát hiện cụm chủ đề tự nhiên, đối chiếu với nhãn thật.
- [ ] **Association rules (Apriori)** trên đồng xuất hiện keyword/hashtag → luật "xuất hiện X thường kèm Y".
- [ ] Mỗi insight **gắn với một quyết định** (vd: lớp `injured_or_dead` hiếm & mất cân bằng → phải tối ưu Recall).
**Đầu ra:** notebook EDA + hình trong `reports/figures/`.

### 🟦 GĐ 3 — Mô hình hóa & So sánh  `[ ]`  → **notebook `notebooks/03_modeling.ipynb`**
**Rubric:** Mục 4 — Mô hình/hệ thống (20đ). **Slide:** DSS05–08.
- [ ] Train nhiều classifier trên **text** cho 2 nhiệm vụ (informative nhị phân + humanitarian đa lớp):
      Logistic Regression, Decision Tree, Naive Bayes, k-NN, SVM, Random Forest (±Gradient Boosting).
- [ ] Cross-validation + xử lý mất cân bằng (`class_weight`/resampling).
- [ ] Nhánh **ảnh**: CLIP embedding → classifier (LogReg/SVM) — phụ trợ.
- [ ] **Late fusion** text + ảnh.
- [ ] Bảng so sánh: Accuracy, Precision, Recall, **F1, F2 (ưu tiên Recall)**, macro-F1, confusion matrix → chọn mô hình tốt nhất **có lý do**.
- [ ] Lưu best models vào `models/`.
**Đầu ra:** notebook modeling + bảng so sánh + model files.

### 🟦 GĐ 4 — Tầng hỗ trợ quyết định (DSS Layer)  `[ ]`
**Rubric:** Mục 4 (phần hệ thống quyết định). **Slide:** DSS0–1.
- [ ] Giữ & hiệu chỉnh `risk_scoring.py`, `decision_rules.py` (phần này vốn đã tốt — chỉ tinh chỉnh trọng số & gắn vào model mới).
- [ ] Cập nhật `fusion.py` dùng best models.
- [ ] `test_integration.py`: kiểm thử E2E các kịch bản (khẩn cấp / mâu thuẫn / spam).
**Đầu ra:** DSS layer chạy thông với model thật + test pass.

### 🟦 GĐ 5 — Dashboard (Presentation)  `[ ]`
**Rubric:** Mục 4 + Mục 6 (minh chứng). **Slide:** DSS12 (vai trò BI).
- [ ] Cập nhật 5 trang dùng **dữ liệu & model thật**.
- [ ] Trang EDA: bổ sung kết quả clustering + association.
- [ ] Trang Model Evaluation: hiển thị **so sánh nhiều mô hình** (không còn 1 mô hình).
**Đầu ra:** app Streamlit chạy mượt trên dữ liệu thật.

### 🟦 GĐ 6 — Minh chứng & Tài liệu  `[ ]`
**Rubric:** Mục 6 (10đ) + Mục 7 (5đ).
- [ ] Chụp screenshot từng trang dashboard → `reports/figures/screenshots/`.
- [ ] Cập nhật `README.md` (hướng dẫn chạy đúng pipeline mới).
- [ ] Cập nhật `contribution_log.md`, `ai_usage_declaration.md` (khai báo đúng công cụ AI thực tế).
- [ ] Hoàn thiện **bảng tự chấm điểm** (số liệu thật, không tự cho điểm tối đa vô căn cứ).
**Đầu ra:** hồ sơ minh chứng đầy đủ.

### 🟦 GĐ 7 — Báo cáo LaTeX → PDF  `[ ]`
**Rubric:** Mục 5 (15đ) + Mục 6 (10đ).
- [ ] Setup LaTeX tiếng Việt (XeLaTeX + `fontspec`, hoặc `babel` vietnamese) trong `reports/latex/`.
- [ ] Viết đúng cấu trúc **10 mục** của hướng dẫn + trả lời **8 câu hỏi bắt buộc**.
- [ ] Nhúng hình từ `reports/figures/`, bảng kết quả **thật** (số khớp 100% với notebook/dashboard).
- [ ] Compile ra `final_report.pdf`.
**Đầu ra:** `final_report.pdf` hoàn chỉnh.

---

## Thứ tự ưu tiên & phụ thuộc
```
GĐ0 (tải data) ──▶ GĐ1 (tiền xử lý) ──▶ GĐ2 (EDA) ──▶ GĐ3 (model) ──▶ GĐ4 (DSS) ──▶ GĐ5 (dashboard)
                                                                                          │
                                          GĐ6 (minh chứng) ◀───────────────────────────────┘
                                                   │
                                                   ▼
                                          GĐ7 (báo cáo LaTeX → PDF, làm cuối)
```
**Trạng thái hiện tại:** không còn blocker dữ liệu. Xem `docs/PROGRESS.md`.

## Runbook tái tạo
Ảnh và CLIP cache đã hoàn tất. Để tái tạo toàn bộ artefact, chạy:
```
python -m scripts.run_all
```
Lệnh này audit dữ liệu, tái dùng CLIP cache, train/so sánh model text và ảnh,
đánh giá fusion/DSS, rebuild notebook và chạy integration test.

## Ghi chú rủi ro
- Không xóa CLIP cache nếu dữ liệu và thứ tự hàng không đổi.
- EDA phục vụ mô hình phải train-only; dev/test chỉ dùng đánh giá.
- Báo cáo: tuyệt đối **không để số liệu mâu thuẫn** giữa code và báo cáo (lỗi cũ: "Recall 1.00" vô căn cứ).
