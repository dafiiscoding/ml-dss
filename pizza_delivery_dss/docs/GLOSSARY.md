# GLOSSARY — Từ điển thuật ngữ (cho người mới)

> Dành cho thành viên nhóm **mới bắt đầu** và người review. Mọi thuật ngữ kỹ
> thuật trong notebook/báo cáo/code đều giải thích ở đây bằng **tiếng Việt dễ
> hiểu**. Quy ước: giữ tên tiếng Anh (vì tài liệu/khoá học dùng tiếng Anh) +
> giải thích tiếng Việt. Đọc file này trước khi mở notebook.

---

## 1. Bài toán & DSS

- **DSS (Decision Support System) — Hệ hỗ trợ quyết định**: hệ thống biến dữ liệu
  thành *gợi ý hành động* cho người quản lý. Ở đây: dự báo đơn trễ → xếp ưu tiên →
  gợi ý xử lý.
- **Target (biến mục tiêu) `is_delayed`**: cái ta muốn dự báo — đơn có giao trễ
  (True) hay không (False).
- **Feature (đặc trưng)**: thông tin đầu vào để dự báo (quãng đường, traffic, size…).
- **Leakage (rò rỉ dữ liệu)**: lỡ dùng thông tin **chỉ biết sau khi giao** làm
  feature ⇒ mô hình "gian lận", điểm ảo. Ví dụ cấm: `delivery_duration_min`.
- **Baseline (mốc tham chiếu)**: cách đoán ngây thơ để so sánh (đoán "luôn đúng
  giờ" / "luôn trễ"). Mô hình tốt phải vượt baseline.

## 2. Chia dữ liệu & đánh giá

- **Train / Dev / Test**: 3 phần dữ liệu. *Train* để học; *Dev* (validation) để
  chọn mô hình/ngưỡng; *Test* để báo kết quả cuối, **chỉ dùng 1 lần**.
- **Stratified split**: chia sao cho tỷ lệ trễ giữ **giống nhau** ở cả 3 phần.
- **Cross-validation (CV) — kiểm định chéo**: chia train thành k phần, lần lượt
  học k-1 phần và kiểm trên phần còn lại; báo **trung bình ± độ lệch chuẩn** để
  biết mô hình *ổn định* hay *ăn may*.
- **Bootstrap**: lấy mẫu lại có hoàn lại nhiều lần để ước lượng **khoảng tin cậy**.
- **CI (Confidence Interval) — khoảng tin cậy 95%**: dải giá trị hợp lý của một
  con số. CI **rộng** = không chắc (thường do ít dữ liệu). CI chứa 0 = hiệu ứng
  có thể chỉ là nhiễu.

## 3. Chỉ số mô hình (metrics)

> Lớp "trễ" là thiểu số (~21%) nên **không dùng Accuracy một mình**.

- **Accuracy**: tỷ lệ đoán đúng tổng. Dễ bị lớp đa số đánh lừa.
- **Precision (độ chính xác)** = TP/(TP+FP): trong các đơn *bị cảnh báo trễ*, bao
  nhiêu % trễ thật. Cao = ít báo động giả.
- **Recall (độ phủ/độ nhạy)** = TP/(TP+FN): trong các đơn *thực sự trễ*, bắt được
  bao nhiêu %. Cao = ít bỏ sót.
- **F1**: trung bình điều hòa của Precision và Recall (cân bằng hai bên).
- **F2**: như F1 nhưng **ưu tiên Recall gấp đôi**. Chọn F2 vì *bỏ sót đơn trễ
  (FN) tốn hơn* cảnh báo dư (FP).
- **Balanced Accuracy**: trung bình recall của hai lớp — công bằng với lớp thiểu số.
- **MCC (Matthews Correlation Coefficient)**: điểm tổng hợp trong [-1, 1]; 0 =
  đoán mò, 1 = hoàn hảo. Bền với mất cân bằng lớp.
- **ROC-AUC**: khả năng *xếp hạng* (đơn trễ có điểm cao hơn đơn không trễ); 0.5 =
  ngẫu nhiên, 1.0 = hoàn hảo.
- **Confusion matrix (ma trận nhầm lẫn)**: bảng 2×2 đếm TN/FP/FN/TP.
  - **TP** đúng-dương (trễ, đoán trễ), **TN** đúng-âm, **FP** dương-giả (báo trễ
    nhầm), **FN** âm-giả (**bỏ sót đơn trễ** — tốn nhất).
- **PR curve (Precision-Recall)**: đường đánh đổi Precision↔Recall theo ngưỡng;
  dùng để chọn ngưỡng tối ưu F2.

## 4. Các mô hình (classifiers)

- **Logistic Regression**: mô hình tuyến tính ước lượng xác suất; hệ số *diễn
  giải được*.
- **Decision Tree (cây quyết định)**: chuỗi luật if-then chia ngưỡng.
- **Naive Bayes**: dựa xác suất, giả định feature độc lập.
- **k-NN (k láng giềng gần nhất)**: phân loại theo các điểm dữ liệu giống nhất.
- **SVM**: tìm biên phân tách tối ưu (kernel RBF cho biên cong).
- **Random Forest / Ensemble**: trung bình nhiều cây để giảm phương sai.
- **class_weight="balanced"**: tăng trọng số lớp thiểu số để mô hình không bỏ quên.

## 5. EDA, thống kê & clustering

- **EDA (Exploratory Data Analysis)**: phân tích khám phá để hiểu dữ liệu trước
  khi mô hình hoá.
- **Delay rate**: tỷ lệ đơn trễ trong một nhóm.
- **Wilson CI**: cách tính khoảng tin cậy cho tỷ lệ, **đáng tin với nhóm ít đơn**.
- **Chi-square test (kiểm định khi bình phương)**: kiểm tra biến phân loại có
  *liên hệ* với nhãn trễ không; `p < 0.05` = có bằng chứng liên hệ.
- **Cramér's V**: độ *mạnh* của liên hệ trong [0,1]; <0.1 không đáng kể, 0.1–0.3
  yếu, 0.3–0.5 vừa, >0.5 mạnh.
- **K-Means / Silhouette**: gom dữ liệu thành cụm; *silhouette* (−1..1) đo cụm
  tách tốt hay không.
- **Association rule (luật kết hợp)**: "nếu A thì thường B".
  - **Support**: tần suất A&B cùng xảy ra. **Confidence**: P(trễ | A). **Lift**:
    confidence ÷ tỷ lệ trễ chung; **>1** = A làm *tăng* khả năng trễ.
- **Mann-Kendall**: kiểm định xu hướng phi tham số; `tau` là cường độ, `p < 0.05`
  mới kết luận có trend (tăng/giảm).
- **Mutual Information (MI)**: lượng thông tin (bit) một biến cho biết về nhãn;
  0 = độc lập. **MI có điều kiện**: phần thông tin còn lại *sau khi đã biết* biến
  khác (vd quãng đường).
- **Permutation test**: xáo trộn nhãn ngẫu nhiên nhiều lần để dựng "mức nhiễu
  nền"; nếu giá trị quan sát không vượt mức nền ⇒ chỉ là nhiễu, không phải tín
  hiệu thật.

## 6. Forecast & dữ liệu synthetic

- **Time series (chuỗi thời gian)**: dữ liệu theo thời gian (đơn theo tháng/giờ).
- **Seasonal-naive**: dự báo bằng giá trị **cùng kỳ năm trước**.
- **Moving average (trung bình trượt)**: dự báo bằng trung bình vài kỳ gần nhất.
- **MAE / MAPE**: sai số tuyệt đối trung bình / sai số phần trăm; MAPE >40% = yếu.
- **Synthetic data (dữ liệu sinh tự động)**: dữ liệu do chương trình tạo (không
  phải vận hành thật) ⇒ "sạch" bất thường, có công thức ẩn.
- **Deterministic (tất định)**: cột tính chính xác từ cột khác bằng công thức
  (sai số ~0), vd `estimated_duration = 2.4 × distance`.
- **Quantization (lượng tử hoá)**: giá trị bị ép về lưới rời rạc (duration chỉ là
  bội số 5 phút).
- **Ablation**: bỏ một feature rồi đo mức sụt metric để biết đóng góp thật.

## 7. Tầng quyết định & tối ưu

- **Delay Risk Score (0–100)**: điểm rủi ro trễ = trộn có trọng số xác suất mô
  hình + các "áp lực" vận hành (traffic, distance, peak, complexity, weekend).
- **Priority (mức ưu tiên)**: Low / Medium / High theo ngưỡng điểm (35 / 65).
- **Calibration (hiệu chuẩn)**: kiểm tra đơn điểm-cao có *thật sự* trễ nhiều hơn.
- **Assignment / Transportation problem (bài toán phân công)**: gán đơn ↔ tài xế
  sao cho **tổng chi phí nhỏ nhất**.
- **Hungarian algorithm**: thuật toán giải phân công tối ưu; **greedy** = phương
  án "tham lam" đơn giản dùng khi thiếu thư viện.

## 8. Công cụ & hạ tầng

- **scikit-learn (sklearn)**: thư viện ML chính.
- **Pipeline / ColumnTransformer**: gói các bước tiền xử lý + mô hình thành một
  khối, **fit trên train** để tránh leakage.
- **One-Hot Encoding**: biến cột phân loại thành các cột 0/1.
- **StandardScaler**: chuẩn hoá số về cùng thang (trung bình 0, độ lệch 1).
- **joblib**: lưu/đọc mô hình đã huấn luyện (`.joblib`).
- **Streamlit**: dựng dashboard web. **Power BI pack**: bộ CSV/DAX để dựng báo
  cáo trong Power BI Desktop.
- **Fallback thuần Python**: bản thay thế numpy cho vài mô hình sklearn, dùng khi
  máy chặn DLL native (WDAC); pipeline vẫn chạy đủ.
