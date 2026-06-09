# KẾ HOẠCH TỔNG THỂ & KHUNG BÁO CÁO (BLUEPRINT)

> **TÀI LIỆU THIẾT KẾ LỊCH SỬ.** Một số giả định ban đầu bên dưới đã được thay
> thế sau audit. Trạng thái, protocol và số liệu chính thức nằm trong
> [`docs/PROGRESS.md`](PROGRESS.md).

> Tài liệu chốt **quy trình thật**, **độ phủ yêu cầu**, **độ phủ kiến thức môn học** và **khung báo cáo** trước khi viết. Dùng kèm `docs/ROADMAP.md` (lộ trình 7 GĐ thực thi).

---

## A. QUY TRÌNH THẬT CỦA DỰ ÁN (end-to-end pipeline)

Input: **1 tweet = văn bản + (các) ảnh đính kèm**. Output: **quyết định điều phối cứu trợ**.

```
[1] NẠP DỮ LIỆU (data_loader)
    CrisisMMD v2.0 → schema thống nhất:
    tweet_text, image, event_name, label_top(8 lớp), label(informative chính thức),
    label_image_cat, multimodal_agree(Positive/Negative)
        │
[2] TIỀN XỬ LÝ
    • Text: regex clean → tokenize → stopwords → TF-IDF (1–2 gram)        [Text Mining]
    • Ảnh : load → resize 224 → CLIP ViT-B/32 → embedding 512-d (cache .npy)
            (fallback đặc trưng màu nếu không có torch)                   [CLIP = phụ trợ]
        │
[3] EDA (phân tích offline — KHÔNG nằm trong luồng phục vụ)
    phân bố/mất cân bằng • text mining • K-Means • Apriori • mâu thuẫn text-ảnh
        │
[4] TẦNG DỰ BÁO (Predictive) — mỗi phương thức, 2 nhiệm vụ
    • Text: classifier "informative" + classifier "humanitarian 8 lớp"
            → SO SÁNH DT / NB / kNN / SVM / RF / LogReg, chọn tốt nhất    [Classification]
    • Ảnh : classifier trên CLIP embedding (informative + humanitarian)
        │
[5] LATE FUSION đa phương thức
    P_fusion = w_text·P_text + w_image·P_image; trọng số/threshold chọn trên dev
    (informative 0.70/0.30; humanitarian 0.55/0.45)
        │
[6] TẦNG QUYẾT ĐỊNH (Prescriptive DSS — rule engine)        [DSS / Tác tử]
    • Risk Score (0–100) = f(fusion_prob, trọng số lớp, mật độ từ khóa, độ tin cậy)
    • Priority: Low / Medium / High
    • Routing: 8 lớp → Emergency / Relief / Infrastructure / Coordination Team
    • Manual-Review override: conflict lớn → đẩy Supervisor (chống tin giả)
        │
[7] TRÌNH DIỄN (Presentation) — Streamlit dashboard          [BI / DW-OLAP vai trò]
    Overview • EDA • Model Evaluation • Single-Case Demo • Priority Queue
```

**Điểm cốt lõi cần nhấn trong báo cáo:** dữ liệu → phân tích → mô hình → **quyết định cụ thể**. Đây là tiêu chí số 1 của đề (mục "chứng minh dữ liệu hỗ trợ một quyết định").

---

## B. SOÁT CHUẨN YÊU CẦU (rubric 100đ)

| Tiêu chí (điểm) | Dự án đáp ứng ở đâu | Trạng thái |
| :--- | :--- | :---: |
| 1. Hiểu vấn đề & bài toán quyết định (15) | CH1: bối cảnh, 5 đối tượng, 5 quyết định cần hỗ trợ | ✅ Đã có nội dung |
| 2. Dữ liệu & tiền xử lý (15) | CH3: CrisisMMD thật, audit nhãn/hash, clean text + CLIP | ✅ |
| 3. Phân tích & insight (20) | CH4: train-only EDA, K-Means, Apriori, ảnh và conflict | ✅ |
| 4. Mô hình/hệ thống (20) | CH5: **so sánh 6 classifier** + fusion + DSS rule engine | ✅ Prototype |
| 5. Diễn giải & khuyến nghị (15) | CH6: 8 câu hỏi + khuyến nghị và giới hạn | ✅ |
| 6. Chất lượng báo cáo & minh chứng (10) | XeLaTeX HUST + screenshots + notebook + figures | ✅ |
| 7. Tổ chức nhóm & đóng góp (5) | Phụ lục + AI declaration | 🟡 Thiếu 3 danh tính/phân công thật |

**8 câu hỏi bắt buộc** → trả lời tập trung trong **CH6** (đã có nháp ở `final_report_draft.md`, sẽ cập nhật số liệu thật).

**Kết luận hiện tại:** kỹ thuật và báo cáo đã có; hồ sơ nhóm chỉ hoàn tất sau khi
bổ sung ba danh tính và phân công thật.

---

## C. ĐỘ PHỦ KIẾN THỨC MÔN HỌC (trả lời: "đã áp dụng đa phần kiến thức chưa?")

**CÓ — sau chuẩn hóa, dự án phủ ~8/13 nhóm slide trọng tâm:**

| Slide môn học | Áp dụng trong dự án | Mức độ |
| :--- | :--- | :---: |
| DSS0–1: Tổng quan DSS, Tác tử thông minh | Tầng quyết định (rule engine, override) | ⭐⭐⭐ Lõi |
| DSS05: Machine Learning | Quy trình train/val/test, CV, metrics, overfitting | ⭐⭐⭐ |
| DSS06: Decision Tree | 1 trong 6 mô hình so sánh | ⭐⭐⭐ |
| DSS07: k-NN, Naive Bayes, Rule Induction | 3 mô hình so sánh | ⭐⭐⭐ |
| DSS08: SVM, Ensemble | SVM + Random Forest | ⭐⭐⭐ |
| DSS09a: Association Analysis | Apriori trên đồng xuất hiện keyword/hashtag | ⭐⭐ |
| DSS09c: Clustering | K-Means phát hiện cụm chủ đề | ⭐⭐ |
| DSS10/10b: Text & Web Mining | Clean, tokenize, TF-IDF, n-grams, keyword | ⭐⭐⭐ Lõi |
| DSS11: Social Media Mining | Phân tích tweet/hashtag thảm họa | ⭐⭐⭐ Lõi |
| DSS12/12b: DW, OLAP, PowerBI | Vai trò BI: Streamlit dashboard theo dõi KPI | ⭐⭐ |
| DSS13: Recommendation Engines | (Routing đội ≈ gợi ý hành động — liên hệ nhẹ) | ⭐ Tùy chọn |

**Thành thật về khoảng trống:**
- **DW/OLAP** chỉ thể hiện gián tiếp qua dashboard, *không* xây kho dữ liệu star-schema thật.
- **Recommendation engine** không dùng đúng nghĩa (chỉ liên hệ ẩn dụ qua routing) → **sẽ không cố nhồi** để tránh khiên cưỡng.
- **CLIP** nằm *ngoài* chương trình → khai báo rõ là *công cụ trích đặc trưng pretrained phụ trợ*, không phải kiến thức môn được "khoe".

→ Độ phủ này **rộng và chính đáng**, vượt mức một dự án chỉ-1-thuật-toán.

---

## D. KHUNG BÁO CÁO (LaTeX, style HUST, ~30–40 trang)

Mượn khung báo cáo ML đã kiểm chứng (`loan-default`) và style HUST. Bản hiện
tại compile bằng **XeLaTeX + `fontspec`** để tiếng Việt và Times New Roman ổn
định; mỗi chương nằm trong một file riêng.

```
Bìa HUST  →  Lời cảm ơn  →  Tóm tắt (abstract)  →  Bảng thuật ngữ  →  Mục lục
```

**CHƯƠNG 1 — GIỚI THIỆU BÀI TOÁN**  *(rubric §1,2 · 15đ)*
- 1.1 Bối cảnh & tính cấp thiết (quá tải thông tin MXH khi thảm họa)
- 1.2 Đối tượng sử dụng (5 nhóm) & các quyết định cần hỗ trợ (5 câu hỏi)
- 1.3 Mục tiêu (SMART) & phạm vi
- 1.4 Đóng góp chính của dự án

**CHƯƠNG 2 — CƠ SỞ LÝ THUYẾT & KIẾN THỨC ÁP DỤNG**  *(chương "phô kiến thức & lập luận")*
- 2.1 Kiến trúc DSS 3 tầng (Predictive → Prescriptive → Presentation)
- 2.2 Text Mining & Social Media Mining (TF-IDF, n-grams, làm sạch tweet)
- 2.3 Các thuật toán phân lớp: DT, NB, k-NN, SVM, Ensemble, LogReg — định nghĩa ngắn + **vì sao chọn từng cái**
- 2.4 Học không giám sát: K-Means (clustering) & Apriori (association)
- 2.5 Đánh giá bài toán mất cân bằng: Precision/Recall/F1/**F2**, confusion matrix — **vì sao ưu tiên Recall/F2**
- 2.6 Late fusion đa phương thức & CLIP (nêu rõ vai trò phụ trợ)

**CHƯƠNG 3 — DỮ LIỆU & TIỀN XỬ LÝ**  *(rubric §3,4 · 15đ)*
- 3.1 Nguồn dữ liệu CrisisMMD v2.0, schema, ý nghĩa biến, phạm vi/hạn chế
- 3.2 Kiểm tra chất lượng: thiếu/trùng/hash xuyên split/kiểu dữ liệu/**mất cân bằng khoảng 219:1 trên train**
- 3.3 Tiền xử lý văn bản (regex + TF-IDF) & hình ảnh (CLIP embedding)
- 3.4 Thiết kế đặc trưng & lý do

**CHƯƠNG 4 — PHÂN TÍCH KHÁM PHÁ & INSIGHT**  *(rubric §5 · 20đ — điểm cao nhất)*
- 4.1 Thống kê mô tả & phân bố nhãn/sự kiện
- 4.2 Text mining: độ dài, top keyword, wordcloud
- 4.3 Clustering K-Means: cụm chủ đề
- 4.4 Association Apriori: luật đồng xuất hiện
- 4.5 **Phân tích mâu thuẫn text–ảnh** (mức nhãn + mức điểm-ảnh t-SNE)
- 4.6 Bảng tổng kết **5 Insight → Quyết định**

**CHƯƠNG 5 — MÔ HÌNH & HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH**  *(rubric §4 · 20đ)*
- 5.1 Thiết kế thử nghiệm (split, CV, xử lý mất cân bằng)
- 5.2 **So sánh 6 mô hình** text (bảng metrics + confusion matrix) → chọn best
- 5.3 Mô hình ảnh + Late Fusion (so với đơn phương thức)
- 5.4 Tầng quyết định DSS: công thức Risk Score, Priority, Routing, Manual-Review + ví dụ

**CHƯƠNG 6 — KẾT QUẢ, KHUYẾN NGHỊ & TRẢ LỜI CÂU HỎI**  *(rubric §7,8 · 15đ)*
- 6.1 Kết quả chính & ví dụ đầu ra
- 6.2 Khuyến nghị hành động cứu trợ (có căn cứ dữ liệu)
- 6.3 **Trả lời 8 câu hỏi bắt buộc**
- 6.4 Hạn chế, rủi ro, điều kiện áp dụng

**CHƯƠNG 7 — SẢN PHẨM: DASHBOARD**  *(minh chứng · rubric §6)*
- 7.1 5 trang dashboard + screenshots
- 7.2 Hướng dẫn chạy & công nghệ

**CHƯƠNG 8 — KẾT LUẬN & HƯỚNG PHÁT TRIỂN**  *(rubric §9)*

**PHỤ LỤC:** A. Bảng phân công công việc · B. Khai báo sử dụng AI · C. Bảng tự chấm điểm · D. Siêu tham số · E. Link code/notebook
**TÀI LIỆU THAM KHẢO:** CrisisMMD papers (ICWSM'18, ISCRAM'20), scikit-learn, CLIP...

→ Khung này **phủ trọn 10 mục bắt buộc** của đề, dồn phần "khoe kiến thức" vào CH2 và phần "lập luận xử lý vấn đề" rải khắp CH3–5.

---

## E. (Tùy chọn) SLIDE BÁO CÁO — Beamer theme HUST
Nếu cần thuyết trình: tái dùng `toiuu/slide/beamerthemeHUST.sty`, ~12–15 slide bám CH1→CH8.

---

## Thứ tự thực thi đề xuất
1. Bổ sung ba tên/MSSV còn thiếu.
2. Xác nhận phân công và tỷ lệ đóng góp thật.
3. Compile lại PDF và nộp cùng notebook, README, screenshot.
```
```
