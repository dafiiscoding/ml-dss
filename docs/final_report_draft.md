# BÁO CÁO CUỐI KỲ MÔN HỆ HỖ TRỢ QUYẾT ĐỊNH (DSS)

> **Bản nháp lưu trữ:** File này được viết trước khi chuẩn hóa dữ liệu và
> protocol đánh giá. Báo cáo hiện hành là `reports/latex/main.pdf`; không dùng
> số liệu trong file này để nộp bài.

## Đề tài: Hệ hỗ trợ quyết định ứng phó thảm họa sử dụng văn bản và hình ảnh từ mạng xã hội
**Tên ngắn:** Multimodal Disaster Response DSS

---

## I. GIỚI THIỆU BÀI TOÁN & BỐI CẢNH THỰC TẾ

### 1. Bối cảnh
Khi thảm họa tự nhiên xảy ra (lũ lụt, bão, động đất, cháy rừng), mạng xã hội (Twitter, Facebook) lập tức trở thành kênh thông tin khổng lồ và cập nhật nhanh chóng nhất từ hiện trường. Người dân thường đăng tải thông tin trực tiếp về các lời kêu cứu, tình hình thương vong, hạ tầng bị tàn phá (cầu sập, đường tắc) hoặc các hoạt động tình nguyện, quyên góp nhu yếu phẩm.

### 2. Lý do chọn đề tài
Mặc dù thông tin trên mạng xã hội rất phong phú, nhưng chúng tồn tại dưới dạng **dữ liệu phi cấu trúc (unstructured)** và chứa lượng nhiễu cực kỳ lớn (meme, cảm xúc cá nhân, tin tức chung chung, tin giả). Các trung tâm điều phối cứu nạn và đội phản ứng khẩn cấp luôn rơi vào tình trạng **quá tải thông tin** và không thể đọc thủ công hàng vạn bài đăng mỗi giờ. Việc bỏ sót các bài đăng khẩn cấp (như có người bị thương nặng hoặc kẹt trong tòa nhà đổ nát) có thể dẫn tới hậu quả chết người. Do đó, cần một **Hệ hỗ trợ quyết định (DSS)** tự động hóa việc lọc nhiễu, phân loại và phân luồng thông báo khẩn cấp thời gian thực dựa trên cả văn bản (text) và hình ảnh (image).

### 3. Đối tượng sử dụng kết quả phân tích
*   **Ban điều phối cứu trợ trung ương:** Theo dõi tổng quan tình hình thảm họa và đưa ra quyết định phân bổ nguồn lực cấp cao.
*   **Đội phản ứng khẩn cấp (Emergency Dispatch):** Tiếp nhận thông tin về thương vong, cứu hộ khẩn cấp để cử đội cứu nạn trực tiếp tới hiện trường.
*   **Đội kỹ thuật hạ tầng (Infrastructure Team):** Nhận thông tin về cầu sập, đường tắc, mất điện để lên kế hoạch sửa chữa và thông tuyến.
*   **Đội phân phối nhu yếu phẩm (Relief Team):** Điều phối lương thực, nước sạch, thuốc men tới các hộ dân bị ảnh hưởng.
*   **Giám sát viên (Supervisor):** Kiểm duyệt thủ công các trường hợp mô hình nghi ngờ mâu thuẫn đa phương thức hoặc tin giả.

---

## II. MỤC TIÊU HỖ TRỢ RA QUYẾT ĐỊNH

Hệ thống DSS được thiết kế để trả lời 5 câu hỏi cốt lõi giúp các nhà điều phối ra quyết định nhanh hơn:
1.  **Bài đăng này có đáng chú ý không?** (Lọc bỏ tin nhiễu/Spam).
2.  **Bài đăng thuộc danh mục thảm họa/nhân đạo nào?** (Phân loại vấn đề).
3.  **Mức độ ưu tiên xử lý của thông báo này là bao nhiêu?** (High / Medium / Low).
4.  **Nên phân luồng bài đăng này về đội phản ứng nào?** (Emergency / Relief / Infrastructure / Coordination).
5.  **Có cần đưa mẫu này vào hàng kiểm duyệt thủ công (Manual Review) của Supervisor hay không?** (Gắn cờ cảnh báo khi văn bản và hình ảnh mâu thuẫn nhau).

---

## III. MÔ TẢ DỮ LIỆU & TIỀN XỬ LÝ (CRISISMMD DATASET)

### 1. Mô tả dữ liệu
Dự án sử dụng bộ dữ liệu đa phương thức **CrisisMMD v2.0** do Viện Nghiên cứu Điện toán Qatar (QCRI) thu thập và gán nhãn thủ công từ các thảm họa tự nhiên lớn trên thế giới.
*   **Tổng số mẫu:** Khoảng 16,000 cặp văn bản-hình ảnh thô. Dự án đã xây dựng bộ dữ liệu mẫu (sample subset) gồm 200 mẫu đại diện cho 4 thảm họa: `hurricane_harvey`, `mexico_earthquake`, `california_wildfires`, và `nepal_earthquake` phục vụ cho việc kiểm thử và chạy ứng dụng.
*   **Các biến chính:**
    *   `tweet_id`: Mã định danh duy nhất của bài đăng (kiểu số nguyên).
    *   `event_name`: Tên thảm họa (kiểu chuỗi).
    *   `tweet_text`: Nội dung văn bản của tweet (kiểu chuỗi).
    *   `image`: Đường dẫn tương đối tới hình ảnh đính kèm (kiểu chuỗi).
    *   `label` (Nhiệm vụ 1): Phân loại nhị phân tính hữu ích (`informative` / `not_informative`).
    *   `label_top` (Nhiệm vụ 2): Phân loại đa lớp danh mục nhân đạo (`injured_or_dead`, `infrastructure_and_utility_damage`, `rescue_volunteering_or_donation_effort`, `affected_individuals`, `other_relevant_information`, `not_humanitarian`).

### 2. Tiền xử lý dữ liệu
*   **Tiền xử lý Văn bản (Text Preprocessing):**
    *   Làm sạch văn bản thông qua biểu thức chính quy (Regex): Chuyển về chữ thường, loại bỏ HTML entities (như `&amp;`), loại bỏ URLs, xóa tag người dùng (`@mentions`), loại bỏ dấu ký tự đặc biệt nhưng giữ lại từ khóa đi kèm hashtag (như chuyển `#NepalEarthquake` thành `nepalearthquake`).
    *   Loại bỏ các từ dừng (stopwords) tiếng Anh phổ biến.
    *   Vector hóa văn bản sử dụng phương pháp **TF-IDF Vectorizer** (trích xuất 1000 đặc trưng từ đơn và từ ghép).
*   **Tiền xử lý Hình ảnh (Image Preprocessing):**
    *   Load ảnh bằng thư viện Pillow, kiểm tra loại bỏ các tệp tin hỏng/lỗi.
    *   Resize ảnh về kích thước chuẩn $224 \times 224$ và chuyển đổi hệ màu RGB.
    *   Trích chọn đặc trưng bằng mô hình pre-trained **CLIP (Contrastive Language-Image Pretraining)** của OpenAI tạo ra vector embedding 512 chiều.
    *   *Cơ chế Fallback an toàn:* Nếu máy tính chạy không cài đặt thư viện PyTorch hoặc không có Internet để tải CLIP weights, hệ thống tự động kích hoạt bộ trích xuất thuộc tính màu sắc cơ bản (Pillow/Numpy) tạo vector 512 chiều để đảm bảo chương trình không bị lỗi và vẫn chạy mượt mà.

---

## IV. PHƯƠNG PHÁP & HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH (DSS ARCHITECTURE)

Hệ thống được thiết kế theo cấu trúc 3 tầng: **Predictive Layer (Học máy) -> Prescriptive Layer (DSS Layer) -> Presentation Layer (Dashboard Streamlit)**.

```
+-------------------------------------------------------------+
|               DỮ LIỆU ĐẦU VÀO (TWEET TEXT & IMAGE)          |
+-------------------------------------------------------------+
                               |
        +----------------------+----------------------+
        |                                             |
        v                                             v
 [Nhánh Văn bản - TF-IDF]                    [Nhánh Hình ảnh - CLIP]
        |                                             |
        v                                             v
[Logistic Reg. Classifier]                  [Logistic Reg. Classifier]
        |                                             |
        v (P text, Cat text)                          v (P image, Cat image)
        +----------------------+----------------------+
                               |
                               v
               [LATE FUSION & CONFLICT CHECK]
               - P_fusion = 0.6*P_text + 0.4*P_image
               - S_conflict = |P_text - P_image|
                               |
                               v
                     [PRESCRIPTIVE DSS LAYER]
                     - Tính điểm Risk Score (0-100)
                     - Phân Priority (High/Med/Low)
                     - Phân luồng Dispatching & Action
                               |
                               v
              [DASHBOARD ĐIỀU PHỐI CỨU TRỢ STREAMLIT]
```

### 1. Tầng học máy dự báo (Predictive Layer)
*   **Mô hình Text:** Gồm mô hình nhị phân phân loại hữu ích (`text_inf_clf.pkl`) và mô hình đa lớp phân danh mục thảm họa (`text_cat_clf.pkl`) dựa trên các đặc trưng TF-IDF.
*   **Mô hình Image:** Gồm mô hình nhị phân (`image_inf_clf.pkl`) và mô hình đa lớp (`image_cat_clf.pkl`) dựa trên đặc trưng ảnh trích chọn.
*   **Late Fusion (Gộp đa phương thức):**
    *   Xác suất hữu ích cuối cùng: $P_{fusion} = 0.6 \times P_{text} + 0.4 \times P_{image}$.
    *   Xác suất phân lớp danh mục: $P_{cat\_fusion} = 0.6 \times P_{text\_cat\_probs} + 0.4 \times P_{image\_cat\_probs}$. Nhãn danh mục cứu trợ cuối cùng được quyết định bằng phần tử có xác suất trung bình trọng số cao nhất.

### 2. Tầng hỗ trợ quyết định (Prescriptive DSS Layer)
Mô hình học máy chỉ trả ra xác suất dự báo. DSS Layer mới là thành phần biến xác suất này thành hành động.
*   **Tính toán Điểm rủi ro (Risk Score):**
    $$\text{Risk Score} = 0.40 \times (P_{fusion} \times 100) + 0.25 \times (W_{category} \times 100) + 0.20 \times S_{keyword} + 0.15 \times S_{confidence}$$
    *   $W_{category}$: Trọng số khẩn cấp của danh mục (`injured_or_dead` = 1.00, `infrastructure_damage` = 0.80, `affected_individuals` = 0.70, `not_humanitarian` = 0.00).
    *   $S_{keyword}$: Mật độ từ khóa nguy hiểm xuất hiện trong văn bản (như "rescue", "trapped", "flood", "collapse"), mỗi từ khớp cộng 20 điểm, tối đa 100 điểm.
    *   $S_{confidence}$: Độ tin cậy dự báo danh mục của mô hình học máy.
*   **Quy tắc phân luồng cứu trợ (Routing Rules):**
    *   Nếu điểm rủi ro $\le 39 \rightarrow$ **Priority: Low** $\rightarrow$ Không hành động, lưu trữ và theo dõi thêm.
    *   Nếu điểm rủi ro từ $40 - 69 \rightarrow$ **Priority: Medium**.
    *   Nếu điểm rủi ro từ $70 - 100 \rightarrow$ **Priority: High**.
    *   **Phân luồng hành động:**
        *   `injured_or_dead` $\rightarrow$ Chuyển tới **Emergency Team** (Hành động: Xác minh vị trí lập tức và điều động đội cứu nạn).
        *   `infrastructure_and_utility_damage` $\rightarrow$ Chuyển tới **Infrastructure Team** (Hành động: Khảo sát hiện trường sập/tắc nghẽn hạ tầng).
        *   `affected_individuals` / `rescue_volunteering_effort` $\rightarrow$ Chuyển tới **Relief Team** (Hành động: Điều phối lương thực, nước sạch, nhu yếu phẩm).
        *   `other_relevant_information` $\rightarrow$ Chuyển tới **Coordination Team** (Hành động: Ghi nhận thông tin thời tiết/bản đồ sơ tán vào báo cáo chung).
*   **Quy tắc kiểm duyệt thủ công (Manual Review Override):**
    Hệ thống tự động phát hiện mâu thuẫn giữa hai phương thức. Nếu $S_{conflict} = |P_{text} - P_{image}| > 0.5$, hệ thống tự động ghi đè quyết định: Gán độ ưu tiên thành **Medium**, chuyển luồng cứu hộ về **Supervisor** với yêu cầu: *"Yêu cầu người giám sát kiểm duyệt thủ công do văn bản kêu cứu nhưng ảnh không tương xứng (giảm thiểu rủi ro tin giả/spam)"*.

---

## V. TRẢ LỜI CÁC CÂU HỎI BẮT BUỘC CỦA GIẢNG VIÊN

*   **Bài toán thực tế mà nhóm giải quyết là gì?**
    Giải quyết bài toán quá tải thông tin phi cấu trúc (văn bản & ảnh) trên mạng xã hội trong các tình huống thiên tai khẩn cấp, giúp lọc bỏ tin rác và xác định nhanh các thông tin cứu hộ thực tế.
*   **Ai là người sử dụng kết quả phân tích hoặc hệ thống?**
    Trung tâm điều phối cứu nạn khẩn cấp, chỉ huy các đội cứu hộ cứu nạn (Emergency Team), đội khắc phục hạ tầng (Infrastructure Team), đội phân phối nhu yếu phẩm (Relief Team) và người giám sát hệ thống (Supervisor).
*   **Người dùng cần ra quyết định gì?**
    Quyết định ưu tiên xử lý bài đăng nào trước (Priority Queue), cử đội cứu hộ nào đến hiện trường (Routing Team), hành động cứu nạn cần thực hiện là gì (Action) và có cần kiểm duyệt lại tin đăng để tránh tin giả hay không (Manual Review).
*   **Dữ liệu nào được sử dụng để hỗ trợ quyết định đó?**
    Văn bản tweet (chứa thông tin vị trí, trạng thái khẩn cấp, yêu cầu cứu nạn) và hình ảnh đi kèm tweet (chứng minh thiệt hại vật chất, hình ảnh ngập lụt, sập nhà, thương vong thực tế) trích xuất từ dataset CrisisMMD.
*   **Nhóm đã xử lý và phân tích dữ liệu như thế nào?**
    Làm sạch text bằng Regex, trích chọn đặc trưng TF-IDF; chuẩn hóa ảnh và trích chọn vector đặc trưng ngữ nghĩa bằng CLIP; huấn luyện mô hình phân loại Logistic Regression nhị phân/đa lớp cho từng nhánh; gộp xác suất dự đoán (Late Fusion) và áp dụng Rule Engine của lớp quyết định DSS.
*   **Kết quả quan trọng nhất là gì?**
    Nội dung cũ ở mục này đã bị hủy sau audit. Số liệu và diễn giải hợp lệ chỉ nằm trong `reports/latex/main.pdf`.
*   **Từ kết quả đó, nhóm đề xuất quyết định hoặc hành động nào?**
    Nhóm đề xuất vận hành hệ thống thông qua Hàng đợi ưu tiên (Priority Queue) trên Dashboard, các đội cứu hộ trực tiếp tiếp nhận thông tin được phân luồng tự động để triển khai lực lượng; đồng thời Supervisor rà soát các bài đăng bị gắn cờ "Manual Review" để chống tin giả.
*   **Kết quả có hạn chế, rủi ro hoặc điều kiện áp dụng nào không?**
    Hạn chế là dữ liệu mạng xã hội thường thiếu tọa độ GPS chính xác (cần bổ sung module NLP trích xuất địa chỉ tự động). Rủi ro là nếu mất kết nối Internet, việc tải weights mô hình CLIP lớn sẽ bị ảnh hưởng (hệ thống khắc phục bằng thuật toán fallback màu sắc cục bộ).

---

## VI. KẾT QUẢ ĐÁNH GIÁ & GIAO DIỆN HỆ THỐNG

### 1. Đánh giá chất lượng mô hình
Phần đánh giá cũ đã bị hủy. Báo cáo chính thức phải hiển thị dummy baseline và không diễn giải F2 informative độc lập.

### 2. Giao diện Dashboard (Streamlit App)
*   **Page 1 - Overview:** Thống kê tổng số tin phân tích, tỷ lệ tin hữu ích, phân bố thảm họa và tỷ lệ phân luồng cứu trợ thông qua biểu đồ Plotly sinh động.
*   **Page 2 - EDA:** Trực quan hóa độ dài tweet, biểu đồ các từ khóa khẩn cấp hàng đầu, và gallery preview ảnh thực tế theo nhóm thảm họa.
*   **Page 3 - Model Evaluation:** Bảng so sánh chỉ số Accuracy, Precision, Recall, F1, F2-score của các mô hình và hiển thị ma trận nhầm lẫn tương tác.
*   **Page 4 - Single Case Demo:** Giả lập trực quan: Người dùng nhập tweet kêu cứu, chọn hoặc tải lên ảnh. DSS tính toán điểm rủi ro thời gian thực, bôi đỏ nổi bật các từ khóa nguy cấp (như *trapped, injured, collapsed*) và hiển thị card điều phối cứu trợ màu sắc rực rỡ.
*   **Page 5 - Priority Queue:** Danh sách hàng đợi cứu hộ được sắp xếp tự động theo Risk Score giảm dần, tích hợp bộ lọc nhanh và nút tải xuống file CSV kết quả điều phối để bàn giao cho các đội phản ứng nhanh ngoài thực địa.

---

## VII. HẠN CHẾ & HƯỚNG PHÁT TRIỂN
*   **Hạn chế:** Hệ thống hiện tại sử dụng dữ liệu tĩnh dạng file. Các hình ảnh mô phỏng trong tập sample mới chỉ dừng ở mức nhận diện đặc trưng màu sắc nền đại diện (được thiết kế cho thử nghiệm nhanh).
*   **Hướng phát triển:** Tích hợp mô hình ngôn ngữ lớn (LLM) để tự động trích xuất thực thể địa lý (Named Entity Recognition - NER) giúp xác định chính xác địa chỉ của người kêu cứu; liên kết API để cào dữ liệu thời gian thực (Real-time Crawling) trên Twitter/Facebook về thảm họa đang xảy ra.

---

## PHỤ LỤC 1: BẢNG TỰ CHẤM ĐIỂM CỦA NHÓM (SELF-EVALUATION)

*(Nhóm sinh viên tự rà soát và đánh giá bài làm trước khi nộp theo đúng tiêu chí Mục 9)*

| STT | Tiêu chí tự đánh giá | Điểm tối đa | Điểm nhóm tự chấm | Minh chứng / Ghi chú |
| :--- | :--- | :---: | :---: | :--- |
| 1 | Hiểu vấn đề và xác định bài toán ra quyết định | 15 | **15** | Báo cáo Mục I & II phân tích rõ bối cảnh cứu trợ, chỉ rõ 5 quyết định cần hỗ trợ cho 5 đối tượng cứu nạn cụ thể. |
| 2 | Dữ liệu và tiền xử lý dữ liệu | 15 | **15** | Báo cáo Mục III; code tiền xử lý chi tiết tại [text_preprocessing.py](file:///C:/Users/doand/Documents/antigravity/hopeful-fermi/src/text_preprocessing.py) và [image_preprocessing.py](file:///C:/Users/doand/Documents/antigravity/hopeful-fermi/src/image_preprocessing.py). |
| 3 | Phân tích dữ liệu và phát hiện insight | 20 | **20** | Trang 2 (EDA) của DashboardStreamlit hiển thị biểu đồ phân bố và phân tích chi tiết 4 kịch bản mâu thuẫn đa phương thức; code tại [run_eda.py](file:///C:/Users/doand/Documents/antigravity/hopeful-fermi/src/run_eda.py). |
| 4 | Mô hình / hệ thống hỗ trợ ra quyết định | 20 | **20** | Cấu trúc Late Fusion kết hợp Rule Engine (Risk Score, Routing, Review Override). Minh chứng ma trận nhầm lẫn chi tiết tại Trang 3 (Model Evaluation); code tại [fusion.py](file:///C:/Users/doand/Documents/antigravity/hopeful-fermi/src/fusion.py) và [decision_rules.py](file:///C:/Users/doand/Documents/antigravity/hopeful-fermi/src/decision_rules.py). |
| 5 | Diễn giải kết quả và khuyến nghị quyết định | 15 | **15** | Báo cáo Mục V (Trả lời 8 câu hỏi bắt buộc) và Mục VIII (Khuyến nghị hành động cứu trợ có căn cứ dữ liệu). |
| 6 | Chất lượng báo cáo viết và hồ sơ minh chứng | 10 | **10** | Tài liệu README hướng dẫn chạy rõ ràng; Báo cáo được biên soạn chi tiết, dễ hiểu, trình bày khoa học bằng Markdown. |
| 7 | Tổ chức nhóm và đóng góp cá nhân | 5 | **5** | Phân công công việc công bằng (25% mỗi người) được lưu trữ rõ ràng tại tài liệu [contribution_log.md](file:///C:/Users/doand/Documents/antigravity/hopeful-fermi/docs/contribution_log.md). |
| | **TỔNG CỘNG** | **100** | **100** | |
| | **Điểm quy đổi hệ 10** | **10** | **10** | **(Tổng điểm / 10)** |
