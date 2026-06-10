# Ma trận khái niệm cho báo cáo học phần

Tài liệu này là checklist biên soạn. Mục tiêu là bảo đảm báo cáo có thể tự
đứng độc lập: người đọc chưa học DSS hoặc học máy vẫn hiểu được ý chính, còn
người đã học môn có thể kiểm tra tính đúng đắn của phương pháp.

## Quy tắc trình bày bắt buộc

Mỗi khái niệm xuất hiện lần đầu phải có đủ:

1. Định nghĩa bằng ngôn ngữ phổ thông.
2. Trực giác hoặc ví dụ ngắn.
3. Cơ chế/công thức ở mức cần thiết.
4. Vai trò cụ thể trong dự án.
5. Lý do chọn thay vì phương án khác.
6. Giả định, giới hạn hoặc cách dễ hiểu sai.

Không dồn toàn bộ định nghĩa vào một chương lý thuyết. Khái niệm phải được
nhắc lại ngắn ở nơi nó được sử dụng, để mạch đọc không bị đứt.

## Nhóm A - Kiến thức cốt lõi trong chương trình

Các mục này có trong slide, nhưng báo cáo vẫn phải giải thích lại thay vì giả
định người đọc nhớ bài.

| Khái niệm | Nội dung phải giải thích | Ví dụ/hình cần có | Vị trí |
|---|---|---|---|
| Hệ hỗ trợ quyết định | DSS hỗ trợ chứ không thay thế người quyết định; gồm dữ liệu, mô hình và giao diện | Một tweet đi từ dữ liệu đến khuyến nghị | Chương nền tảng DSS |
| Quy trình ra quyết định | Nhận biết vấn đề, xây phương án, lựa chọn, thực thi và phản hồi | Chu trình điều phối cứu trợ | Chương nền tảng DSS |
| Học có giám sát | Học ánh xạ từ đầu vào có nhãn sang đầu ra | Tweet/ảnh và nhãn informative | Chương phương pháp |
| Train/dev/test | Train học tham số, dev chọn cấu hình, test đánh giá cuối | Sơ đồ ba tập và các thao tác được phép | Chương thực nghiệm |
| Decision Tree | Chia không gian bằng các luật; entropy/Gini; dễ hiểu nhưng dễ quá khớp | Cây nhỏ minh họa từ khóa cứu trợ | Chương thuật toán |
| Naive Bayes | Bayes và giả định độc lập có điều kiện | Xác suất từ khóa `rescue` theo lớp | Chương thuật toán |
| k-NN | Phân loại bằng các điểm gần nhất; phụ thuộc khoảng cách và số chiều | TF-IDF thưa so với embedding ảnh | Chương thuật toán |
| SVM | Tìm siêu phẳng có biên lớn; đầu ra gốc không phải xác suất | Hình biên phân lớp | Chương thuật toán |
| Random Forest | Nhiều cây trên mẫu/thuộc tính ngẫu nhiên để giảm phương sai | Sơ đồ voting nhiều cây | Chương thuật toán |
| K-Means | Gán điểm về centroid gần nhất và cập nhật centroid lặp lại | Quy trình bốn bước | Chương EDA |
| Apriori | Tìm itemset thường xuyên dựa trên tính chất tập con | Giỏ hashtag của một tweet | Chương EDA |
| Support/confidence/lift | Độ phổ biến, độ tin cậy có điều kiện và mức vượt độc lập | Tính tay một luật hashtag | Chương EDA |
| Text Mining | Chuyển văn bản phi cấu trúc thành đặc trưng có thể phân tích | Tweet trước/sau làm sạch | Chương tiền xử lý |
| TF-IDF | Từ quan trọng khi thường gặp trong tài liệu nhưng hiếm trong corpus | Tính TF-IDF cho 2-3 tweet nhỏ | Chương đặc trưng |
| Document embedding | Vector số biểu diễn nội dung tài liệu | Hai văn bản gần nghĩa nằm gần nhau | Chương đặc trưng |
| Social Media Mining | Đặc điểm dữ liệu mạng xã hội: ngắn, nhiễu, hashtag, sự kiện và lan truyền | Ví dụ URL, mention, RT | Chương dữ liệu |

## Nhóm B - Kiến thức mở rộng, không được trình bày đầy đủ trong môn

Các mục này phải được giới thiệu từ nền tảng, không chỉ nêu tên.

| Khái niệm | Mức giải thích bắt buộc | Ví dụ/hình/công thức | Điều cần tránh |
|---|---|---|---|
| Dữ liệu đa phương thức | Một đối tượng có nhiều nguồn bằng chứng; mỗi nguồn có thể bổ sung hoặc mâu thuẫn | Cùng tweet nhưng text và ảnh nói khác nhau | Không gọi hai modality là hai ground truth độc lập |
| Biểu diễn đặc trưng | Mô hình không đọc trực tiếp chữ/ảnh mà nhận vector số | Text thành TF-IDF, ảnh thành vector 512 chiều | Không đồng nhất embedding với dự báo |
| Mô hình tiền huấn luyện | Mô hình đã học từ dữ liệu lớn trước khi áp dụng sang nhiệm vụ mới | CLIP học từ cặp ảnh--mô tả | Không nói nhóm tự huấn luyện CLIP |
| Transfer learning | Tái sử dụng tri thức đã học ở nhiệm vụ/nguồn dữ liệu khác | Giữ CLIP, huấn luyện classifier trên CrisisMMD | Không gọi là fine-tune khi trọng số CLIP giữ nguyên |
| Frozen feature extractor | Đóng băng trọng số và chỉ lấy đặc trưng đầu ra | Sơ đồ ảnh → CLIP cố định → classifier | Không gộp classifier với CLIP thành một model được huấn luyện end-to-end |
| Vision Transformer | Chia ảnh thành patch và xử lý chuỗi patch bằng Transformer ở mức trực giác | Hình ảnh chia lưới patch | Không đi sâu attention toán học không cần thiết |
| CLIP | Hai encoder ảnh/text được học để đưa cặp phù hợp lại gần trong không gian chung | Ảnh lũ và mô tả “flooded street” | Không tuyên bố CLIP hiểu mức độ khẩn cấp |
| Embedding 512 chiều | Mỗi ảnh được mô tả bằng 512 số; khoảng cách phản ánh một phần tương đồng ngữ nghĩa | Vector rút gọn 3 chiều minh họa | Không diễn giải từng chiều riêng lẻ |
| Chuẩn hóa \(L_2\) | Chia vector cho norm để độ dài bằng 1, giảm ảnh hưởng thang đo | Công thức và vector nhỏ | Không gọi là chuẩn hóa ảnh đầu vào |
| Logistic Regression | Mô hình tuyến tính tạo xác suất bằng sigmoid/softmax | Đường biên và xác suất | Không nhầm với hồi quy biến liên tục |
| Class weight | Tăng chi phí lỗi của lớp hiếm trong hàm tối ưu | Lớp missing có ít mẫu hơn | Không khẳng định class weight tự giải quyết được thiếu dữ liệu |
| Probability calibration | Biến score thành xác suất có ý nghĩa tương đối; cần cho fusion/threshold | Score SVM trước/sau sigmoid | Không gọi xác suất đã hiệu chỉnh là chắc chắn tuyệt đối |
| Late Fusion | Kết hợp xác suất sau khi mỗi modality đã dự báo riêng | Công thức trọng số text/image | Không nhầm với nối vector ở đầu vào |
| Căn chỉnh lớp xác suất | Hai classifier phải dùng cùng thứ tự tám lớp trước khi cộng vector | Ví dụ hoán đổi hai nhãn gây sai | Không cộng trực tiếp theo vị trí nếu class order khác |
| Threshold tuning | Ngưỡng quyết định được chọn trên dev theo mục tiêu, không mặc định 0,5 | Đồ thị threshold–F2/capacity | Không chọn threshold bằng test |
| Conflict score | Định lượng mức bất đồng giữa hai modality | Chênh informative và total variation category | Không gọi conflict cao là tin giả |
| Total variation distance | Một nửa tổng chênh lệch tuyệt đối giữa hai phân bố xác suất | Hai vector xác suất tám lớp rút gọn | Không mô tả như khoảng cách ảnh gốc |
| Human-in-the-loop | Con người xử lý trường hợp rủi ro/không chắc chắn mà mô hình không nên tự quyết | Supervisor review queue | Không tuyên bố hệ thống tự động điều động sinh tử |
| Capacity constraint | Hàng kiểm duyệt phải phù hợp năng lực con người | Tối đa khoảng 25% mẫu dev | Không chỉ tối ưu recall rồi đẩy mọi mẫu cho người |
| Risk Score | Hàm chính sách tổng hợp nhiều tín hiệu thành một thang điểm vận hành | Tính tay một case | Không gọi là ground truth hoặc model học được |
| Policy sensitivity | Thử các ngưỡng khác nhau để thấy quyết định vận hành thay đổi ra sao | Sensitive/current/strict | Không chọn policy tối ưu khi thiếu nhãn Priority |

## Nhóm C - Kiểm soát đánh giá và độ tin cậy

Đây là phần ngoài trọng tâm thuật toán của môn nhưng cần thiết để kết quả có
giá trị học thuật.

| Khái niệm | Nội dung phải giải thích | Ví dụ/hình | Điều cần tránh |
|---|---|---|---|
| Data leakage | Thông tin từ dev/test lọt vào huấn luyện hoặc chọn model làm metric lạc quan | Cùng ảnh xuất hiện ở train và test | Không chỉ kiểm tra trùng ID |
| Exact duplicate | Hai file/nội dung giống hệt dù ID khác | Hai ảnh có cùng SHA-256 | Không nhầm với ảnh cùng chủ đề |
| SHA-256 | Hàm băm tạo dấu vân tay file; cùng file cho cùng hash | Luồng file → hash | Không dùng để đo gần giống |
| Perceptual hash | Dấu vân tay giữ tương đồng thị giác qua resize/nén | Hai ảnh khác byte nhưng cùng cảnh | Không tự động coi mọi pHash gần là duplicate |
| Cosine similarity | Góc giữa hai vector đo hướng tương đồng | Hai embedding chuẩn hóa | Không suy ra đồng nhất nội dung chỉ từ cosine cao |
| Pairwise review | Xem từng cặp ứng viên và gán near-copy/related/distinct | Contact sheet và tiêu chí | Không thay đổi canonical test hậu nghiệm |
| Canonical mask | Tập đánh giá chính chỉ loại duplicate chắc chắn theo quy tắc định trước | Sơ đồ hàng được giữ/loại | Không trộn với robust mask |
| Robust mask | Tập nhạy cảm riêng loại thêm near-duplicate đã xác minh | Canonical 2169 → robust 2032 | Không tune lại trên robust test |
| Dummy baseline | Chiến lược đơn giản để kiểm tra metric có thực sự tốt | Always-informative, majority class | Không chỉ so model phức tạp với nhau |
| Balanced Accuracy | Trung bình recall của các lớp, giảm ảnh hưởng lớp lớn | Binary imbalance example | Không thay thế mọi metric khác |
| MCC | Tương quan giữa dự báo và nhãn, bằng 0 với dự báo hằng | Confusion matrix 2×2 | Không cần chứng minh công thức dài |
| Average Precision | Tóm tắt precision–recall theo ranking xác suất | PR curve | Không nhầm với precision tại một threshold |
| Macro-F1 | Tính F1 từng lớp rồi trung bình đều | Lớp lớn/nhỏ có trọng số ngang nhau | Không nhầm weighted-F1 |
| F2 | F-score đặt Recall nặng hơn Precision | Chi phí bỏ sót tin khẩn cấp | Không dùng riêng khi always-positive đã cao |
| Bootstrap | Lấy mẫu lại có hoàn lại để ước lượng độ dao động metric | Minh họa nhiều mẫu bootstrap | Không gọi đây là thử nghiệm event mới |
| Khoảng tin cậy | Khoảng giá trị tương thích với biến động lấy mẫu theo giả định | Error bar 95% | Không diễn giải là xác suất 95% tham số nằm trong khoảng |
| Paired gain | Tính chênh lệch hai hệ thống trên cùng bootstrap sample | Fusion minus dummy | Không so hai CI riêng để kết luận chênh lệch |
| Domain shift | Phân bố dữ liệu thay đổi giữa sự kiện/thời gian/nguồn | Sri Lanka khác Harvey | Không xem metric 2017 là bảo đảm cho sự kiện mới |
| Event stability | Đánh giá metric theo từng thảm họa để thấy biến động | Bar chart theo event | Không thay thế leave-one-event-out |
| t-SNE | Chiếu dữ liệu nhiều chiều xuống 2D để quan sát lân cận cục bộ | CLIP scatter | Không dùng khoảng cách toàn cục hoặc cụm 2D làm bằng chứng phân lớp |

## Nhóm D - Công nghệ và triển khai

| Khái niệm | Nội dung phải giải thích | Vị trí |
|---|---|---|
| Pipeline tái lập | Các bước dữ liệu → đặc trưng → model → fusion → DSS → dashboard | Kiến trúc hệ thống |
| Cache embedding | Lưu kết quả CLIP tốn thời gian để không chạy lại | Chương triển khai |
| Serialization/pickle | Lưu classifier/vectorizer đã huấn luyện để inference | Chương triển khai |
| Streamlit | Framework Python tạo dashboard tương tác cho prototype phân tích | Chương sản phẩm |
| Plotly | Thư viện biểu đồ tương tác dùng cho filter/drill-down | Chương sản phẩm |
| Community Cloud | Dịch vụ chạy app từ GitHub; khác với URL localhost | Hướng dẫn chạy |
| Lazy loading | Chỉ nạp model ảnh/CLIP khi cần để giảm cold start | Chương triển khai |
| Cold start | Lần chạy đầu tải dependency/model lâu hơn các lần sau | Chương triển khai |

## Chuỗi ví dụ xuyên suốt báo cáo

Báo cáo sẽ dùng một ví dụ giả định duy nhất xuyên các chương:

> “Urgent! People are trapped after the bridge collapsed in rising flood
> water.” kèm một ảnh đường ngập.

Ví dụ này được dùng để minh họa:

1. Làm sạch tweet và tạo TF-IDF.
2. Trích embedding ảnh bằng CLIP cố định.
3. Hai classifier tạo xác suất.
4. Căn chỉnh lớp và late fusion.
5. Tính conflict.
6. Tính Risk Score.
7. Gán Priority, Routing và Manual Review.

Các con số minh họa phải được ghi rõ là ví dụ tính toán hoặc lấy trực tiếp từ
inference thật; không trộn hai loại.

## Tiêu chí hoàn thành checkpoint

- Không còn thuật ngữ ngoài chương trình xuất hiện mà chưa có định nghĩa.
- Không dùng từ `fine-tune CLIP`; mô tả nhất quán là giữ nguyên trọng số CLIP.
- Mỗi công thức đều có giải thích biến và ví dụ số.
- Mỗi kỹ thuật đều trả lời “tại sao cần trong quyết định cứu trợ”.
- Mỗi hình đều có đoạn diễn giải, không chỉ có caption.
