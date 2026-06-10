# Danh sách gói công việc để phân công

File này dùng để nhóm chốt phân công trước khi điền vào `docs/team_info.json`.
Không đưa nguyên danh sách này vào báo cáo cuối; báo cáo chỉ hiển thị nhiệm vụ
đã được gán cho từng thành viên.

## Các gói công việc

1. **Xác định bài toán và yêu cầu DSS**
   - Phân tích bối cảnh cứu trợ thảm họa.
   - Xác định người dùng, quyết định cần hỗ trợ, phạm vi và tiêu chí thành công.

2. **Nghiên cứu dữ liệu CrisisMMD**
   - Tìm hiểu nguồn dữ liệu, schema, nhãn informative/humanitarian.
   - Mô tả phạm vi, hạn chế và ý nghĩa các trường dữ liệu.

3. **Nạp, làm sạch và kiểm soát chất lượng dữ liệu**
   - Xây dựng loader, join nhãn chính thức và cố định train/dev/test.
   - Kiểm tra thiếu, trùng, ảnh hỏng, sai kiểu và rò rỉ xuyên split.

4. **Tiền xử lý và đặc trưng văn bản**
   - Làm sạch tweet, xử lý URL/mention/hashtag.
   - Xây dựng TF-IDF unigram/bigram và kiểm soát leakage của vectorizer.

5. **Tiền xử lý và đặc trưng hình ảnh**
   - Kiểm tra ảnh, chuẩn hóa đầu vào và trích vector CLIP pretrained.
   - Kiểm tra cache embedding và thứ tự metadata.

6. **Phân tích khám phá dữ liệu**
   - Phân tích phân bố lớp, sự kiện, độ dài văn bản và từ khóa.
   - Phân tích ảnh, mâu thuẫn text--image và liên hệ với quyết định.

7. **Clustering và association analysis**
   - Thực hiện K-Means, silhouette sweep và diễn giải cụm.
   - Thực hiện Apriori, lọc luật hiển nhiên và phân tích support/confidence/lift.

8. **Xây dựng và so sánh mô hình**
   - Huấn luyện DT, NB, k-NN, SVM, Random Forest và Logistic Regression.
   - Đánh giá bằng metric phù hợp, confusion matrix và baseline.

9. **Late Fusion và đánh giá robustness**
   - Căn chỉnh xác suất hai phương thức và chọn trọng số trên dev.
   - Phân tích duplicate, robust mask, bootstrap và độ ổn định theo sự kiện.

10. **Thiết kế tầng quyết định DSS**
    - Xây dựng Risk Score, Priority, Routing và Manual Review.
    - Kiểm tra scenario, độ nhạy threshold và các giới hạn chính sách.

11. **Dashboard và minh chứng sản phẩm**
    - Xây dựng các trang Streamlit, bộ lọc, priority queue và single-case demo.
    - Kiểm thử giao diện và chuẩn bị screenshot.

12. **Báo cáo, tài liệu và nghiệm thu**
    - Viết báo cáo, tổng hợp hình/bảng, tài liệu tham khảo và khuyến nghị.
    - Kiểm tra tính nhất quán số liệu, hướng dẫn chạy và hồ sơ nộp.

## Cách điền sau khi phân công

Mỗi thành viên có thể nhận nhiều gói. Trong `docs/team_info.json`:

- `contribution`: mô tả ngắn các gói và trách nhiệm thực tế;
- `share_percent`: tỷ lệ đóng góp, không kèm ký hiệu `%`;
- tổng tỷ lệ của bốn thành viên nên bằng 100.
