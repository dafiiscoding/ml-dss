# Hệ hỗ trợ quyết định điều phối giao pizza dựa trên dự báo đơn giao trễ

> Bản báo cáo chính thức nằm ở `reports/latex/main.tex` và
> `reports/PIZZA_DSS_REPORT.pdf`. File markdown này là bản đọc nhanh để nhóm
> nắm bố cục, luận điểm và thứ tự trình bày.

## Tóm tắt

Dự án xây dựng một nguyên mẫu Hệ hỗ trợ quyết định (Decision Support System -
DSS) cho bài toán điều phối giao pizza. Dữ liệu gồm 1.004 đơn hàng từ Kaggle,
có thông tin về nhà hàng, địa điểm, loại pizza, size, số topping, khoảng cách,
mức traffic, phương thức thanh toán và trạng thái giao trễ.

Mục tiêu của báo cáo không phải chứng minh hệ thống có thể triển khai ngay
trong vận hành thật, mà là thể hiện một quy trình DSS học thuật đầy đủ: hiểu
bài toán ra quyết định, kiểm soát chất lượng dữ liệu, phân tích dữ liệu,
kiểm định giả thiết, xây dựng mô hình dự báo, chuyển kết quả dự báo thành
khuyến nghị vận hành, minh họa dashboard và bài toán tối ưu phân công.

## 1. Bài toán ra quyết định

Người quản lý vận hành cần trả lời các câu hỏi:

1. Đơn nào có nguy cơ giao trễ trước khi đơn hoàn tất?
2. Các yếu tố nào liên quan đến rủi ro trễ?
3. Khi có nhiều đơn cần theo dõi, đơn nào nên được ưu tiên?
4. Dashboard cần hiển thị gì để hỗ trợ điều phối và báo cáo?

Từ đó DSS được chia thành ba lớp:

- Predictive layer: mô hình dự báo xác suất `is_delayed`.
- Prescriptive layer: Risk Score, Priority và Recommended Action.
- Presentation layer: dashboard Streamlit và Power BI-ready data pack.

## 2. Dữ liệu và kiểm soát chất lượng

Dataset có 1.004 dòng, 25 cột gốc, không có missing value và không trùng
`order_id`. Target `is_delayed` có 210 đơn trễ và 794 đơn đúng giờ, tỷ lệ trễ
20,92%.

Phát hiện quan trọng nhất là nguồn dữ liệu chỉ cung cấp nhãn `is_delayed`,
không cung cấp tài liệu SLA nói rõ bao nhiêu phút thì trễ. Nhóm suy luận ranh
giới nhãn bằng cách thử các luật ngưỡng trên `delivery_duration_min` và đếm số
dòng mismatch với nhãn gốc. Kết quả:

- duration lớn nhất của nhóm on-time là 30 phút;
- duration nhỏ nhất của nhóm delayed là 35 phút;
- luật `delivery_duration_min > 30` mismatch 0/1.004 dòng;
- luật `delivery_duration_min >= 35` cũng mismatch 0/1.004 dòng.

Vì duration chỉ nhận bội số 5 phút, hai luật trên là tương đương trong dataset.
Báo cáo dùng cách viết `duration > 30` như ranh giới hiệu dụng, không phải SLA
do nhóm tự đặt. Từ đó, các cột chỉ biết sau khi giao xong như
`delivery_duration_min`, `delivery_time`, `delay_min`,
`delivery_efficiency_min_per_km` và `restaurant_avg_time` bị loại khỏi feature
mô hình. Đây là bước chống leakage chính của dự án.

Dữ liệu cũng có nhiều dấu hiệu synthetic: duration nằm trên lưới 5 phút, một
số biến là công thức tất định, và nhiều biến categorical có dấu hiệu được sinh
ngẫu nhiên hoặc theo generator đơn giản. Báo cáo vì vậy diễn giải kết quả theo
hướng học thuật, không overclaim thành kết luận kinh doanh thật.

## 3. Quy trình phân tích

Pipeline học thuật gồm các bước:

1. Audit dữ liệu và chuẩn hóa schema.
2. Feature engineering có kiểm soát leakage.
3. Chia train/dev/test stratified.
4. EDA: phân phối delay, traffic, distance, size, type, hour.
5. Kiểm định giả thiết cho biến categorical.
6. Data forensics: phục dựng công thức sinh dữ liệu và kiểm tra artifact.
7. Threshold inference: suy luận ranh giới trễ từ nhãn `is_delayed`.
8. Delay deep-dive: duration grid, delay severity, distance/traffic/complexity.
9. Customer behavior: size/type preference, type-size combo, peak hour.
10. Brand/location/state analysis: mix sản phẩm và rủi ro theo quán/khu vực.
11. Forecasting: minh họa demand forecast và preference trend forecast.
12. Recommendation: rule-based recommendation theo popularity/context.
13. Modeling: so sánh sáu classifier và chọn model trên dev.
14. DSS: Risk Score, Priority, Recommended Action.
15. Tối ưu vận tải: assignment đơn High priority cho fleet giả lập.
16. Dashboard, Power BI pack, report, slide và kiểm thử.

### Cách tìm 7 công thức tất định

Nhóm không chỉ liệt kê công thức, mà kiểm chứng bằng quy trình:

1. đọc tên cột và quan hệ nghiệp vụ có thể có;
2. đề xuất biểu thức ứng viên;
3. tính biểu thức trên toàn bộ 1.004 dòng;
4. đo `max_abs_error` giữa biểu thức và cột gốc;
5. chỉ gọi là tất định nếu sai số lớn nhất bằng 0.

Các công thức khớp tuyệt đối:

| Cột | Công thức/luật khớp | Cách phát hiện |
|---|---|---|
| `estimated_duration_min` | `2.4 * distance_km` | ratio estimated/distance không đổi |
| `topping_density` | `toppings_count / distance_km` | tên cột gợi ý mật độ |
| `pizza_complexity` | `toppings_count * size_score` | mã hóa size 1-4 rồi nhân topping |
| `traffic_impact` | Low=1, Medium=2, High=3 | mapping một-một với traffic |
| `delay_min` | `duration - estimated_duration` | định nghĩa delay hậu nghiệm |
| `delivery_efficiency_min_per_km` | `duration / distance` | phút/km |
| `is_delayed` | ranh giới giữa 30 và 35 phút | threshold sweep mismatch 0 |

## 3b. EDA chi tiết cần trình bày

Phần EDA hiện có các nhóm phân tích sau:

- Độ trễ: on-time tối đa 30 phút, delayed tối thiểu 35 phút; nhóm Late 40 và
  Late 45-50 có high-traffic share rất cao.
- Món yêu thích: Medium là size phổ biến nhất; Non-Veg là type nhiều đơn nhất;
  Cheese Burst có delay rate cao hơn Non-Veg/Veg.
- Combo type-size: dùng để gợi ý recommendation và mix sản phẩm, nhưng nhóm
  nhỏ phải đọc kèm số đơn.
- Brand/quán: mỗi restaurant có orders, delay rate, avg distance, traffic share,
  top type và top size khác nhau; không kết luận chất lượng brand thật.
- Location/state: dataset không có cột bang chính thức; `state_code` được suy
  ra từ chuỗi `location` dạng `City, ST`, nên chỉ dùng mô tả.
- Kiểm định giả thiết: chi-square cho traffic, payment, size, type, location,
  restaurant và time segment; dùng như bằng chứng association, không suy luận
  nhân quả.

## 4. Mô hình và kết quả

Dự án so sánh Logistic Regression, Decision Tree, Gaussian Naive Bayes, k-NN,
SVM và Random Forest. Model được chọn theo F2 trên dev vì bài toán ưu tiên
không bỏ sót đơn có nguy cơ trễ. Tuy vậy báo cáo không dùng F2 một mình mà luôn
đặt cạnh Accuracy, Balanced Accuracy, F1 và MCC.

Model khóa là Logistic Regression trên feature set compact. Kết quả test:

| Metric | Giá trị |
|---|---:|
| Accuracy | 0,9602 |
| Balanced Accuracy | 0,9661 |
| Precision | 0,8542 |
| Recall | 0,9762 |
| F1 | 0,9111 |
| F2 | 0,9491 |
| MCC | 0,8889 |
| ROC-AUC | 0,9961 |

Baseline always-on-time có Accuracy 0,7910 nhưng F2 bằng 0 vì bỏ sót toàn bộ
đơn trễ. Baseline always-delayed có Recall cao nhưng cảnh báo quá nhiều. Mô
hình Logistic Regression vượt hai baseline trên các metric cân bằng.

## 5. Tầng DSS và dashboard

Xác suất delayed được chuyển thành Risk Score bằng cách kết hợp xác suất mô
hình với traffic, distance, peak hour, complexity và weekend. Risk Score sau đó
được ánh xạ thành ba mức Priority: Low, Medium, High.

Recommended Action:

- High: báo điều phối trưởng, chuẩn bị tài xế dự phòng, thông báo customer
  support.
- Medium: theo dõi tài xế/traffic và cập nhật khách nếu ETA xấu đi.
- Low: giữ trong hàng đợi bình thường.

Dashboard Streamlit gồm các tab Overview, EDA, Customer Behavior, Forecast &
Staffing, Model Evaluation, Single Order Demo, Delay Queue và Data Quality.
Power BI pack cung cấp fact/dim CSV, measures DAX và dashboard spec để dựng
dashboard native nếu lớp yêu cầu.

## 6. Giới hạn học thuật

1. Dataset nhỏ và có tính synthetic.
2. Target được suy luận là có ranh giới giữa 30 và 35 phút duration, nên kiểm
   soát leakage là bắt buộc.
3. Chronological holdout không có đơn trễ, nên báo cáo dùng stratified split.
4. Forecast chỉ minh họa quy trình vì MAPE còn cao.
5. Brand, preference trend và location có thể phản ánh artifact của generator.
6. Fleet/tài xế trong bài toán vận tải là kịch bản giả lập.
7. Priority là chính sách minh bạch, chưa tối ưu bằng cost thật hoặc SLA thật.

## 7. Thứ tự đọc và trình bày

Khi đọc hoặc thuyết trình, nên đi theo thứ tự:

1. Chương 1: xác định bài toán DSS và quyết định cần hỗ trợ.
2. Chương 3: giải thích dataset, leakage và vì sao phải audit synthetic.
3. Chương 4: trình bày EDA, kiểm định và insight kinh doanh.
4. Chương 5: trình bày feature hợp lệ, model, baseline và test metric.
5. Chương 6: giải thích Risk Score, Priority, dashboard và assignment.
6. Chương 7: nêu minh chứng tái lập, runbook và đối chiếu rubric học phần.
7. Chương 8: kết luận thận trọng, nhấn mạnh giới hạn dữ liệu.

Luận điểm chính cần giữ nhất quán: dự án đầy đủ quy trình DSS, nhưng kết luận
kinh doanh được diễn giải thận trọng vì dữ liệu Kaggle có nhiều dấu hiệu được
sinh tự động.
