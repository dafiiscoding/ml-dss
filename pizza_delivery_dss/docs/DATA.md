# Data - Pizza Delivery DSS

## Nguồn

Dataset Kaggle:

```text
akshaygaikwad448/pizza-delivery-data-with-enhanced-features
```

Tên trên Kaggle: `Enhanced Pizza Sales Data (2024-2025)`.

File dùng trong dự án:

```text
data/raw/Enhanced_pizza_sell_data_2024-25.xlsx
```

## Schema chuẩn hóa

| Cột chuẩn hóa | Ý nghĩa | Vai trò |
|---|---|---|
| `order_id` | Mã đơn hàng | ID |
| `restaurant_name` | Chuỗi cửa hàng | Feature |
| `location` | Chuỗi địa điểm dạng `City, ST` trong file | Feature |
| `order_time` | Thời điểm đặt | Nguồn tạo feature |
| `delivery_time` | Thời điểm giao xong | Leakage, không dùng dự báo |
| `delivery_duration_min` | Thời gian giao thực tế | Leakage, không dùng dự báo |
| `pizza_size` | Size pizza | Feature |
| `pizza_type` | Loại pizza | Feature |
| `toppings_count` | Số topping | Feature |
| `distance_km` | Khoảng cách giao | Feature |
| `traffic_level` | Mức giao thông | Feature |
| `payment_method` | Phương thức thanh toán | Feature |
| `is_peak_hour` | Có phải giờ cao điểm | Feature |
| `is_weekend` | Có phải cuối tuần | Feature |
| `delivery_efficiency_min_per_km` | Hiệu suất sau giao | Leakage, không dùng dự báo |
| `topping_density` | Topping/km hoặc topping/distance | Dẫn xuất; dùng audit/sensitivity |
| `order_month` | Tháng đặt | Feature |
| `payment_category` | Online/offline | Feature |
| `estimated_duration_min` | Thời gian ước tính | Dẫn xuất; compact model không dùng |
| `delay_min` | Chênh thời gian thực tế và ước tính | Leakage, không dùng dự báo |
| `is_delayed` | Đơn trễ hay không | Target |
| `pizza_complexity` | Độ phức tạp pizza | Dẫn xuất; dùng EDA/audit |
| `traffic_impact` | Mức tác động traffic | Dẫn xuất; compact model không dùng |
| `order_hour` | Giờ đặt | Feature |
| `restaurant_avg_time` | Trung bình toàn cục theo nhà hàng | Leakage, không dùng dự báo |
| `pizza_size_score` | Encoding Small=1, Medium=2, Large=3, XL=4 | Active feature |
| `pizza_size_code` | Nhãn 01--04 cho size | Analysis/reporting |
| `pizza_size_label` | Kết hợp code và size | Analysis/reporting |
| `order_period` | Tháng dạng YYYY-MM | Time series |
| `order_weekday` | Thứ trong tuần dạng số | Analysis |
| `order_weekday_name` | Tên thứ trong tuần | Analysis |
| `time_segment` | Nhóm giờ Lunch/Afternoon/Dinner/Late | Analysis |
| `complexity_band` | Nhóm độ phức tạp | Analysis |
| `distance_band` | Nhóm khoảng cách | Analysis |

## Ghi chú chất lượng

- Dữ liệu không có missing value trong bản đã đọc.
- `Marco's Pizza` và `Marco’s Pizza` được chuẩn hóa về cùng một nhãn.
- File chỉ có nhãn `is_delayed`, không có tài liệu SLA. Audit suy luận ranh
  giới nhãn nằm giữa 30 và 35 phút; trên lưới duration bội số 5,
  `delivery_duration_min > 30` và `delivery_duration_min >= 35` đều khớp nhãn
  0 mismatch.
- Tiêu đề dataset ghi 2024-2025, nhưng file hiện có chứa thêm bản ghi năm 2026;
  báo cáo cần nêu rõ theo audit thực tế.
- Có nhiều dấu hiệu synthetic: duration chỉ có 8 mức, estimated duration là
  `2.4 * distance_km`, topping density là công thức từ topping và distance,
  order hour chỉ có 8 giá trị, dữ liệu bắt đầu từ 2024-01-05.
- Mô hình chính dùng feature set compact để giảm cột trùng thông tin; full
  feature set được giữ để sensitivity analysis.
- Riêng EDA có suy ra thêm `city` và `state_code` từ chuỗi `location` dạng
  `City, ST`; đây không phải cột gốc chính thức và không nằm trong schema
  processed chính.
