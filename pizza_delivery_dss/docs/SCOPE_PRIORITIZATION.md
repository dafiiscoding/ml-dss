# Scope Prioritization - Giảm Tải Báo Cáo/Slide

Tài liệu này dùng **trước Pha 2** để tránh project bị quá to so với yêu cầu môn
học. Nguyên tắc: pipeline/notebook/artifact có thể giữ đầy đủ, nhưng report và
slide phải kể một câu chuyện gọn, đúng trọng tâm DSS, không lan man.

## 1. Nguyên Tắc Trình Bày

- **Report/slide là deliverable chính cho thầy**: phải rõ bài toán, dữ liệu,
  model, quyết định, dashboard, giới hạn.
- **Notebook là minh chứng quy trình**: để đủ module và trả lời khi thầy hỏi
  sâu, không cần bê hết vào slide.
- **Dữ liệu synthetic là trục giải thích bắt buộc**: điểm cao, forecast yếu,
  recommendation đơn giản đều phải đọc dưới caveat này.
- **Không overclaim**: forecast/recommendation/transport chỉ là prototype học
  thuật, không phải hệ thống sản xuất.
- **Mỗi nội dung trong slide phải trả lời được "vậy quyết định gì?"** Nếu không
  trả lời được, đẩy xuống phụ lục/notebook.

## 2. Phân Tầng Nội Dung

| Tầng | Mục tiêu | Dùng ở report | Dùng ở slide | Dùng ở notebook/artifact |
|---|---|---|---|---|
| A - Lõi chấm điểm | Phải thấy rõ để hiểu project | Chương chính | Slide chính | Có đầy đủ |
| B - Minh chứng đủ môn | Cho thấy đủ kỹ thuật/rubric | Tóm tắt ngắn hoặc bảng phụ | 1 dòng/không đưa | Có đầy đủ |
| C - Mở rộng/phụ lục | Hữu ích nếu bị hỏi sâu | Phụ lục/caveat | Không đưa hoặc backup | Có đầy đủ |

## 3. Bảng Quyết Định Giữ/Giảm

| Mảng | Tầng | Vai trò thật | Report nên làm gì | Slide nên làm gì | Lý do |
|---|---|---|---|---|---|
| Bài toán DSS | A | Định nghĩa người dùng, quyết định, output | Giữ rõ ở mở đầu + 8 câu hỏi | 1-2 slide đầu | Đây là xương sống môn DSS |
| Data audit + leakage | A | Chứng minh làm đúng bài toán trước-giao-hàng | Giữ chương chính | 1 slide data/leakage | Nếu không nói rõ sẽ bị nghi model gian lận |
| Target threshold inference | A | Trả lời "sao biết trễ là bao nhiêu phút?" | Giữ rõ, có bảng/hình | 1 ý trong slide data/forensics | Đây là câu hỏi dễ bị thầy bắt |
| Synthetic/data forensics | A | Biến data rác thành phân tích trung thực | Giữ nhưng gọn: công thức, duration grid, caveat | 1 slide | Giải thích vì sao điểm cao và forecast/RCM yếu |
| EDA traffic/distance/delay | A | Insight chính dẫn tới risk và model | Giữ 2-3 hình/bảng chính | 1 slide | Gắn trực tiếp với quyết định vận hành |
| Feature engineering compact | A | Chống redundancy/leakage | Giữ ngắn trong data/model | 1 dòng trong slide model | Quan trọng nhưng không nên quá dài |
| Model comparison 6 classifier | A | Đủ supervised + chọn model | Giữ bảng dev + baseline/test | 1 slide | Bắt buộc theo rubric |
| Bayes không được chọn | A/B | Trả lời "sao không Bayes?" | 1 đoạn trong model | 1 bullet nếu còn chỗ | Cần nói đã thử, không phải bỏ qua |
| F-beta/threshold analysis | A | Giải thích metric/ngưỡng, tránh chọn tùy tiện | Thêm bảng/hình ngắn | 1 bullet hoặc backup slide | Nên làm vì liên quan chọn F2 |
| 100-run stability | A | Trả lời "có ăn may split không?" | Thêm hình phân phối + kết luận | 1 slide hoặc backup | Nâng độ chắc của model |
| Risk Score formula | A | Chuyển ML thành DSS decision policy | Giữ công thức + breakdown | 1 slide riêng | Đây là phần DSS nhất của project |
| Priority queue/action | A | Output vận hành thật | Giữ bảng queue/action | 1 slide DSS | Cho thấy hệ thống ra quyết định |
| Streamlit dashboard | A | Demo người dùng cuối | Mô tả tab + vai trò | 1 slide screenshot/role | Thầy nhìn trực tiếp, cần show-off rõ |
| Power BI pack | B | BI/dashboard theo yêu cầu | Tóm tắt data pack + guide | 1 bullet trong dashboard slide | Có pack, nhưng không phải trọng tâm nếu không nộp `.pbix` |
| Transport assignment | B | Optimization/prescriptive demo | Giữ ngắn, tách thật/giả lập/cost | 1 slide nếu đủ thời gian | Đủ môn tối ưu, nhưng phải tránh claim thật |
| Hypothesis testing | B | Đủ thống kê/EDA | Bảng tóm tắt significant + caveat | Không cần slide riêng | Hỗ trợ insight, không phải trục chính |
| K-Means clustering | B/C | Đủ clustering module | 1 đoạn/appendix | Không đưa slide chính | Không gắn mạnh với quyết định cuối |
| Association rules | B/C | Đủ association analysis | 1 đoạn/appendix | Không đưa slide chính | Có ích nhưng dễ lan man |
| Customer behavior size/type | B | Business insight | Giữ ngắn: top size/type | 1 bullet hoặc nhỏ trong slide EDA | Có giá trị nhưng không phải target chính |
| Forecast demand | B/C | Minh họa time series/staffing | Giữ ngắn + caveat MAPE 45,7% | 1 bullet, không làm slide chính nếu thiếu thời gian | Sai số lớn, không nên làm trọng tâm |
| Preference trend forecast | C | Trả lời câu hỏi trend nhưng synthetic | Đưa phụ lục hoặc 1 đoạn caveat | Không đưa slide chính | Dễ overclaim thị hiếu giả |
| Recommendation rules | C | Popularity/context heuristic | Đưa phụ lục hoặc bảng nhỏ | Không đưa slide chính | Không phải recommender system thật |
| Brand/location/state detail | C | Data artifact/EDA phụ | Phụ lục/caveat | Không đưa slide chính | Dễ bị hiểu sai là chất lượng brand thật |

## 4. Forecast Sai Số Lớn: Có Fine-tune Không?

**Không nên "fine-tune" forecast để cố làm đẹp MAPE.** Lý do:

- Dữ liệu chỉ 1.004 đơn, theo tháng có ít điểm, lại có tháng partial và năm
  2026 trong file 2024-25.
- Dataset synthetic nên pattern thời gian không đáng tin để tối ưu sâu.
- Hiện đã có so sánh phương pháp đơn giản:
  - `seasonal_naive_prior_year`: MAE 12,211; MAPE 45,7%.
  - `moving_average_3`: MAE 12,048; MAPE 59,0%.
- MAPE cao là **kết luận phân tích**, không phải lỗi cần che. Nên nói forecast
  chỉ là demo planning/staffing để phủ time series trong môn học.

Report/slide nên viết:

> Forecast được giữ ở mức minh họa quy trình planning. Sai số MAPE 45,7% cho
> thấy dữ liệu synthetic/partial không đủ đáng tin để triển khai dự báo nhu cầu
> sản xuất. Nhóm không tiếp tục tune sâu để tránh overfit vào chuỗi ngắn.

Nếu muốn mở rộng nhẹ, chỉ nên thêm:

- so sánh 2-3 baseline forecast đơn giản;
- ghi rõ seasonal-naive được chọn vì dễ giải thích;
- không quảng bá forecast là kết quả kinh doanh chính.

## 5. Recommendation Có Phải Hệ RCM Không?

Không nên gọi là recommender system phức tạp. Tên đúng hơn:

**Rule-based / popularity-based recommendation heuristic**.

Cách hiểu:

- Dựa trên món/size/type có tần suất cao, delay rate, context như traffic hoặc
  restaurant.
- Không có user history thật, rating, session, click, purchase sequence.
- Không có collaborative filtering, matrix factorization, neural recommender.
- Vì vậy đây là **business suggestion rules**, không phải hệ cá nhân hóa.

Report/slide nên viết:

> Recommendation trong project là heuristic dựa trên popularity/context để minh
> họa cách chuyển phân tích behavior thành gợi ý vận hành. Đây không phải hệ
> recommender cá nhân hóa vì dataset không có lịch sử người dùng/rating.

## 6. Outline Report Rút Gọn Khuyến Nghị

Nên giữ report theo 7 chương chính, nhưng giảm độ dài các phần B/C:

1. Giới thiệu bài toán DSS và 8 câu hỏi bắt buộc.
2. Dữ liệu, target, leakage, feature contract.
3. Data realism/forensics: vì sao data synthetic và cách xử lý trung thực.
4. EDA chính: delay distribution, traffic/distance, customer behavior ngắn.
5. Modeling: 6 classifier, Bayes không chọn, tuning LR, F-beta/stability, test.
6. DSS: Risk Score, Priority Queue, Streamlit, Power BI, transport assignment.
7. Kết luận, giới hạn, phụ lục.

Đẩy xuống phụ lục/notebook:

- K-Means chi tiết.
- Association rules chi tiết.
- Preference trend forecast chi tiết.
- Brand/state/location detail.
- Recommendation rules đầy đủ.

## 7. Outline Slide Rút Gọn Khuyến Nghị

10-12 phút nên giữ khoảng 12-14 slide:

1. Bài toán và quyết định cần hỗ trợ.
2. Data + leakage + target threshold.
3. Synthetic forensics: vì sao phải caveat.
4. EDA chính: delayed imbalance, traffic/distance.
5. Model comparison + vì sao không Bayes.
6. F-beta/tuning/stability: vì sao kết quả không chỉ ăn may.
7. Test result + baseline.
8. Risk Score formula + Priority.
9. Delay Queue + Recommended Action.
10. Streamlit dashboard: người dùng tương tác gì.
11. Power BI + transport assignment: nói ngắn là BI/prescriptive demo.
12. Giới hạn + kết luận.

Backup/appendix slide nếu cần:

- Forecast/staffing.
- Recommendation heuristic.
- Association/K-Means.
- Detailed forensics.

## 8. Quy Tắc Khi Sửa Các Pha Sau

- Không xóa artifact đã có nếu không cần; chỉ giảm mức độ xuất hiện trong
  report/slide.
- Mỗi phần B/C trong report tối đa 1-2 đoạn + 1 bảng/hình nếu thật sự cần.
- Slide chính không nên có quá 1 bảng lớn hoặc 1 hình chính.
- Nếu một mục không giúp trả lời quyết định vận hành, đưa vào appendix/notebook.
- Luôn giữ caveat synthetic ở cạnh forecast, recommendation, brand và transport.
