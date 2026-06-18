# Hướng dẫn nội dung slide

File này dùng để thuyết trình theo `slides/PIZZA_DSS_SLIDE_DECK.pdf`. Source
chỉnh sửa là `slides/pizza_dss_slides.tex`.

Trước khi thêm slide mới, đọc `docs/SCOPE_PRIORITIZATION.md`. Slide chính chỉ
nên giữ tầng A/lõi; các phần forecast chi tiết, recommendation rules, K-Means,
association, brand/state detail nên để backup hoặc notebook.

## 1. Vai trò của slide

Slide là bản kể ngắn của báo cáo trong khoảng 10-12 phút. Mục tiêu là dẫn người
nghe theo logic DSS:

1. Có bài toán quyết định thật.
2. Dữ liệu có vấn đề và đã được kiểm soát.
3. Phân tích tìm insight và giới hạn.
4. Mô hình được chọn/tune đúng protocol.
5. Dự báo được chuyển thành hành động vận hành.
6. Project có dashboard, Power BI pack và khả năng tái lập.

## 2. Phân bổ thời gian gợi ý

| Cụm slide | Thời lượng | Mục tiêu |
|---|---:|---|
| 1-3 | 1 phút | Mở bài, bài toán quyết định |
| 4-7 | 2 phút | Dữ liệu, leakage, synthetic/forensics |
| 8-12 | 2 phút | EDA, behavior, forecast, rubric môn học |
| 13-17 | 3 phút | Feature engineering, model, tuning, test |
| 18-21 | 2 phút | DSS, risk transparency, assignment, dashboard/Power BI |
| 22-25 | 1-2 phút | Tái lập, câu hỏi phản biện, giới hạn, kết luận |

## 3. Slide-by-slide speaking notes

| Slide | Nội dung chính | Cách nói ngắn |
|---|---|---|
| 1. Title | Đề tài, nhóm, học phần | Nêu đây là DSS dự báo và ưu tiên đơn pizza có nguy cơ trễ. |
| 2. Thông điệp chính | 3 điểm khóa | Nhấn mạnh dự án đủ quy trình nhưng trung thực với dữ liệu synthetic. |
| 3. Bài toán quyết định | Ai dùng, quyết định gì, output gì | Người quản lý cần biết đơn nào phải ưu tiên trước khi giao. |
| 4. Dữ liệu và leakage | Dataset, target, cột cấm | Duration/delay là thông tin sau giao, dùng sẽ làm mô hình gian lận. |
| 5. Dữ liệu rác | Cách biến điểm yếu thành audit | Không né data rác; dùng forensics để chứng minh và đặt caveat. |
| 6. Truy ngược generator | Công thức tất định, duration grid, threshold | `is_delayed` được suy luận từ dữ liệu, không giả định SLA. |
| 7. EDA lớp thiểu số | Delay rate ~21% | Accuracy một mình không đủ vì đoán on-time cũng cao. |
| 8. Traffic/distance | Driver rủi ro chính | Traffic và distance là tín hiệu vận hành rõ nhất. |
| 9. Customer behavior | Size/type/location | Behavior dùng để hiểu demand, không kết luận thị trường thật. |
| 10. Forecasting/staffing | Demand forecast, peak hour | Forecast là demo planning; staffing peak 19h. |
| 11. Trend sở thích | Share size/type forecast | Trả lời câu hỏi trend, nhưng caveat synthetic. |
| 12. Kiến thức môn học | Mapping rubric | Chỉ nhanh các mảng: supervised, clustering, rules, testing, BI. |
| 13. Full vs compact | Feature engineering | Compact bỏ feature deterministic/trùng thông tin để chống redundancy. |
| 14. So sánh model | 6 classifier dev | LR thắng dev F2 và dễ giải thích cho DSS. |
| 15. Tuning | GridSearchCV LR | CV chọn `C=0.3`, nhưng dev kém default nên giữ `C=1.0`. |
| 16. F-beta/stability | Threshold transfer + 100-run audit | Dev-best F2 giảm FN nhưng tăng FP; 100-run cho thấy điểm cao ổn định tương đối. |
| 17. Test | Metric + confusion matrix | Báo test một lần sau khi khóa; F2/Recall cao, kèm caveat data. |
| 18. DSS layer | Risk Score -> Priority -> Action | Mô hình chỉ là một phần; DSS biến xác suất thành hành động. |
| 19. Risk transparency | Component weight + normalization | Risk Score không phải hộp đen; từng thành phần cộng ra điểm cuối. |
| 20. Vận tải | Assignment scenario | Dùng đơn thật, driver/fleet giả lập vì dataset không có tài xế. |
| 21. Dashboard | Streamlit + Power BI pack | Streamlit demo trực tiếp; Power BI có CSV/DAX/spec để dựng `.pbix`. |
| 22. Minh chứng | run_all, tests, notebooks | Nhấn mạnh tái lập: 18/18 bước, 30/30 tests. |
| 23. Câu hỏi phản biện | Threshold, leakage, Bayes, F-beta, stability, assignment | Dùng làm slide phòng thủ khi thầy hỏi trực tiếp. |
| 24. Giới hạn | Synthetic, forecast demo, no production claim | Nói rõ giới hạn trước khi bị hỏi. |
| 25. Kết luận | Đủ quy trình DSS nhỏ gọn | Chốt: dự án nhỏ nhưng có đủ analysis, prediction, decision, BI. |

## 4. Khi muốn chỉnh slide

- Sửa source: `slides/pizza_dss_slides.tex`.
- Hình lấy từ `reports/figures/`.
- Nếu thêm hình, ưu tiên các hình đã đánh dấu `[SLIDE] nên thêm` trong
  `docs/WORKFLOW_PRESENTATION_GUIDE.md`.
- Không sửa trực tiếp PDF. PDF được build lại từ `.tex`.

Build lại:

```powershell
cd pizza_delivery_dss
.\.venv\Scripts\python.exe -m scripts.build_slides_pdf
```

File nộp chính là `slides/PIZZA_DSS_SLIDE_DECK.pdf`.

## 5. Câu trả lời ngắn khi bị hỏi

- Vì sao chỉ tune LR? Vì chọn họ mô hình trước rồi tune họ thắng; tune cả 6 model
  trên dataset nhỏ dễ overfit dev và không cần thiết.
- Vì sao không dùng duration? Vì duration biết sau giao, audit cho thấy phục
  dựng nhãn gần như trực tiếp.
- Forecast sai số lớn thì sao? Không cố tune để làm đẹp MAPE; forecast chỉ là
  demo time-series/staffing vì chuỗi ngắn, partial và synthetic.
- Recommendation là gì? Là rule/popularity/context heuristic, không phải hệ cá
  nhân hóa vì không có user history/rating.
- Power BI có chưa? Có data pack để dựng dashboard; chưa sinh `.pbix` tự động.
- Kết quả có dùng ngoài đời được không? Không claim như vậy, vì dataset synthetic.
