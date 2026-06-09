# TIẾN ĐỘ DỰ ÁN

> Cập nhật kiểm toán: 10/06/2026
> Đây là nguồn trạng thái hiện tại. `ROADMAP.md` và `PROJECT_BLUEPRINT.md` chỉ
> là kế hoạch lịch sử.

## Kết luận nghiệm thu theo giai đoạn

| Giai đoạn | Trạng thái thực | Điều kiện đã kiểm tra |
|---|---|---|
| GĐ0 - Dữ liệu/môi trường | Hoàn thành | CrisisMMD thật; 18.082 ảnh tham chiếu tồn tại; 512-D CLIP cache đủ ba split |
| GĐ1 - Nạp và tiền xử lý | Hoàn thành | Join nhãn informative chính thức; 13.608/2.237/2.237; metadata/embedding đúng thứ tự; processed CSV đúng target |
| GĐ2 - EDA | Hoàn thành | EDA train-only; kiểm kê ảnh/hash/kích thước; gallery; CLIP t-SNE; K-Means sweep; Apriori đã lọc luật hiển nhiên; notebook chạy 0 lỗi |
| GĐ3 - Mô hình | Hoàn thành | Sáu model cho bốn nhánh; chọn/tune trên dev sạch; báo cáo một lần trên test sạch; per-class/confusion có đủ |
| GĐ4 - DSS | Hoàn thành ở mức prototype | Risk/Priority/Routing/Manual Review chạy E2E; tám lớp có routing; scenario và sensitivity pass; policy không được nhận là tối ưu thống kê |
| GĐ5 - Dashboard | Hoàn thành ở mức prototype | Năm trang dùng artefact thật; AppTest 0 exception; browser QA và screenshot đạt; không cần frontend riêng theo rubric |
| GĐ6 - Minh chứng | Gần hoàn thành | Notebook, metrics, screenshot, README, AI declaration có đủ; còn thiếu danh tính/phân công thật của ba thành viên |
| GĐ7 - Báo cáo | Gần hoàn thành | XeLaTeX tạo PDF 34 trang, log sạch overfull/undefined và đã render kiểm tra; còn thiếu ba tên/MSSV và phân công nhóm thật |
| GĐ8 - Robustness/validation | Hoàn thành | Dummy baseline, pairwise near-duplicate review, robust sensitivity, 2.000 bootstrap và event/class stability đã có artefact/test |

## Các lỗi audit đã phát hiện và sửa

1. Nhãn informative cũ từng được suy diễn từ `label_top`, làm sai 2.649/18.082
   mẫu. Loader hiện join nhãn chính thức theo `(tweet_id, image_id)`.
2. Split không trùng ID nhưng có ảnh giống hệt theo SHA-256. Đánh giá hiện loại
   48 hàng dev và 68 hàng test có ảnh/text đã xuất hiện ở split trước.
3. EDA cũ từng gộp train/dev/test. EDA hiện chỉ dùng 13.608 mẫu train; kiểm kê
   corpus và audit leakage được trình bày riêng.
4. K-Means có silhouette thấp (0,014-0,024), nên chỉ dùng chẩn đoán cấu trúc chủ
   đề, không tuyên bố tìm được phân cụm tối ưu.
5. Risk Score/Priority không có ground truth. Đây là chính sách minh bạch, được
   kiểm thử scenario/sensitivity, không phải kết quả học máy đã xác thực.
6. F2 informative bị ảnh hưởng mạnh bởi tỷ lệ lớp: baseline luôn dự báo
   `informative` đã đạt F2 0,8939. Fusion đạt 0,9035, chỉ hơn 0,0096 F2; vì vậy
   phải báo cáo thêm Accuracy, F1, Balanced Accuracy/MCC và dummy baseline.
7. Audit pHash 64-bit với bán kính Hamming 4 phát hiện 261 hàng dev/test có ảnh
   gần một ảnh ở split trước nhưng SHA-256 khác nhau. Có 167 trường hợp khoảng
   cách 0, 241 cùng sự kiện và chỉ 123 cùng nhãn humanitarian. Đây là ứng viên
   cần xem ảnh, chưa phải rò rỉ đã xác nhận; evaluation mask chưa thay đổi.
8. Audit text dùng CLIP cosine ở ngưỡng bảo thủ 0,98 và TF-IDF ký tự/từ ở
   ngưỡng 0,80 phát hiện 16 hàng dev/test gần text ở split trước: 6 từ tín hiệu
   semantic và 10 từ tín hiệu lexical. Nhiều cặp là bản tin cùng chủ đề nhưng
   khác số liệu/thời điểm; chỉ 6/16 cùng nhãn humanitarian. Review từng cặp xác
   nhận 5 near-copy (3 dev, 2 test), 7 cặp liên quan nhưng khác nội dung và 4
   template khác đối tượng. Năm near-copy được đưa vào robust mask riêng.
9. Review trực quan 15 contact sheet bao phủ đủ 261 ứng viên ảnh: xác nhận 244
   near-duplicate; giữ lại 10 bản đồ/quan sát khác thời điểm, 4 template khác
   nội dung và 3 pHash collision. Hợp với 5 near-copy text tạo 248 loại trừ bổ
   sung (1 hàng trùng cả hai nguồn), nhưng được lưu ở robust mask riêng.
10. Sensitivity test trên robust mask giữ nguyên model, trọng số và threshold
    đã chọn từ dev canonical. Informative F2 giảm 0,9035 -> 0,9006, MCC giảm
    0,3602 -> 0,3592; humanitarian Macro-F1 giảm 0,4005 -> 0,3908; Manual
    Review F1 giảm 0,4687 -> 0,4665. Không có dấu hiệu kết quả phụ thuộc mạnh
    vào các near-duplicate đã xác nhận, nhưng lớp hiếm vẫn dao động đáng kể.
11. Bootstrap phân tầng 2.000 lần trên robust test cho F2 CI 95%
    [0,8938; 0,9068], paired gain so với always-informative [0,0032; 0,0161].
    Humanitarian Macro-F1 CI [0,3636; 0,4208], gain so với majority
    [0,2936; 0,3508]. Gain informative khác 0 theo row-bootstrap nhưng chỉ
    khoảng 0,01, nên không được thổi phồng ý nghĩa vận hành.
12. Theo event, informative F2 thấp nhất ở `srilanka_floods` (0,8377);
    humanitarian Macro-F1 thấp nhất ở `iraq_iran_earthquake` (0,3458, 61
    mẫu). Ba lớp injured/dead, missing/found và vehicle damage còn dưới 30
    mẫu; không lớp support đủ lớn nào dịch F1 quá 0,05 giữa hai mask.

## Số liệu đã khóa

- Corpus: 18.082 hàng; 17.777 hash ảnh duy nhất; 305 hàng ảnh lặp theo exact hash.
- Evaluation leakage-safe: dev 2.189 hàng; test 2.169 hàng.
- Near-duplicate image candidates: dev 120; test 141; tổng 261; chưa loại mẫu.
- Near-duplicate text candidates: dev 7; test 9; tổng 16; chưa loại mẫu.
- Text review: 5 near-copy xác nhận (dev 3, test 2); canonical mask chưa đổi.
- Image review: 244 near-duplicate xác nhận (dev 109, test 135).
- Robust evaluation: dev 2.078; test 2.032; loại thêm 248 hàng đã xác minh.
- Robust sensitivity, không tune lại: informative Accuracy 0,7018; F1 0,8029;
  F2 0,9006; MCC 0,3592. Humanitarian Macro-F1 0,3908; Weighted F1 0,5789.
  Manual Review Precision 0,7529; Recall 0,3380; F1 0,4665.
- Informative Late Fusion: threshold 0,38; text/image 0,70/0,30;
  Accuracy 0,7072; Balanced Accuracy 0,6136; F1 0,8079; F2 0,9035;
  MCC 0,3602. Always-positive baseline có F2 0,8939 và MCC 0.
- Humanitarian Late Fusion: text/image 0,55/0,45;
  Accuracy 0,5569; Macro-F1 0,4005.
- Manual Review: threshold 0,54; Precision 0,7536; Recall 0,3401;
  F1 0,4687; review rate 25,63%.
- Robust bootstrap: F2 gain so với dummy CI [0,0032; 0,0161]; humanitarian
  Macro-F1 gain CI [0,2936; 0,3508]. Đây là row-bootstrap, không thay thế
  leave-one-event-out.
- Test DSS: 621 Low, 1.493 Medium, 55 High; 556 ca Manual Review.
- Lớp yếu nhất: `missing_or_found_people`, F1 0,0714 trên 5 mẫu test sạch.

## EDA và suy luận đã có

- Mất cân bằng train lớn nhất khoảng 219:1, nên Macro-F1 và F2 được ưu tiên.
- Độ dài informative/not-informative gần nhau, nên độ dài không phải tín hiệu đủ.
- K-Means không có elbow rõ; `k=8` chỉ là đối chiếu với taxonomy tám lớp.
- Apriori giữ các luật có ý nghĩa sau khi loại quan hệ hashtag/từ trùng nghĩa.
- Disagreement text-image cao ở lớp khẩn cấp là tín hiệu cần human review,
  không phải bằng chứng rằng ảnh sai.
- Mỗi kết luận trên đã được nối với quyết định mô hình/DSS trong Chương 4-5.

## Frontend

Streamlit là đủ cho sản phẩm học thuật: có KPI, biểu đồ, bộ lọc, model
evaluation, single-case inference, priority queue và export. React/frontend tách
riêng chỉ cần khi có yêu cầu production như API độc lập, phân quyền hoặc tải lớn.

Dashboard hiện chạy tại `http://localhost:8501`.

## Việc duy nhất còn cần người dùng cung cấp

- Họ tên và MSSV của ba thành viên còn lại.
- Phần việc thực tế của cả bốn người, kèm tỷ lệ đóng góp nếu môn yêu cầu.

Sau khi có thông tin, sửa bìa/phụ lục rồi chạy `xelatex main.tex` hai lần trong
`reports/latex/`.

## Kiểm thử chuẩn

Nghiệm thu cuối hiện tại: 37 unit tests pass; 2 notebook execute 0 lỗi;
6/6 Streamlit entrypoint AppTest không exception; DSS audit và integration ảnh
thật pass; PDF 34 trang biên dịch hai lượt, không có overfull/undefined warning.

```powershell
python -m unittest discover -s tests -v
python -m scripts.audit_data
python -m scripts.audit_near_duplicate_images
python -m scripts.build_image_duplicate_review_artifacts
python -m scripts.review_near_duplicate_images
python -m scripts.audit_near_duplicate_texts
python -m scripts.review_near_duplicate_texts
python -m scripts.build_robust_evaluation_mask
python -m scripts.evaluate_baselines
python -m src.evaluate_fusion
python -m scripts.evaluate_robustness
python -m scripts.evaluate_stability
python -m scripts.audit_dss
python -m src.test_integration
streamlit run app/streamlit_app.py
```
