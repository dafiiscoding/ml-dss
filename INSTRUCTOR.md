# Hướng dẫn dành cho giảng viên và người chấm

## Thông tin bài nộp

- Học phần: Hệ hỗ trợ quyết định
- Trường: Đại học Bách khoa Hà Nội
- Khoa: Toán - Tin
- Giảng viên: TS. Lê Hải Hà
- Nhóm: 29
- Đề tài: Hệ hỗ trợ quyết định ứng phó thảm họa từ văn bản và hình ảnh
  mạng xã hội

**Báo cáo chính:** [BAO_CAO_NHOM_29.pdf](BAO_CAO_NHOM_29.pdf)

Thông tin ba thành viên còn lại và phân công thực tế đang để placeholder có
chủ đích. Nhóm cần điền một lần tại
[`docs/team_info.json`](docs/team_info.json) trước khi nộp chính thức.

## Lộ trình đọc nhanh

1. Đọc báo cáo PDF để xem bài toán, phương pháp, kết quả và kết luận.
2. Mở [notebook EDA](notebooks/02_eda.ipynb) để kiểm tra phân tích chạy thật.
3. Mở [notebook modeling](notebooks/03_modeling.ipynb) để kiểm tra so sánh mô
   hình, baseline, late fusion và robustness.
4. Xem [ảnh dashboard](reports/figures/screenshots) nếu không cài dữ liệu/model.
5. Đối chiếu các con số với [reports/metrics](reports/metrics).

Thời gian duyệt nhanh toàn bộ minh chứng chính khoảng 15-20 phút và không cần
tải corpus ảnh CrisisMMD.

Dashboard là ứng dụng Streamlit chạy local tại `http://localhost:8501` sau khi
thực hiện lệnh ở phần cuối tài liệu. Đây không phải URL public; screenshot đã
được lưu để việc chấm trên GitHub không phụ thuộc vào một dịch vụ triển khai.

## Bản đồ tiêu chí và minh chứng

| Nội dung đánh giá | Phần báo cáo | Minh chứng trực tiếp |
|---|---|---|
| Bài toán và quyết định cần hỗ trợ | Chương 1 | `src/decision_rules.py`, dashboard |
| Cơ sở lý thuyết môn học | Chương 2 | DT, NB, k-NN, SVM, RF, LR, K-Means, Apriori |
| Dữ liệu và tiền xử lý | Chương 3 | `src/data_loader.py`, `src/preprocessing.py`, data audit |
| EDA và insight | Chương 4 | `notebooks/02_eda.ipynb`, `reports/figures/` |
| Feature engineering | Chương 3-5 | TF-IDF n-gram, CLIP 512-D, conflict features |
| So sánh/tune mô hình | Chương 5-6 | `notebooks/03_modeling.ipynb`, validation CSV |
| Late fusion đa phương thức | Chương 5-6 | `src/fusion.py`, fusion tuning/test metrics |
| DSS prescriptive | Chương 5-7 | Risk, Priority, Routing, Manual Review audit |
| Sản phẩm dashboard | Chương 7 | `app/`, screenshot và Streamlit AppTest |
| Robustness và leakage | Chương 3, 6 | exact hash, pairwise review, robust mask, bootstrap |
| Khai báo AI/phân công | Phụ lục | `docs/ai_usage_declaration.md`, contribution log |

## Artefact quan trọng

- Báo cáo: [`BAO_CAO_NHOM_29.pdf`](BAO_CAO_NHOM_29.pdf)
- Nguồn LaTeX: [`reports/latex`](reports/latex)
- Tiến độ đã kiểm toán: [`docs/PROGRESS.md`](docs/PROGRESS.md)
- Mô tả dữ liệu và tái tạo: [`docs/DATA.md`](docs/DATA.md)
- EDA: [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb)
- Modeling: [`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb)
- Metrics: [`reports/metrics`](reports/metrics)
- Hình và dashboard: [`reports/figures`](reports/figures)
- Unit tests: [`tests`](tests)
- AI declaration: [`docs/ai_usage_declaration.md`](docs/ai_usage_declaration.md)

## Kết quả khóa để đối chiếu

| Chỉ số | Canonical test | Robust test |
|---|---:|---:|
| Informative Accuracy | 0,7072 | 0,7018 |
| Informative F1 | 0,8079 | 0,8029 |
| Informative F2 | 0,9035 | 0,9006 |
| Informative MCC | 0,3602 | 0,3592 |
| Humanitarian Macro-F1 | 0,4005 | 0,3908 |
| Manual Review F1 | 0,4687 | 0,4665 |

Các kết quả canonical dùng test đã loại exact duplicate xuyên split. Robust
test loại thêm các near-duplicate được review theo cặp, không tune lại model,
trọng số fusion hoặc threshold.

## Kiểm soát liêm chính và rò rỉ

- Dùng CrisisMMD v2.0 thật, không dùng kết quả synthetic trong báo cáo.
- Nhãn informative được join từ annotation chính thức theo
  `(tweet_id, image_id)`.
- EDA phục vụ quyết định mô hình chỉ dùng train.
- Fit trên train, chọn/tune trên dev, báo cáo test sau khi khóa cấu hình.
- Exact image/text duplicate ở split trước bị loại khỏi dev/test metrics.
- 261 ứng viên ảnh và 16 ứng viên text gần trùng đã được review theo cặp.
- Robust mask riêng loại 248 hàng được xác nhận; canonical mask không bị sửa
  hậu nghiệm.
- Baseline luôn informative và majority-class được đặt cạnh kết quả fusion.

## Kiểm thử đã hoàn thành

- 37 unit tests.
- Hai notebook execute, 0 lỗi.
- 6/6 trang/entrypoint Streamlit AppTest, 0 exception.
- DSS scenario/sensitivity audit pass.
- Integration test sử dụng ảnh thật pass.
- XeLaTeX compile hai lượt; PDF 34 trang, không có cảnh báo bố cục nghiêm trọng.
- GitHub Actions chạy preflight cho bộ file được nộp.

## Chạy kiểm tra

Không có dữ liệu thô vẫn chạy được kiểm tra cấu trúc bản nộp:

```powershell
python -m scripts.submission_preflight
```

Khi đã đặt CrisisMMD theo [`docs/DATA.md`](docs/DATA.md):

```powershell
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.run_all
streamlit run app/streamlit_app.py
```

## Giới hạn cần lưu ý khi chấm

1. Risk Score/Priority là policy prototype minh bạch, chưa có ground truth để
   chứng minh tối ưu thống kê.
2. F2 informative chỉ hơn baseline luôn-positive khoảng 0,01; báo cáo không
   thổi phồng mức tăng này.
3. Humanitarian Macro-F1 có cải thiện rõ so với majority baseline nhưng ba lớp
   hiếm còn support thấp.
4. Bootstrap theo hàng đo bất định trên tập hiện có, không chứng minh khả năng
   tổng quát sang một thảm họa hoàn toàn mới.
