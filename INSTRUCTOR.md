# Hướng dẫn dành cho giảng viên và người chấm

## Thông tin bài nộp

- Học phần: Hệ hỗ trợ quyết định
- Trường: Đại học Bách khoa Hà Nội
- Khoa: Toán - Tin
- Giảng viên: TS. Lê Hải Hà
- Nhóm: 29
- Đề tài: Hệ hỗ trợ quyết định ứng phó thảm họa từ văn bản và hình ảnh mạng xã hội

**Báo cáo chính:** [BAO_CAO_NHOM_29.pdf](BAO_CAO_NHOM_29.pdf)

## Lộ trình đọc

1. Đọc báo cáo để xem bài toán, cơ sở DSS, dữ liệu, EDA, mô hình và khuyến nghị.
2. Mở [notebook EDA](notebooks/02_eda.ipynb) để kiểm tra phân tích train-only.
3. Mở [notebook modeling](notebooks/03_modeling.ipynb) để kiểm tra so sánh sáu
   model, tuning, baseline, fusion và robustness.
4. Xem [ảnh dashboard](reports/figures/screenshots) hoặc chạy Streamlit.
5. Đối chiếu số liệu với [reports/metrics](reports/metrics).

Không cần tải corpus ảnh để duyệt các kết quả đã lưu.

## Bản đồ nội dung

| Nội dung | Báo cáo | Mã hoặc tệp kiểm chứng |
|---|---|---|
| Bài toán và quyết định | Chương 1–2 | `src/decision_rules.py` |
| Dữ liệu và split | Chương 3 | `src/data_loader.py`, `scripts/audit_data.py` |
| Feature engineering | Chương 3 | `src/text_preprocessing.py`, `src/image_preprocessing.py` |
| EDA, K-Means, Apriori | Chương 4 | `notebooks/02_eda.ipynb`, `src/eda_analysis.py` |
| Sáu classifier | Chương 5 | `src/modeling.py`, validation CSV |
| Tuning classifier | Chương 5 | `src/tune_selected_models.py`, tuning CSV |
| Late Fusion | Chương 5 | `src/evaluate_fusion.py`, `models/fusion_config.json` |
| Robustness và bootstrap | Chương 5 | `scripts/evaluate_robustness.py`, `evaluate_stability.py` |
| Khuyến nghị và giới hạn | Chương 6 | metric theo lớp/event và policy sensitivity |
| Dashboard | Chương 7 | `app/`, screenshot và AppTest |

## Giao thức đánh giá

- Master split là humanitarian; nhãn informative chính thức được join theo
  `(tweet_id, image_id)`.
- TF-IDF chỉ fit train; CLIP là frozen feature extractor.
- Sáu họ model được so sánh trên canonical dev.
- Chỉ họ thắng được tune trên dev.
- Fusion weight, threshold và conflict capacity được chọn trên dev.
- Canonical test chỉ dùng sau khi khóa cấu hình.
- Robust test loại thêm near-duplicate đã duyệt và không tune lại.
- EDA phục vụ thiết kế chỉ dùng train.

## Kết quả khóa

| Chỉ số | Canonical test | Robust test |
|---|---:|---:|
| Informative Accuracy | 0,6948 | 0,6900 |
| Informative F1 | 0,8025 | 0,7978 |
| Informative F2 | 0,9045 | 0,9016 |
| Informative MCC | 0,3325 | 0,3340 |
| Humanitarian Macro-F1 | 0,4005 | 0,3908 |
| Humanitarian Weighted-F1 | 0,5776 | 0,5789 |
| Manual Review F1 | 0,4585 | 0,4558 |

Baseline informative luôn-positive có F2 0,8939 và MCC 0. Majority baseline
humanitarian có Macro-F1 0,0686.

## Dashboard

Chạy local:

```powershell
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

URL mặc định: `http://localhost:8501`.

Deploy Streamlit Community Cloud:

```text
Repository: dafiiscoding/ml-dss
Branch: main
Main file: app/streamlit_app.py
Python: 3.12
```

Chi tiết tại [docs/DEPLOY_STREAMLIT.md](docs/DEPLOY_STREAMLIT.md).

## Kiểm thử

- 41 unit tests.
- Hai notebook thực thi không có cell lỗi.
- 6/6 entrypoint Streamlit AppTest không có exception.
- GitHub Actions kiểm tra repository và bundle cloud.
- Integration test chạy một ảnh thật qua CLIP.
- XeLaTeX compile hai lượt và kiểm tra cảnh báo bố cục.

Chạy:

```powershell
python -m pytest -q
python -m scripts.verify_cloud_bundle
python -m scripts.test_streamlit_cloud
python -m scripts.submission_preflight
```

## Giới hạn cần lưu ý

1. F2 informative chỉ hơn baseline khoảng 0,01; báo cáo không dùng F2 riêng để
   khẳng định chất lượng.
2. Humanitarian cải thiện rõ hơn nhưng lớp missing/vehicle có support rất thấp.
3. Bootstrap theo hàng không thay thế leave-one-event-out.
4. Priority và Risk Score chưa có ground truth nghiệp vụ.
5. Streamlit là prototype hỗ trợ quyết định, không phải hệ thống chỉ huy production.

Tên và MSSV đã được điền. Phần công việc và tỷ lệ đóng góp cần nhóm xác nhận
trước khi chạy `python -m scripts.finalize_submission`.
