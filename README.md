# Hệ hỗ trợ quyết định ứng phó thảm họa đa phương thức

[![Repository preflight](https://github.com/dafiiscoding/ml-dss/actions/workflows/repository-preflight.yml/badge.svg)](https://github.com/dafiiscoding/ml-dss/actions/workflows/repository-preflight.yml)
[![Streamlit cloud smoke](https://github.com/dafiiscoding/ml-dss/actions/workflows/streamlit-cloud-smoke.yml/badge.svg)](https://github.com/dafiiscoding/ml-dss/actions/workflows/streamlit-cloud-smoke.yml)

Đồ án Nhóm 29, học phần **Hệ hỗ trợ quyết định**, Đại học Bách khoa Hà Nội.
Hệ thống dùng văn bản và hình ảnh CrisisMMD v2.0 để sàng lọc thông tin, phân
loại nhu cầu nhân đạo, tạo mức ưu tiên, phân luồng đội xử lý và đưa trường hợp
bất đồng vào hàng đợi xác minh.

## Tài liệu chính

| Nội dung | Liên kết |
|---|---|
| Báo cáo PDF | [BAO_CAO_NHOM_29.pdf](BAO_CAO_NHOM_29.pdf) |
| Hướng dẫn dành cho giảng viên | [INSTRUCTOR.md](INSTRUCTOR.md) |
| Notebook EDA đã thực thi | [notebooks/02_eda.ipynb](notebooks/02_eda.ipynb) |
| Notebook mô hình đã thực thi | [notebooks/03_modeling.ipynb](notebooks/03_modeling.ipynb) |
| Kết quả định lượng | [reports/metrics](reports/metrics) |
| Ảnh dashboard | [reports/figures/screenshots](reports/figures/screenshots) |
| Hướng dẫn dữ liệu | [docs/DATA.md](docs/DATA.md) |
| Hướng dẫn deploy | [docs/DEPLOY_STREAMLIT.md](docs/DEPLOY_STREAMLIT.md) |

Người đọc có thể kiểm tra báo cáo, notebook, metric và screenshot mà không cần
tải corpus ảnh CrisisMMD khoảng 1,8 GB.

## Phương pháp

```text
CrisisMMD text + image
  -> audit nhãn, split và duplicate
  -> TF-IDF 2.000 chiều + CLIP frozen embedding 512 chiều
  -> so sánh 6 classifier trên dev
  -> tune classifier thắng trên dev
  -> late fusion và threshold chọn trên dev
  -> test canonical + robust sensitivity + bootstrap
  -> Risk Score / Priority / Routing / Manual Review
  -> Streamlit dashboard
```

CLIP chỉ là bộ trích đặc trưng tiền huấn luyện với trọng số giữ nguyên. Dự án
không fine-tune CLIP; các classifier cổ điển phía sau được huấn luyện bằng
CrisisMMD.

## Kết quả chính

| Bài toán | Hệ thống | Canonical test | Robust test |
|---|---|---:|---:|
| Informative Accuracy | Late Fusion | 0,6948 | 0,6900 |
| Informative F1 | Late Fusion | 0,8025 | 0,7978 |
| Informative F2 | Late Fusion | 0,9045 | 0,9016 |
| Informative MCC | Late Fusion | 0,3325 | 0,3340 |
| Humanitarian Macro-F1 | Late Fusion | 0,4005 | 0,3908 |
| Manual Review F1 | Conflict rule | 0,4585 | 0,4558 |

Baseline luôn dự báo informative đạt F2 0,8939, nên F2 không được diễn giải
độc lập. Humanitarian majority baseline chỉ đạt Macro-F1 0,0686. Bootstrap
2.000 lần trên robust test cho CI 95% của F2 gain là `[0,0054; 0,0163]` và
Macro-F1 gain là `[0,2936; 0,3508]`.

## Dashboard Streamlit

Năm trang:

1. Overview
2. EDA
3. Model Evaluation
4. Single-Case Demo
5. Priority Queue

Chạy local:

```powershell
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

URL mặc định: `http://localhost:8501`.

### Deploy Streamlit Community Cloud

Repository đã có bundle cloud và workflow smoke test trên Python 3.12.

```text
Repository: dafiiscoding/ml-dss
Branch: main
Main file: app/streamlit_app.py
Python: 3.12
Secrets: không yêu cầu
```

Kiểm tra trước deploy:

```powershell
python -m scripts.verify_cloud_bundle
python -m scripts.test_streamlit_cloud
```

## Tái lập thực nghiệm

Kiểm thử nhanh:

```powershell
python -m pytest -q
python -m scripts.submission_preflight
```

Tái tạo đầy đủ khi đã đặt dữ liệu theo [docs/DATA.md](docs/DATA.md):

```powershell
python -m scripts.run_all
```

Quy trình đầy đủ sẽ trích lại CLIP embedding cho 18.082 ảnh, huấn luyện/tune
model, đánh giá fusion, dựng notebook và cache dashboard.

## Cấu trúc repository

```text
app/                 Ứng dụng Streamlit
src/                 Loader, tiền xử lý, mô hình, fusion và DSS
scripts/             Audit, tái tạo, đánh giá và kiểm tra deploy
notebooks/           EDA và modeling có output
reports/latex/       Nguồn XeLaTeX
reports/metrics/     Bảng kết quả và robustness
reports/figures/     Hình báo cáo và screenshot
data/processed/      Split chuẩn hóa và cache dashboard
models/              Model inference dùng local/cloud
tests/               Unit và integration tests
```

## Giới hạn

- Dữ liệu thuộc bảy thảm họa năm 2017; chưa chứng minh tổng quát sang event mới.
- Ba lớp khẩn cấp có support test rất thấp.
- Risk Score và Priority là policy minh bạch, chưa có ground truth tối ưu.
- Bản cloud là prototype trình diễn, chưa có authentication và kiểm thử tải.

## Hoàn tất phân công nhóm

Tên và MSSV đã được điền. Nhóm cần xác nhận phần việc và tỷ lệ đóng góp trong
[docs/team_info.json](docs/team_info.json), sau đó chạy:

```powershell
python -m scripts.finalize_submission
```

Lệnh này đồng bộ bảng phân công, biên dịch PDF và chạy kiểm tra bản nộp cuối.
