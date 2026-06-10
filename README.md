# Hệ hỗ trợ quyết định ứng phó thảm họa đa phương thức

[![Repository preflight](https://github.com/dafiiscoding/ml-dss/actions/workflows/repository-preflight.yml/badge.svg)](https://github.com/dafiiscoding/ml-dss/actions/workflows/repository-preflight.yml)

Đồ án Nhóm 29, học phần **Hệ hỗ trợ quyết định**, Đại học Bách khoa
Hà Nội. Hệ thống sử dụng văn bản và hình ảnh từ CrisisMMD v2.0 để lọc tin
hữu ích, phân loại nhu cầu nhân đạo, tính mức ưu tiên, phân luồng xử lý và
đưa các trường hợp bất đồng đa phương thức vào hàng đợi duyệt thủ công.

## Dành cho người chấm

| Nội dung | Liên kết |
|---|---|
| **Báo cáo PDF 34 trang** | [BAO_CAO_NHOM_29.pdf](BAO_CAO_NHOM_29.pdf) |
| **Hướng dẫn chấm và bản đồ minh chứng** | [INSTRUCTOR.md](INSTRUCTOR.md) |
| Notebook EDA đã chạy | [notebooks/02_eda.ipynb](notebooks/02_eda.ipynb) |
| Notebook mô hình đã chạy | [notebooks/03_modeling.ipynb](notebooks/03_modeling.ipynb) |
| Ảnh chụp dashboard | [reports/figures/screenshots](reports/figures/screenshots) |
| Kết quả định lượng | [reports/metrics](reports/metrics) |
| Tiến độ và kết luận kiểm toán | [docs/PROGRESS.md](docs/PROGRESS.md) |
| Khai báo sử dụng AI | [docs/ai_usage_declaration.md](docs/ai_usage_declaration.md) |

Người đọc có thể đánh giá báo cáo, notebook đã thực thi, bảng kết quả và
screenshot mà **không cần tải bộ ảnh 1,8 GB hoặc chạy lại CLIP**.

## Pipeline

```text
CrisisMMD annotations + images
  -> làm sạch văn bản + TF-IDF
  -> CLIP image embedding 512 chiều
  -> so sánh 6 mô hình cổ điển trên dev
  -> late fusion text/image
  -> Risk Score / Priority / Routing / Manual Review
  -> Streamlit dashboard
```

Pipeline join nhãn informative chính thức theo `(tweet_id, image_id)`, thay vì
suy diễn từ nhãn humanitarian. Mô hình được fit trên train, chọn và tune trên
dev đã loại rò rỉ, sau đó chỉ báo cáo một lần trên test leakage-safe.

## Kết quả chính

| Bài toán | Hệ thống | Kết quả test |
|---|---|---:|
| Lọc informative | Late Fusion | Accuracy 0,7072; F1 0,8079; F2 0,9035 |
| Baseline informative | Luôn dự báo informative | F2 0,8939; MCC 0 |
| Humanitarian 8 lớp | Late Fusion | Macro-F1 0,4005 |
| Baseline humanitarian | Lớp đa số train | Macro-F1 0,0686 |
| Manual Review | Luật conflict có giới hạn công suất | Precision 0,7536 |
| Robust sensitivity | Giữ nguyên model và threshold | F2 0,9006; Macro-F1 0,3908 |

F2 của informative không được diễn giải riêng lẻ vì baseline luôn dự báo
informative đã đạt 0,8939. Trên 2.000 bootstrap phân tầng của robust test,
khoảng tin cậy 95% cho mức tăng F2 so với baseline là `[0,0032; 0,0161]`.
Lợi ích có tính nhất quán nhưng độ lớn thực tế nhỏ.

## Cấu trúc chính

```text
app/                 Dashboard Streamlit
src/                 Loader, tiền xử lý, model, fusion và DSS
scripts/             Audit, đánh giá, tái tạo và kiểm tra bản nộp
notebooks/           Notebook EDA và modeling đã có output
reports/latex/       Nguồn XeLaTeX và PDF biên dịch
reports/metrics/     Metrics, baseline, robustness và DSS audit
reports/figures/     Hình báo cáo và screenshot dashboard
docs/                Dữ liệu, tiến độ, AI declaration và thông tin nhóm
tests/               Unit/integration tests
```

## Chạy dashboard

Môi trường đã xác minh: Python 3.14.2.

```powershell
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

URL local sau khi chạy: `http://localhost:8501`

Đây không phải liên kết public. Bản GitHub cung cấp screenshot và notebook đã
thực thi để người chấm duyệt không cần khởi động dashboard.

Dữ liệu CrisisMMD thô và model/embedding sinh ra không đưa lên Git vì kích
thước. Xem nguồn, cấu trúc thư mục và cách tái tạo tại
[docs/DATA.md](docs/DATA.md).

## Kiểm thử và tái tạo

Kiểm thử nhanh phần code:

```powershell
python -m unittest discover -s tests -v
python -m scripts.submission_preflight
```

Tái tạo toàn bộ pipeline khi đã có dữ liệu:

```powershell
python -m scripts.run_all
```

Trạng thái nghiệm thu hiện tại:

- 37 unit tests pass.
- Hai notebook execute với 0 lỗi.
- 6/6 Streamlit entrypoint AppTest không có exception.
- DSS audit và integration bằng ảnh thật pass.
- Báo cáo XeLaTeX 34 trang, không có overfull/undefined warning.

## Hoàn tất thông tin nhóm

Chỉ cần điền đúng thông tin thực tế trong
[docs/team_info.json](docs/team_info.json), sau đó chạy:

```powershell
python -m scripts.finalize_submission
git add .
git commit -m "Finalize team information"
git push
```

Lệnh hoàn tất sẽ đồng bộ bìa, phụ lục phân công, contribution log, biên dịch
PDF và cập nhật `BAO_CAO_NHOM_29.pdf` ở thư mục gốc.

## Giới hạn được công khai

- Risk Score và Priority là policy prototype vì CrisisMMD không có ground
  truth ưu tiên vận hành; dự án không tuyên bố chúng tối ưu thống kê.
- Robust bootstrap theo hàng không thay thế kiểm thử leave-one-event-out.
- Các lớp hiếm như `missing_or_found_people` còn rất ít mẫu test.
- Thông tin và phân công của ba thành viên còn lại phải do nhóm cung cấp thật.
