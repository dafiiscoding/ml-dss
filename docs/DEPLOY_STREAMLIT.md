# Triển khai Streamlit Community Cloud

## Cấu hình đã chuẩn bị

- Repository: `https://github.com/dafiiscoding/ml-dss`
- Branch: `main`
- Main file: `app/streamlit_app.py`
- Python đề xuất: `3.12`
- Dependencies cloud: `app/requirements.txt`
- Không cần Secrets.

Bản cloud dùng các CSV, metrics, figures và năm model inference đã đóng gói
trong Git. Corpus ảnh CrisisMMD 1,8 GB và các embedding `.npy` không được tải
lên repository.

## Kiểm tra trước khi deploy

```powershell
python -m scripts.verify_cloud_bundle
python -m scripts.test_streamlit_cloud
python -m scripts.submission_preflight
```

Kết quả đầu tiên phải kết thúc bằng:

```text
[OK] Streamlit Cloud bundle is ready.
```

Kết quả smoke test phải xác nhận `6/6 cloud entrypoints passed`.

## Các bước trên Streamlit Community Cloud

1. Mở `https://share.streamlit.io/` và đăng nhập bằng GitHub.
2. Chọn **Create app**.
3. Chọn repository `dafiiscoding/ml-dss`.
4. Chọn branch `main`.
5. Nhập main file path: `app/streamlit_app.py`.
6. Trong Advanced settings, chọn Python `3.12`.
7. Không thêm secret.
8. Nhấn **Deploy**.

Sau khi deploy, đổi subdomain trong phần Settings nếu muốn, ví dụ:

```text
https://ml-dss-group29.streamlit.app
```

## Hành vi của bản online

- Overview, EDA, Model Evaluation và Priority Queue đọc artefact thật đã lưu.
- Single Case mặc định chạy text-only, nên không phải tải CLIP khi vừa mở app.
- Khi người dùng upload ảnh lần đầu, server tải
  `openai/clip-vit-base-patch32` rồi mới trích đặc trưng. Lần đầu sẽ chậm hơn.
- Mục chọn ảnh trực tiếp từ corpus chỉ xuất hiện trên máy local có bộ ảnh thật.

## Xử lý lỗi thường gặp

### App báo thiếu model

Chạy:

```powershell
python -m scripts.verify_cloud_bundle
git status
```

Xác nhận năm file sau đã được commit:

```text
models/text_vectorizer.pkl
models/text_inf_clf.pkl
models/text_cat_clf.pkl
models/image_inf_clf.pkl
models/image_cat_clf.pkl
```

### Trang ảnh khởi động chậm

Đây là cold start của CLIP. Text-only vẫn hoạt động độc lập. Không đưa corpus
ảnh hoặc embedding train lên Git để khắc phục vì chúng không cần cho inference.

### Hết bộ nhớ khi dùng ảnh

Giữ app ở text-only trong buổi chấm và dùng screenshot để minh chứng nhánh ảnh.
Nếu cần inference ảnh ổn định hơn, bước tiếp theo là chuyển CLIP sang một dịch
vụ inference riêng; không giảm chất lượng bằng đặc trưng màu giả.

### Dependency không tương thích

Đảm bảo app dùng Python 3.12 và Streamlit đọc `app/requirements.txt`. Không
chọn Python 3.14 cho Community Cloud.
