# Buoi 15 - Dashboard Streamlit, CSV, cloud bundle va kiem thu

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> Dashboard la mat giao tiep cua DSS. No khong chi hien metric, ma cho xem workload, EDA, ket qua model, thu mot ca moi va thao tac Priority Queue. Ban hien tai la prototype Streamlit, khong phai production command system.

## Tu khoa

- Dashboard
- Streamlit
- Overview
- EDA page
- Model Evaluation
- Single-Case Demo
- Priority Queue
- CSV
- Database
- Cloud bundle
- AppTest
- Integration test

## Vi sao can dashboard

Neu chi co model va file CSV, nguoi dieu phoi kho dung.

Dashboard giup:

- xem tong quan workload;
- loc theo event/team/priority;
- truy vet vi sao mot ca duoc goi y;
- thu mot tweet/anh moi;
- tai hang doi de xu ly tiep.

Noi:

> DSS khong chi la model. DSS can giao dien de con nguoi ra quyet dinh.

## Nam trang chuc nang

### 1. Overview

Hien:

- so case;
- Priority;
- Manual Review;
- routing team;
- category;
- phan bo Risk Score;
- filter event.

Dung de:

- ban dieu phoi xem workload.

### 2. EDA

Hien:

- thong ke train-only;
- phan bo nhan;
- K-Means;
- Apriori;
- kiem ke anh;
- bat dong text-image.

Dung de:

- giai thich vi sao thiet ke metric/fusion/review nhu vay.

Can nhac:

> Trang EDA khong dung test de kham pha pattern thiet ke.

### 3. Model Evaluation

Hien:

- so sanh 6 model tren dev;
- tuning;
- baseline;
- fusion;
- robust test;
- metric theo lop.

Dung de:

- chung minh model duoc chon co quy trinh;
- nguoi doc thay ca diem manh lan lop yeu.

### 4. Single-Case Demo

Nhan:

- tweet text;
- anh upload.

Che do:

- text-only: nhanh;
- text+image: chay CLIP roi fusion-DSS.

Tra ve:

- xac suat;
- category;
- conflict;
- Risk;
- Priority;
- team;
- action.

### 5. Priority Queue

Cho:

- loc theo Priority;
- loc theo team;
- loc theo event;
- loc theo Manual Review;
- xem chi tiet tung dong;
- tai CSV.

Dung de:

- thao tac hang doi xu ly.

## Kien truc du lieu dashboard

Dashboard khong huan luyen lai model khi mo trang.

Batch test da xu ly thanh:

```text
dashboard_test_predictions.csv
```

File nay co:

- xac suat fusion;
- category;
- Risk Score;
- Priority;
- routing;
- co review.

Loi ich:

- tai trang nhanh;
- metric dung model da khoa;
- tach batch evaluation khoi single-case inference.

## Lazy-load CLIP

Single-Case Demo:

- tai vectorizer va 4 classifier;
- model anh lazy-load;
- chi khi user upload anh moi nap PyTorch/Transformers/CLIP.

Loi ich:

- text-only khoi dong nhanh;
- cloud khong can tai CLIP moi trang;
- giam cold start phan khong can anh.

## Vi sao CSV thay database

Ban main co muc rieng.

CSV phu hop vi:

- prototype hoc thuat;
- du lieu nho;
- read-mostly;
- tinh mot lan, dashboard chi doc;
- de mo bang nhieu cong cu;
- de version va tai lap;
- khong can server.

CSV khong phu hop production vi thieu:

- ghi dong thoi an toan;
- transaction;
- index;
- phan quyen;
- audit log;
- truy van lon.

Neu production:

- PostgreSQL;
- object storage;
- queue ingest;
- auth;
- audit log.

## Cong nghe su dung

| Cong nghe | Vai tro |
|---|---|
| Python | Ngon ngu pipeline |
| pandas/NumPy | Bang du lieu, tinh toan |
| scikit-learn | TF-IDF, 6 classifier, metric |
| PyTorch + Transformers | Nap CLIP va forward anh |
| Streamlit | Frontend prototype |
| Plotly/Matplotlib | Bieu do |
| Jupyter | Notebook EDA/model co output |
| XeLaTeX | Bao cao PDF tieng Viet |

## Chay local

Lenh:

```powershell
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Full rebuild:

```powershell
python -m scripts.run_all
```

Can nhac:

- Full rebuild can raw image corpus khoang 1,8 GB.
- Khong can chay lai pipeline neu model/cache da co.

## Cloud bundle

Bundle cloud co:

- processed train/test cho EDA/dashboard;
- vectorizer;
- 4 classifier da khoa;
- cau hinh fusion;
- bang metric;
- Streamlit app;
- requirements CPU.

Khong dua len Git:

- raw image corpus 1,8 GB;
- embedding lon.

Deploy:

- repo: `dafiiscoding/ml-dss`;
- branch: main;
- main file: `app/streamlit_app.py`;
- Python: 3.12;
- secrets: khong yeu cau.

## Kiem thu

Bao cao noi:

- 6 Streamlit entrypoint chay AppTest;
- GitHub Actions kiem tra bundle, dependency, text-only inference;
- unit test kiem tra du lieu, duplicate mask, tuning, robustness, DSS;
- integration test chay mot anh that qua CLIP.

Y nghia:

> App khoi dong va luong chinh khong phat sinh exception.

Chua chung minh:

- chiu tai nhieu nguoi;
- bao mat upload;
- phan quyen;
- audit log;
- SLA;
- production latency.

## Bai tap

Hoi:

Neu dashboard mo len ma train lai model moi lan, co van de gi?

Dap an:

- Cham.
- Metric co the khong dung model da khoa.
- Khong tai lap.
- User khong can train khi chi muon xem dashboard.

Hoi:

Vi sao Single-Case Demo nen lazy-load CLIP?

Dap an:

- CLIP nang.
- Text-only khong can CLIP.
- Giam cold start.

## Loi hieu sai can chan

1. "Dashboard la production system."
   - Sai. La prototype hoc thuat.

2. "CSV la thieu sot."
   - Sai trong prototype read-mostly; production moi can DB.

3. "AppTest chung minh app chiu tai."
   - Sai. AppTest chung minh luong chinh khong exception.

4. "Mo dashboard la train lai model."
   - Sai. Dashboard doc cache/model da khoa.

## Cau hoi kiem tra

1. Ke ten 5 trang dashboard.
2. Single-Case Demo tra ve nhung gi?
3. Vi sao dung CSV trong prototype?
4. Khi nao can database?
5. AppTest chung minh dieu gi?
6. Dashboard co duoc dung de tu dong dieu dong khong?

