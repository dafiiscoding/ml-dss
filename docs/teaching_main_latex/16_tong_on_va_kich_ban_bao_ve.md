# Buoi 16 - Tong on, kich ban bao ve va nhung cau cam noi sai

## Muc tieu

Sau buoi nay, nguoi hoc phai co the trinh bay du an trong 3 phut, 10 phut va tra loi cau hoi phan bien.

## Ban 30 giay

Du an xay dung mot he ho tro quyet dinh ung pho tham hoa tren CrisisMMD v2.0. Moi mau gom tweet va anh. Text duoc lam sach va bien thanh TF-IDF, anh duoc bien thanh embedding CLIP frozen. Sau do du an so sanh 6 classifier, tune tren dev, tron text-anh bang Late Fusion, tinh Risk/Priority/Routing/Manual Review va hien tren dashboard Streamlit. He thong chi ho tro sang loc va sap xep thong tin, khong tu dong dieu dong.

## Ban 3 phut

1. Bai toan:
   - Trong tham hoa, mang xa hoi co nhieu tweet nhanh nhung nhieu nhieu.
   - Nguoi dieu phoi can biet bai nao dang doc, thuoc nhom nao, uu tien nao, doi nao xu ly, co can review khong.

2. Du lieu:
   - CrisisMMD v2.0, 18.082 cap tweet-anh, 7 tham hoa 2017.
   - Khoa mau la `(tweet_id, image_id)`.
   - Dung `label` chinh thuc cho informative va `label_top` cho humanitarian.
   - Khong suy informative tu humanitarian vi se sai 2.649 mau.

3. Tien xu ly:
   - Text lam sach URL, mention, HTML entity, giu noi dung hashtag.
   - Text thanh TF-IDF 2.000 chieu.
   - Anh qua CLIP ViT-B/32 frozen thanh embedding 512 chieu.

4. Kiem soat ro ri:
   - Dung SHA-256 cho anh trung byte.
   - Dung pHash/near duplicate review cho robust.
   - Canonical test con 2.169 dong, robust test con 2.032 dong.

5. EDA:
   - EDA thiet ke chi dung train.
   - Mat can bang lop 219:1.
   - Event shift co that.
   - 55% text-image category bat dong.
   - K-Means/Apriori/PCA/t-SNE chi de kham pha.

6. Model:
   - So sanh 6 classifier.
   - Text informative: calibrated Linear SVM C=3.
   - Text humanitarian: Logistic Regression C=1.
   - Image informative: k-NN k=41 distance.
   - Image humanitarian: Logistic Regression C=1.

7. Fusion va DSS:
   - Informative fusion text/image 0,70/0,30, threshold 0,38.
   - Humanitarian fusion 0,55/0,45.
   - Conflict threshold 0,54, review capacity 25%.
   - Risk Score la policy trong so, Priority khong co ground truth.

8. Ket qua:
   - Informative fusion Accuracy 0,6948, F1 0,8025, F2 0,9045, MCC 0,3325.
   - Always-informative F2 da 0,8939, nen khong khoe F2 mot minh.
   - Humanitarian fusion Macro-F1 0,4005 so voi majority 0,0686.
   - Robust test giu ket luan, nhung lop hiem va event shift con gioi han.

9. San pham:
   - Dashboard Streamlit 5 trang.
   - Prototype hoc thuat, chua production.

## Ban 10 phut

Dung cau truc:

1. Mo bai bang van de thuc te.
2. Noi ranh gioi he thong: CrisisMMD da luu san, khong Twitter API, khong thay hotline.
3. Giai thich du lieu va nhan.
4. Giai thich hai nhanh feature: TF-IDF va CLIP.
5. Giai thich train/dev/test va duplicate mask.
6. Giai thich EDA dan den thiet ke.
7. Noi 4 bai toan con va 6 classifier.
8. Noi quy trinh tuning tren dev.
9. Noi Late Fusion va Manual Review.
10. Noi Risk/Priority la policy.
11. Doc ket qua voi baseline.
12. Ket luan bang gioi han va huong phat trien.

## Cau hoi bao ve va cach tra loi

### 1. Vi sao khong thu thap tweet live?

Tra loi:

Du an pham vi hoc thuat, dung CrisisMMD da luu san de co nhan ground truth va tai lap. Thu thap live can API, quyen du lieu, ingest, xac minh, database va governance. Day la huong phat trien, khong phai pham vi hien tai.

### 2. Vi sao khong thay hotline?

Tra loi:

Hotline la kenh chinh de cuu tro truc tiep va xac minh hai chieu. DSS chi bo sung bang cach sap xep thong tin mang xa hoi ma con nguoi khong doc xue.

### 3. Vi sao dung CLIP nhung khong fine-tune?

Tra loi:

Muc tieu hoc phan la classifier co dien va DSS. Train 13.608 mau, lop hiem rat it, fine-tune de overfit va ton chi phi. CLIP frozen giup trich dac trung on dinh va so sanh cong bang cac classifier.

### 4. Vi sao khong XGBoost?

Tra loi:

Khong phai XGBoost kem. Pham vi bao phu cac thuat toan trong mon hoc, SVM/LR da la baseline manh voi TF-IDF, lop hiem it de overfit, va them model moi can protocol tuning cong bang.

### 5. Vi sao F2 cao nhung van can can than?

Tra loi:

Vi informative la lop da so. Always-informative da dat F2 0,8939. Fusion dat 0,9045, gain chi 0,0106. Gia tri ro hon nam o Accuracy, Balanced Accuracy, F1 va MCC.

### 6. Macro-F1 0,4005 co thap khong?

Tra loi:

Can so voi baseline va lop hiem. Majority baseline chi 0,0686, text-only 0,3260, image-only 0,3646. Fusion cao hon. Nhung lop hiem nhu missing/found van yeu, nen khong tu dong hoa.

### 7. Risk Score co phai model hoc tu du lieu khong?

Tra loi:

Khong. Risk Score la weighted scoring policy minh bach. CrisisMMD khong co ground truth Priority nen khong the khang dinh toi uu thong ke.

### 8. Manual Review co bat het moi conflict khong?

Tra loi:

Khong. Voi capacity 25%, threshold 0,54 cho Precision 0,7532 nhung Recall 0,3295. Day la triage trong gioi han cong suat.

### 9. K-Means co chung minh 8 lop tach tot khong?

Tra loi:

Khong. Silhouette rat thap, K=8 chi de doi chieu taxonomy. K-Means chi kham pha chu de, khong thay classifier.

### 10. t-SNE nhin chong lop co nghia CLIP vo dung?

Tra loi:

Khong. t-SNE la hinh 2D va co the lam meo khoang cach. PCA cho thay 2 truc dau chi giu 12,4% phuong sai. Gia tri CLIP phai xem tren dev/test classifier.

## Nhung cau cam noi

Khong noi:

- "AI tu dong cuu ho."
- "He thong thay hotline."
- "Du an crawl tweet live."
- "CLIP duoc fine-tune tren CrisisMMD."
- "F2 0,9045 chung minh model rat xuat sac."
- "Risk Score toi uu thong ke."
- "Priority High la lenh dieu dong."
- "K-Means tim dung 8 lop."
- "Manual Review bat het conflict."
- "Dashboard da san sang production."

Noi thay the:

- "He thong ho tro sang loc va sap xep thong tin."
- "DSS bo sung cho hotline."
- "Du an doc CrisisMMD da luu san."
- "CLIP frozen de trich embedding."
- "F2 phai doc cung baseline va MCC."
- "Risk/Priority la policy minh bach can du lieu nghiep vu de xac thuc."
- "High la uu tien trong hang doi."
- "K-Means chi kham pha chu de."
- "Manual Review la triage duoi capacity."
- "Dashboard la prototype hoc thuat."

## Bai tap tong ket

Yeu cau nguoi hoc ke pipeline khong nhin tai lieu:

```text
CrisisMMD
-> join nhan chinh thuc
-> loc duplicate/mask
-> clean tweet + TF-IDF
-> anh + CLIP embedding
-> EDA train-only
-> 6 classifier
-> tune tren dev
-> Late Fusion
-> conflict/Manual Review
-> Risk/Priority/Routing
-> dashboard
-> bao cao test va robust
```

Neu thieu bat ky diem nao sau day, bat hoc lai:

- nhan chinh thuc;
- train/dev/test;
- duplicate mask;
- baseline;
- Risk la policy;
- con nguoi quyet dinh cuoi.

