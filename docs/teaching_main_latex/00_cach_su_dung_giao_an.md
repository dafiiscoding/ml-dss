# Buoi 0 - Cach day nguoi mat goc hieu du an

## Muc tieu cua buoi nay

Buoi nay khong day noi dung ky thuat. Buoi nay day cach day.

Nguoi day phai nam bon nguyen tac:

1. Di tu cau chuyen doi thuong den tu khoa.
2. Moi tu khoa phai co vi du.
3. Moi cong thuc phai co cau noi bang tieng Viet truoc.
4. Moi ket qua phai noi kem gioi han.

Nguoi hoc mat goc khong can nghe ngay "CLIP ViT-B/32 frozen embedding 512-D". Ho can nghe:

> Anh qua may phien dich CLIP de thanh mot day so 512 o. Day so nay khong phai nhan, chi la cach may mo ta noi dung anh.

Sau do moi noi thuat ngu.

## Cach chia tang giai thich

Moi khai niem nen day theo 5 tang:

1. Doi thuong: giai thich nhu chuyen ngoai doi.
2. Trong du an: khai niem nay nam o dau pipeline.
3. Vi du cu the: dung mot tweet/anh gia dinh.
4. So lieu trong bao cao: dua con so that neu co.
5. Gioi han: khai niem nay khong chung minh dieu gi.

Vi du voi `Risk Score`:

- Doi thuong: diem de sap xep cai nao can xem truoc.
- Trong du an: tinh sau khi co xac suat fusion va category.
- Vi du: cau "bridge collapsed, people trapped" co risk cao.
- So lieu: nguong 0-39 Low, 40-69 Medium, 70-100 High.
- Gioi han: khong phai lenh dieu dong va khong co ground truth Priority.

## Thu tu day nen giu

Khong day model truoc. Day theo thu tu:

1. Van de thuc te.
2. Ranh gioi he thong.
3. Du lieu.
4. Cach bien du lieu thanh so.
5. Cach xem du lieu.
6. Cach huan luyen model.
7. Cach danh gia.
8. Cach tron text va anh.
9. Cach bien du bao thanh hang doi.
10. Ket qua va gioi han.

Neu day nguoc lai, nguoi hoc se thuoc cong thuc nhung khong hieu du an lam gi.

## Cach dung ngon ngu

Nen noi:

- "May goi y", khong noi "may quyet dinh".
- "Hang doi uu tien", khong noi "lenh dieu dong".
- "Du lieu da luu san", khong noi "lay tweet truc tiep".
- "Chinh sach minh bach", khong noi "model Risk toi uu".
- "Anh la mot nguon bang chung doc lap", khong noi "anh luon dung hon text".

Tranh noi:

- "AI cuu ho tu dong".
- "F2 cao nen mo hinh rat tot".
- "K-Means tim ra 8 nhom dung nhu nhan".
- "t-SNE nhin tach lop nen CLIP phan lop tot".
- "Lop missing/found bat duoc nen co the tu dong xu ly".

## Kich ban mo dau moi buoi

Moi buoi nen bat dau bang 3 cau:

1. "Buoi truoc ta dang o dau trong pipeline?"
2. "Buoi nay tra loi cau hoi nao?"
3. "Sau buoi nay em phai noi lai duoc dieu gi bang mot cau?"

Vi du buoi TF-IDF:

- Buoi truoc: ta biet tweet la chu tho, model khong doc chu truc tiep.
- Buoi nay: lam sao bien chu thanh so.
- Sau buoi nay: em phai noi duoc "TF-IDF cho diem cao hon cho tu vua xuat hien trong tweet nay, vua khong qua pho bien o moi tweet".

## Kiem tra hieu that

Nguoi hoc hieu that khi co the:

- ke lai pipeline ma khong can nhin file;
- giai thich tai sao can train/dev/test;
- noi duoc tai sao baseline bat buoc phai co;
- noi duoc tai sao Priority chua phai ground truth;
- nhin mot con so F2 va hoi "baseline la bao nhieu?";
- nhin mot hinh t-SNE va khong ket luan qua da.

## Neu chi co 30 phut de day

Day 6 y:

1. Du an doc CrisisMMD da luu san, khong lay tweet truc tiep.
2. Moi mau la tweet + anh + nhan.
3. Text di qua TF-IDF, anh di qua CLIP frozen.
4. Model du bao informative va humanitarian.
5. Late Fusion tron text va anh.
6. DSS tao Risk/Priority/Routing/Review cho con nguoi, khong tu dieu dong.

## Neu co 2 gio de day

Them 6 y:

1. Train/dev/test va ro ri du lieu.
2. SHA-256 va pHash.
3. EDA train-only.
4. Mat can bang lop va vi sao Accuracy khong du.
5. Baseline va ket qua informative.
6. Gioi han lop hiem va production.

