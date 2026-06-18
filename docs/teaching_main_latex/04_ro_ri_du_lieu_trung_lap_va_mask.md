# Buoi 4 - Ro ri du lieu, trung lap anh/text va hai mat na danh gia

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> Ro ri du lieu la khi model duoc thay thong tin cua dev/test trong luc hoc hoac chon cach lam. Du an phai loc anh/text trung bang canonical mask va kiem tra them bang robustness mask.

## Tu khoa

- Data leakage
- Duplicate
- SHA-256
- Perceptual hash
- pHash
- Hamming distance
- Cleaned text
- Canonical mask
- Robustness mask
- Near-duplicate

## Mo dau bang vi du hoc sinh

Noi:

"Neu hoc sinh duoc xem de thi truoc, diem cao khong chung minh hoc sinh gioi. Trong machine learning cung vay. Neu test co anh/text giong train, model co the nho thay vi hieu."

Sau do hoi:

"Vay diem test co con dang tin khong?"

Dap an:

"Kem dang tin hon, vi bi lac quan."

## Ro ri du lieu la gi

Ro ri du lieu xay ra khi thong tin cua dev/test di vao:

- huan luyen model;
- fit TF-IDF;
- chon hyperparameter;
- chon threshold;
- chon cach fusion;
- EDA de thiet ke model.

Trong du an mang xa hoi, ro ri co the tinh vi:

- ID khac nhau nhung anh giong het.
- Text khac do URL/mention, nhung sau lam sach lai giong.
- Anh bi crop/resize nhe nhung noi dung gan nhu cung mot anh.
- Tweet cung su kien lap lai y chang.

## Bon muc kiem tra trung lap

Du an kiem tra:

1. Trung `tweet_id`, `image_id`, hoac cap `(tweet_id, image_id)`.
2. Trung anh chinh xac bang SHA-256.
3. Trung van ban sau lam sach.
4. Gan trung bang pHash cho anh va similarity cho text.

## Ket qua quan trong

Bao cao noi:

- Khong co ID hoac cap ID giao nhau giua split.
- Co 43 hash anh chung train-dev.
- Co 46 hash anh chung train-test.
- Co 21 hash anh chung dev-test.
- Toan corpus 18.082 dong chi co 17.777 hash anh duy nhat.
- Nghia la co 305 lan lap ngoai anh dai dien dau tien.
- Co mot cleaned text trung train-dev va mot cleaned text trung train-test.

Day nguoi hoc:

> Khong trung ID khong co nghia la khong trung noi dung.

## SHA-256 la gi

Giai thich doi thuong:

> SHA-256 la van tay cua file. Cung mot file tung byte giong nhau thi van tay giong nhau. Doi mot byte thoi thi van tay khac han.

Dung de bat:

- anh giong het;
- file copy y nguyen.

Khong bat tot:

- anh bi resize;
- anh bi crop;
- anh chup lai man hinh;
- anh doi nen mau nhe.

## pHash la gi

Giai thich doi thuong:

> pHash la van tay dua tren viec anh trong nhu the nao, khong dua tung byte. Anh crop/resize nhe van co the ra van tay gan nhau.

pHash dung:

- dac trung thi giac tan so thap;
- van tay 64 bit;
- so sanh bang Hamming distance.

## Hamming distance la gi

Noi:

> Hamming distance la so bit khac nhau giua hai chuoi bit.

Vi du:

```text
110010
110011
```

Khac 1 bit, Hamming = 1.

Trong du an:

- pHash DCT 64-bit.
- Ung vien gan trung neu Hamming <= 4.

Can noi ro:

> pHash chi tao ung vien can review, khong tu dong xoa tat ca.

## Cleaned text trung la gi

Vi du:

```text
Tweet A: "Need help @abc https://x #Harvey"
Tweet B: "Need help #Harvey"
```

Sau lam sach:

```text
need help harvey
need help harvey
```

Hai tweet co ID khac nhau nhung noi dung model thay giong nhau.

## Canonical mask

Canonical mask loai mot dong dev/test neu:

- anh SHA-256 cua no da xuat hien o split truoc;
- hoac cleaned text cua no da xuat hien o split truoc.

Noi:

> Canonical mask la bo loc chinh thuc cho danh gia chinh, vi tieu chi ro rang va tu dong.

Sau canonical:

- dev con 2.189;
- test con 2.169.

## Robustness mask

Robustness mask bat dau tu canonical mask, roi loai them cac near-duplicate da duoc review.

Dung:

- pHash cho anh;
- cosine cua CLIP text embedding;
- TF-IDF ky tu/tu;
- review tung cap.

Sau robustness:

- dev con 2.078;
- test con 2.032.

Noi:

> Robustness mask khong phai de tune lai. No la de hoi: neu loai them cac ban gan trung da xac nhan, ket luan co doi khong?

## Hai cau hoi khac nhau

Canonical test tra loi:

> Model hoat dong the nao sau khi loai exact duplicate ro rang?

Robust test tra loi:

> Ket luan co ben khong neu loai them near-duplicate da review?

## Dieu cam ky

Khong duoc:

- xoa pHash candidate tu dong neu chua review;
- tune lai model tren robust test;
- bao cao synthetic fallback;
- bo qua mask khi tinh metric dev/test.

## Vi du day tren bang

Tao bang:

| Mau | Split | Anh | Text sau lam sach |
|---|---|---|---|
| A | train | hash X | need rescue |
| B | test | hash X | help now |
| C | test | hash Y | need rescue |
| D | test | hash Z | bridge collapse |

Hoi:

- B co bi canonical mask loai khong? Co, vi anh hash X da o train.
- C co bi loai khong? Co, vi cleaned text da o train.
- D co bi loai khong? Khong neu khong trung.

## Loi hieu sai can chan

1. "Khong trung ID la an toan."
   - Sai. Co the trung anh/text.

2. "pHash gan nhau thi chac chan trung."
   - Sai. Chi la ung vien review.

3. "Robust test dung de chon model moi."
   - Sai. Robust test dung de kiem tra do ben sau khi da khoa model.

4. "Loai duplicate la lam mat du lieu nen khong can."
   - Sai. Neu khong loai, metric co the lac quan.

## Cau hoi kiem tra

1. Ro ri du lieu la gi?
2. Vi sao SHA-256 khong bat duoc anh crop?
3. pHash khac SHA-256 o dau?
4. Canonical mask loai cai gi?
5. Robustness mask dung de lam gi?
6. Co duoc tune model tren robust test khong?

