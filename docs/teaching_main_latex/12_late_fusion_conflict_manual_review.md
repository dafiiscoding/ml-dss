# Buoi 12 - Calibration, Late Fusion, Conflict Score va Manual Review

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> Late Fusion tron xac suat cua model text va model anh sau khi tung nhanh da du bao. Conflict score do muc bat dong giua hai nhanh, dung de dua ca kho vao Manual Review trong gioi han cong suat.

## Tu khoa

- Calibration
- Predict_proba
- Late Fusion
- Weighted average
- Threshold
- Alpha
- Beta
- Conflict score
- Total variation distance
- Manual Review
- Capacity constraint

## Vi sao can xac suat

Neu chi co nhan:

```text
text: informative
image: not-informative
```

Ta khong biet muc do tu tin.

Neu co xac suat:

```text
text: 0,80 informative
image: 0,40 informative
```

Ta co the tron va tinh conflict.

## Calibration la gi

Calibration la chuyen score cua model thanh xac suat co the dung de dien giai hon.

Trong du an:

- LinearSVC khong co predict_proba truc tiep.
- Dung CalibratedClassifierCV voi sigmoid va 3 fold noi bo.
- Calibration chi dung train, khong dung dev/test.

Can noi:

> Calibration khong lam model "dung hon" mot cach than ky. No giup score tro thanh xac suat de fusion va policy doc duoc.

## Late Fusion la gi

Late Fusion = gop sau khi moi nhanh da du bao.

Khac voi early fusion:

- Early fusion: noi feature text va anh roi train model chung.
- Late fusion: train model text rieng, model anh rieng, sau do tron xac suat.

Du an dung Late Fusion vi:

- text va anh co feature khac nhau;
- moi nhanh co classifier tot nhat khac nhau;
- de quan sat dong gop tung nhanh;
- don gian va minh bach.

## Informative fusion

Cong thuc:

```text
p_fused = alpha * p_text + (1 - alpha) * p_image
y_hat = 1 neu p_fused >= tau
```

Trong du an:

- alpha = 0,70.
- image weight = 0,30.
- tau = 0,38.

Vi du:

```text
p_text = 0,80
p_image = 0,40
alpha = 0,70

p_fused = 0,70 * 0,80 + 0,30 * 0,40
        = 0,56 + 0,12
        = 0,68
```

So voi threshold tau = 0,38:

```text
0,68 >= 0,38 => informative
```

Giai thich:

Text duoc trong so lon hon vi tren dev text phu hop hon cho task nay, nhung anh van keo ket qua xuong neu anh khong dong tinh.

## Humanitarian fusion

Moi nhanh cho vector 8 xac suat.

Vi du rut gon 3 lop:

```text
q_text  = [0,60; 0,30; 0,10]
q_image = [0,20; 0,50; 0,30]
beta = 0,55
```

Tinh:

```text
q_fused = 0,55*q_text + 0,45*q_image
        = [0,42; 0,39; 0,19]
```

Chon lop co xac suat cao nhat: lop 1.

Trong du an:

- beta = 0,55 cho text;
- image weight = 0,45.

## Vi sao threshold informative la 0,38 chu khong 0,5

Threshold duoc chon tren dev theo muc tieu F2 va trade-off Recall/Precision.

Vi F2 uu tien Recall, threshold co the thap hon 0,5 de bat nhieu informative hon.

Can noi:

> 0,38 khong phai so tuy y. No duoc chon tren dev truoc khi test.

## Conflict informative

Conflict informative:

```text
|p_text - p_image|
```

Vi du:

```text
p_text = 0,90
p_image = 0,20
conflict = 0,70
```

Hai nhanh bat dong manh.

## Conflict category

Voi 8 lop, dung total variation distance:

```text
C_cat = 1/2 * sum_k |q_text,k - q_image,k|
```

Truc giac:

> Do hai phan bo xac suat khac nhau bao nhieu.

Neu text va anh phan bo giong nhau, C_cat gan 0.

Neu text va anh gan nhu chon hai lop khac nhau hoan toan, C_cat cao.

## Conflict score cuoi

Conflict score la gia tri lon hon giua:

- conflict informative;
- conflict category.

Noi:

> Neu bat dong o bat ky tang quan trong nao, ca nay dang xem lai.

## Manual Review

Manual Review khong phai "model sai". No la:

- co che quan tri rui ro;
- dua ca kho cho con nguoi;
- giu he thong khong tu tin qua muc.

Quy trinh con nguoi:

1. He thong neu ly do: conflict cao.
2. Nguoi xem tweet, anh, nguon, thoi diem.
3. Doi chieu nguon thu hai: hotline, bao, co quan dia phuong.
4. Chap nhan hoac sua category/priority.
5. Ghi lai ket qua xac minh.

## Nguong conflict 0,54

Trong du an:

- threshold conflict = 0,54.
- chon tren dev.
- toi da F1 phat hien bat dong.
- review rate khong vuot 25%.

Giai thich capacity:

Neu threshold qua thap:

- review nhieu;
- con nguoi qua tai;
- DSS mat tac dung giam tai.

Neu threshold qua cao:

- bo qua nhieu conflict;
- rui ro tin nham.

## Ket qua Manual Review

Canonical test:

- review rate 24,85%;
- 539 ca vao hang review tren 2.169 dong test;
- Precision = 0,7532;
- Recall = 0,3295;
- F1 = 0,4585.

Dien giai:

- Khoang 3/4 ca duoc review that su co bat dong theo annotation.
- Chi bat duoc khoang 1/3 tong ca bat dong.
- Day la triage, khong phai detector toan dien.

## Bai tap

Cho:

```text
p_text = 0,85
p_image = 0,30
alpha = 0,70
tau = 0,38
```

Tinh:

```text
p_fused = 0,70*0,85 + 0,30*0,30 = 0,595 + 0,09 = 0,685
```

Ket luan:

- informative vi 0,685 >= 0,38.
- informative conflict = 0,55.
- neu conflict threshold = 0,54 thi bat Manual Review.

## Loi hieu sai can chan

1. "Fusion la noi feature text va anh."
   - Sai. Day la Late Fusion tren xac suat.

2. "Conflict cao thi ha uu tien."
   - Sai. Conflict cao thi review; ca High van giu High neu nguy hiem.

3. "Manual Review bat het moi bat dong."
   - Sai. Recall chi 0,3295 do gioi han capacity.

4. "Threshold 0,54 lay tu test."
   - Sai. Chon tren dev, khoa truoc test.

## Cau hoi kiem tra

1. Late Fusion la gi?
2. Informative fusion dung trong so nao?
3. Humanitarian fusion dung trong so nao?
4. Conflict score do gi?
5. Vi sao review rate bi gioi han 25%?
6. Conflict cao co tu dong ha Priority khong?

