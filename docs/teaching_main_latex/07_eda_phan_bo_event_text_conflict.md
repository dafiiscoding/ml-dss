# Buoi 7 - EDA train-only: phan bo nhan, event shift, text va bat dong text-image

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> EDA la xem va hieu du lieu truoc khi xay model. EDA dung de thiet ke chi duoc dung train. Trong du an, EDA phat hien mat can bang lop, khac biet theo event, do dai tweet khong du phan loai, va 55% mau train co nhan text-image bat dong.

## Tu khoa

- EDA
- Train-only
- Distribution
- Class imbalance
- Event shift
- Token
- Conflict
- Modality
- Text label
- Image label

## EDA la gi

EDA = Exploratory Data Analysis.

Noi bang doi thuong:

> EDA la mo hop du lieu ra xem ben trong co gi truoc khi xay model.

EDA tra loi:

- Co bao nhieu mau moi lop?
- Lop nao hiem?
- Moi tham hoa co phan bo khac nhau khong?
- Text co do dai khac nhau theo nhan khong?
- Anh co ve dung du lieu that khong?
- Text va anh co dong y voi nhau khong?

## Vi sao EDA phai train-only

Neu dung EDA test de quyet dinh model, test da lo thong tin.

Vi du:

Neu nhin test thay missing/found rat it, roi doi threshold rieng de lam diem cao hon, test khong con la de thi cuoi.

Quy tac:

- EDA de thiet ke: train only.
- Kiem ke toan ven file/duplicate: co the toan corpus, vi khong dung pattern de toi uu model.

## Phan bo informative

Train:

- informative: 8.338 mau;
- not-informative: 5.270 mau;
- informative chiem 61,27%.

Y nghia:

- Luon doan informative da co Recall = 1.
- F2 co the cao ngay ca voi dummy baseline.
- Can bao cao them Accuracy, Balanced Accuracy, F1, MCC.

## Phan bo humanitarian

Task humanitarian mat can bang nghiem trong:

- not_humanitarian: 5.260 mau train;
- missing_or_found_people: 24 mau train;
- chenhlech khoang 219:1.

Giai thich:

> Neu lop lon qua nhieu, Accuracy co the cao chi vi doan lop lon. Macro-F1 bat moi lop co tieng noi ngang nhau.

## Ba he qua thiet ke

Tu EDA phan bo nhan, du an quyet dinh:

1. Khong chon model chi bang Accuracy.
2. Dung class weight khi thuat toan ho tro.
3. Bao cao Macro-F1, F2, confusion matrix va F1 theo lop.

## Khac biet theo event

Ban main neu:

- Sri Lanka floods co 69,29% not-humanitarian.
- Iraq-Iran earthquake chi 22,76% not-humanitarian.
- Mexico earthquake co 27,95% rescue/donation.
- Iraq-Iran co 13,57% injured/dead.

Y nghia:

> Moi tham hoa co dang tin khac nhau. Model co the tot o event nay nhung kem o event khac.

Day khai niem `event shift`:

> Event shift la khi phan bo du lieu thay doi theo su kien.

## Leave-one-event-out

Ban main noi danh gia nghiem ngat hon trong tuong lai:

```text
giu toan bo mot event ngoai train
```

Vi du:

- Train tren 6 tham hoa.
- Test tren Hurricane Maria chua tung thay.

Day nguoi hoc:

> Split chinh thuc hien tai de hon thuc te vi moi event deu co mat trong train/dev/test.

## Kham pha text

Quy trinh EDA text trong ban main:

1. Dat cau hoi: do dai co phan biet nhan khong?
2. Tao bien do dai ky tu va so token.
3. Phan nhom theo nhan va event.
4. So sanh phan bo.
5. Xem top token.
6. Kiem tra nhieu con sot.
7. Rut ket luan co gioi han.

Ket qua:

- Tweet trung binh 117,48 ky tu.
- Trung binh 11,72 tu sau lam sach.
- Informative: 11,75 tu.
- Not-informative: 11,68 tu.

Ket luan:

> Do dai tweet gan nhu khong phan biet informative. Khong the dung rule "tweet dai thi huu ich".

## Token noi bat

Top token phan anh:

- dia danh;
- cuu tro;
- thiet hai;
- ten event;
- ngu canh thoi su.

Can can than:

Neu clustering chi nhin thay `harvey`, `irma`, `maria`, no co the chi dang gom theo su kien, khong phai theo nhu cau.

Do do du an loai them token nen tang nhu:

- rt;
- amp;
- http;
- ten event trong mot so phan tich.

## Bat dong da phuong thuc

Modality = phuong thuc du lieu.

O du an:

- text la mot modality;
- image la mot modality.

Bat dong text-image nghia la:

- nhan rieng cua text khac nhan rieng cua image.

Ket qua trong train:

- 55,0% mau co nhan category text va image khac nhau.
- missing/found bat dong 91,7%.
- injured/dead bat dong 91,0%.
- affected bat dong 88,9%.
- vehicle damage bat dong 87,3%.
- not-humanitarian van bat dong 38,2%.

## Dien giai dung

Sai:

> Text va anh khac nhau nen mot cai sai.

Dung:

> Text va anh co the noi ve hai mat khac nhau cua cung tweet.

Vi du:

Tweet noi:

```text
"missing people reported in town"
```

Anh lai la:

```text
anh duong ngap
```

Text noi ve nguoi mat tich, anh cho thay ha tang/ngap. Khong the ket luan text sai hay anh sai. Chi co the noi bang chung khong dong nhat.

## Tu EDA den thiet ke

Bang quyet dinh:

| Bang chung train | Suy luan co gioi han | Quyet dinh |
|---|---|---|
| Mat can bang 219:1 | Accuracy che lop hiem | Dung F2, Macro-F1, class weight |
| Event profile khac nhau | Co domain shift | Filter/metric theo event |
| Do dai hai lop gan nhau | Length khong du phan loai | Dung TF-IDF noi dung |
| 55% modality bat dong | Khong tin tuyet doi mot nhanh | Tune fusion, Manual Review |
| Anh lap xuyen split | Metric co the lac quan | Canonical/robust mask |

## Bai tap

Hoi:

Neu EDA thay 90% lop A, 10% lop B, model doan tat ca A duoc Accuracy 90%. Co tot khong?

Dap an:

Chua chac. Phai xem lop B co quan trong khong, Macro-F1/Recall lop B ra sao.

Hoi:

Neu text label la missing, image label la infrastructure, nen lam gi?

Dap an:

Khong tu chon mot ben. Dung fusion va bat Manual Review neu bat dong cao.

## Loi hieu sai can chan

1. "EDA la ve bieu do cho dep."
   - Sai. EDA phai dan den quyet dinh thiet ke.

2. "Xem test de hieu du lieu truoc cung duoc."
   - Sai neu dung de chon model.

3. "Tweet dai hon thi informative."
   - Sai. So tu trung binh gan nhu bang nhau.

4. "Text-image bat dong nghia la model se hong."
   - Sai. No la tin hieu can fusion va review.

## Cau hoi kiem tra

1. EDA dung de lam gi?
2. Vi sao EDA thiet ke chi dung train?
3. Informative chiem bao nhieu phan tram train?
4. Lop humanitarian nao rat hiem?
5. Event shift la gi?
6. 55% modality bat dong dan den quyet dinh nao?

