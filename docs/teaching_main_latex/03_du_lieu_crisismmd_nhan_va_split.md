# Buoi 3 - Du lieu CrisisMMD, nhan chinh thuc va train/dev/test

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> Du an dung 18.082 cap tweet-anh CrisisMMD v2.0. Moi mau duoc khoa bang `(tweet_id, image_id)`. Nhan informative phai join tu annotation chinh thuc, khong duoc suy tu nhan humanitarian.

## Tu khoa

- CrisisMMD v2.0
- Annotation
- tweet_id
- image_id
- label
- label_top
- label_text_*
- label_image_*
- train/dev/test
- master partition

## CrisisMMD la gi

CrisisMMD la bo du lieu nghien cuu ve tham hoa da phuong thuc. Da phuong thuc nghia la co nhieu loai du lieu, o day la:

- text: noi dung tweet;
- image: anh dinh kem.

Ban main noi du lieu gom:

- 18.082 cap tweet-anh;
- 7 tham hoa nam 2017;
- tieng Anh;
- co nhan do con nguoi gan.

Bay su kien:

1. Hurricane Harvey.
2. Hurricane Irma.
3. Hurricane Maria.
4. California wildfires.
5. Mexico earthquake.
6. Iraq-Iran earthquake.
7. Sri Lanka floods.

## Don vi quan sat

Day that cham:

> Don vi quan sat khong phai mot tweet don le. Don vi quan sat la mot cap `(tweet_id, image_id)`.

Vi sao?

Mot tweet co the di kem nhieu anh. Neu chi dung `tweet_id`, co the nham nhieu anh thanh mot mau.

Vi du:

```text
tweet_id = 123
image_id = A
tweet_text = "Flood in street"

tweet_id = 123
image_id = B
tweet_text = "Flood in street"
```

Hai dong co cung tweet nhung anh khac nhau. Vay khoa dung la `(tweet_id, image_id)`.

## Cac truong du lieu quan trong

### `event_name`

Ten tham hoa. Dung de EDA va xem event shift.

### `tweet_id`, `image_id`

Dinh danh mau. Dung de:

- join annotation;
- kiem tra trung lap;
- can thu tu cache embedding voi bang du lieu.

### `tweet_text`, `image`

Du lieu goc:

- tweet_text la chu;
- image la duong dan file anh.

### `label`

Nhan chinh thuc cho task informative:

- informative;
- not_informative.

Day la nhan dung de huan luyen va cham diem task informative.

### `label_top`

Nhan chinh thuc cho task humanitarian 8 lop.

Day la nhan dung de huan luyen va cham diem task humanitarian.

### `label_text_*`

Nhan rieng nhin tu text.

Dung de phan tich bat dong text-image, khong dung thay ground truth chung.

### `label_image_*`

Nhan rieng nhin tu anh.

Dung de phan tich bat dong text-image, khong dung thay ground truth chung.

## Hai task du bao

### Task 1 - Informativeness

Cau hoi:

> Bai nay co huu ich cho ung pho tham hoa khong?

Nhan:

- informative;
- not_informative.

Day la bai toan nhi phan.

### Task 2 - Humanitarian category

Cau hoi:

> Bai nay thuoc nhom nhu cau/thiet hai nao?

Co 8 lop:

1. injured/dead.
2. missing/found.
3. rescue/donation.
4. infrastructure.
5. affected.
6. vehicle damage.
7. other relevant.
8. not humanitarian.

## Loi join nhan rat quan trong

Bao cao phat hien:

> Neu suy dien nhan informative tu humanitarian, 2.649/18.082 mau se bi gan sai, tuong duong 14,65%.

Day la con so phai bat hoc thuoc.

Cach noi de nguoi hoc hieu:

"Khong the noi cu khac not_humanitarian thi chac la informative. Bo du lieu co nhan informative rieng. Phai lay nhan chinh thuc, khong tu che nhan."

## Cach pipeline join dung

Ba buoc:

1. Dung partition humanitarian lam ranh gioi train/dev/test chung.
2. Gom annotation informative thanh bang tra cuu theo `(tweet_id, image_id)`.
3. Left join `label` informative chinh thuc vao tung partition.

Y nghia:

- Giu du 18.082 dong.
- Hai task dung chung split.
- Khong suy dien nhan sai.

## Kich thuoc split

Sau hop nhat:

| Split | So cap tweet-anh |
|---|---:|
| Train | 13.608 |
| Dev | 2.237 |
| Test | 2.237 |

Sau exact duplicate mask:

| Split | So hang sach |
|---|---:|
| Train | 13.608 |
| Dev | 2.189 |
| Test | 2.169 |

Sau robustness mask:

| Split | So hang |
|---|---:|
| Train | 13.608 |
| Dev | 2.078 |
| Test | 2.032 |

## Train/dev/test giai thich bang doi thuong

### Train

Noi:

> Train la vo bai tap de model hoc.

Dung train de:

- fit TF-IDF;
- fit classifier;
- calibration noi bo;
- EDA phuc vu thiet ke.

### Dev

Noi:

> Dev la de thi thu de chon cach lam.

Dung dev de:

- chon model;
- chon hyperparameter;
- chon fusion weight;
- chon threshold;
- chon conflict threshold.

### Test

Noi:

> Test la de thi cuoi. Chi mo sau khi da chot cach lam.

Dung test de:

- bao cao ket qua cuoi;
- khong tuning;
- khong doi threshold sau khi nhin test.

## Bai tap

Hoi nguoi hoc:

Neu co:

```text
tweet_id = 10, image_id = A
tweet_id = 10, image_id = B
```

Day la mot mau hay hai mau?

Dap an: hai mau, vi cap tweet-anh khac nhau.

Hoi tiep:

Neu mot mau co `label_top = not_humanitarian`, co duoc tu suy ra `label = not_informative` khong?

Dap an: khong. Phai dung `label` chinh thuc.

## Loi hieu sai can chan

1. "tweet_id la khoa duy nhat."
   - Sai. Khoa la `(tweet_id, image_id)`.

2. "Humanitarian khac not_humanitarian la informative."
   - Sai. Bao cao da chi ra 2.649 mau se sai neu suy dien.

3. "EDA co the xem test cho biet du lieu."
   - Sai neu EDA dung de quyet dinh model. EDA thiet ke chi dung train.

4. "Dev va test giong nhau."
   - Sai. Dev de chon, test de bao cao.

## Cau hoi kiem tra

1. CrisisMMD co bao nhieu cap tweet-anh?
2. Don vi quan sat la gi?
3. `label` dung cho task nao?
4. `label_top` dung cho task nao?
5. Tai sao khong suy `label` tu `label_top`?
6. Train/dev/test moi cai dung de lam gi?

