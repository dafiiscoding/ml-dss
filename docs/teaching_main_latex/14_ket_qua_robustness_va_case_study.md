# Buoi 14 - Ket qua cuoi, robustness, bootstrap va case study

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> Ket qua cua du an phai doc cung baseline va gioi han. Informative fusion co F2 cao nhung chi hon always-informative 0,0106. Humanitarian fusion cai thien ro hon majority baseline. Robust test va bootstrap cho thay ket luan tong the on, nhung lop hiem va domain shift van la gioi han lon.

## Tu khoa

- Canonical test
- Robust test
- Bootstrap
- Confidence interval
- Event stability
- Class stability
- Case study
- Baseline gain
- Support

## Ket qua end-to-end

Pipeline hoan chinh:

```text
du lieu -> dac trung -> xac suat -> fusion -> chinh sach DSS -> hang doi
```

Integration test that:

- Bai dang Hurricane Harvey ve hoat dong cuu tro.
- He thong du bao dung rescue/donation.
- Risk Score = 54,3.
- Priority = Medium.
- Routing = Relief Team.

Can noi:

> Integration test chung minh module noi duoc voi nhau tren mot anh that, khong thay the danh gia thong ke tren toan test.

## Dashboard workload vs metric khoa hoc

Dashboard chay tren 2.237 dong test de nguoi dung co batch day du:

- 50 case High.
- 571 case Manual Review.

Metric khoa hoc tinh tren canonical test 2.169 dong:

- 49 High.
- 539 Manual Review.

Can day:

> So dong dashboard va so dong metric khac nhau vi metric da loc duplicate theo canonical mask.

## Ket qua informative

Canonical test 2.169 hang:

| He thong | Thr | Acc | Bal. Acc | Prec | Recall | F1 | F2 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always informative | - | .6275 | .5000 | .6275 | 1.0000 | .7711 | .8939 | .0000 |
| Text-only | .38 | .6823 | .5792 | .6675 | .9838 | .7954 | .8987 | .2904 |
| Image-only | .20 | .6773 | .5759 | .6662 | .9735 | .7910 | .8913 | .2660 |
| Late Fusion | .38 | .6948 | .5944 | .6755 | .9882 | .8025 | .9045 | .3325 |

## Cach doc ket qua informative

Sai:

> F2 = 0,9045 nen model rat tot.

Dung:

> F2 = 0,9045, nhung always-informative da 0,8939. Gain F2 chi 0,0106. Gia tri that ro hon o Accuracy, F1, Balanced Accuracy va MCC.

Gain so voi always-informative:

- F2: +0,0106.
- F1: +0,0314.
- Accuracy: +0,0673.
- Balanced Accuracy: +0,0944.
- MCC: +0,3325.

Ket luan dung:

> Fusion co phan biet hai lop hon du bao hang, nhung informative task khong phai thanh tich qua manh neu chi nhin F2.

## Ket qua humanitarian

Canonical test:

| He thong | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Majority baseline | .3785 | .0686 | .2079 |
| Text-only | .4481 | .3260 | .4693 |
| Image-only | .5440 | .3646 | .5695 |
| Late Fusion | .5569 | .4005 | .5776 |

Dien giai:

- Fusion cao nhat.
- Image-only tot hon text-only o Macro-F1.
- Fusion hon image-only 0,0359 Macro-F1.
- So voi majority baseline, cai thien rat ro.

## Ket qua theo lop

Late Fusion humanitarian:

| Lop | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| injured/dead | .2202 | .6316 | .3265 | 38 |
| missing/found | .0435 | .2000 | .0714 | 5 |
| rescue/donation | .5262 | .5468 | .5363 | 331 |
| infrastructure | .4925 | .5305 | .5108 | 311 |
| affected | .2124 | .5000 | .2982 | 82 |
| vehicle damage | .1385 | .5000 | .2169 | 18 |
| other relevant | .6588 | .4938 | .5645 | 563 |
| not humanitarian | .7507 | .6200 | .6791 | 821 |

Can noi:

> Lop missing/found chi co 5 mau test, F1 = 0,0714. Khong du tin de tu dong dieu dong.

## Robust test

Robustness mask loai them 137 hang test gan trung da review, con 2.032.

Model, fusion, threshold giu nguyen, khong tune lai.

| Metric | Canonical | Robust | Chenh |
|---|---:|---:|---:|
| Informative Accuracy | .6948 | .6900 | -.0048 |
| Informative F1 | .8025 | .7978 | -.0047 |
| Informative F2 | .9045 | .9016 | -.0029 |
| Informative MCC | .3325 | .3340 | +.0015 |
| Humanitarian Macro-F1 | .4005 | .3908 | -.0097 |
| Humanitarian Weighted-F1 | .5776 | .5789 | +.0013 |
| Manual Review F1 | .4585 | .4558 | -.0027 |

Ket luan:

> Ket luan tong the khong phu thuoc manh vao near-duplicate da xac nhan.

## Bootstrap la gi

Noi don gian:

> Bootstrap la boc tham co hoan lai tu test nhieu lan de xem metric dao dong ra sao.

Du an dung bootstrap phan tang 2.000 lan tren robust test.

Ket qua:

| Dai luong | Uoc luong | CI thap | CI cao |
|---|---:|---:|---:|
| Informative F2 | .9016 | .8960 | .9070 |
| F2 gain so voi always-informative | .0110 | .0054 | .0163 |
| Humanitarian Macro-F1 | .3908 | .3636 | .4208 |
| Macro-F1 gain so voi majority | .3208 | .2936 | .3508 |
| Manual Review F1 | .4558 | .4251 | .4868 |

Dien giai:

- CI gain khong chua 0 theo gia dinh lay mau theo hang.
- Nhung row-bootstrap khong mo phong tham hoa hoan toan moi.
- Nhieu tweet trong cung event co tuong quan.

## Domain shift con lai

Bao cao noi:

- Informative F2 theo event: 0,8205 den 0,9243.
- Humanitarian Macro-F1 theo event hien dien: 0,3458 den 0,4615.

Can day:

> Metric gop co the che dau su khac nhau giua cac tham hoa.

## Manual Review bang so ca

Canonical test:

- 2.169 dong.
- 539 ca vao review.
- Precision .7532.
- Recall .3295.
- F1 .4585.

Dien giai bang so:

- Khoang 406/539 ca review that su bat dong.
- Khoang 133 ca review la bao nham.
- Chi bat duoc khoang 1/3 tong ca bat dong.

Ket luan:

> Manual Review la triage trong gioi han cong suat, khong phai detector bat het xung dot.

## Case study trong ban main

Ban main co 3 ca minh hoa:

### Ca A - Dong thuan cao

- Tin dong thuan.
- Risk 87,5.
- Priority High.
- Khong can review.

Dien giai:

Text va anh cung ung ho muc nghiem trong. He thong dua len dau hang doi.

### Ca B - Conflict cao

- Conflict 0,92.
- Bat Manual Review.
- Van uu tien neu noi dung co nguy co.

Dien giai:

Mau thuan khong lam he thong bo qua. No lam he thong goi supervisor.

### Ca C - Phu dinh "no missing child"

- Co tu `missing`.
- Lop missing/found hiem.
- He thong chuyen review thay vi tin tuyet doi.

Dien giai:

Day la minh hoa diem yeu TF-IDF voi phu dinh.

## Dieu da chung minh va chua chung minh

Da co bang chung:

- Fusion hon dummy o nhieu metric va hon tung nhanh o humanitarian.
- Classifier so sanh va tune tren dev.
- Conflict rule co Precision .7532 duoi capacity 25%.
- Routing phu 8 category va qua scenario test.
- Robustness/CI khong dao nguoc ket luan tong the.
- Dashboard chay voi model/cache da khoa.

Chua co bang chung:

- F2 gain 0,01 tao loi ich van hanh lon.
- Cau hinh tong quat sang tham hoa chua thay.
- Manual Review bat het moi xung dot quan trong.
- Priority threshold/severity weight toi uu thuc te.
- Lop missing du tin cay de tu dong dieu dong.
- He thong dap ung bao mat/tai/do tre production.

## Bai tap

Hoi:

Neu nguoi hoc noi "F2 0,9045 nen model informative rat xuat sac", sua the nao?

Dap an:

Phai so voi always-informative 0,8939. Gain F2 chi 0,0106; gia tri ro hon la MCC tang tu 0 len 0,3325 va Balanced Accuracy tang.

Hoi:

Neu Macro-F1 humanitarian 0,4005 co nen that vong khong?

Dap an:

Khong doc rieng. No hon majority 0,0686 va hon text/image-only, nhung lop hiem van yeu.

## Loi hieu sai can chan

1. "F2 cao nen model qua tot."
   - Sai neu baseline cung cao.

2. "Robust test giam it nen model chac chan dung ngoai doi."
   - Sai. Robust test khong thay leave-one-event-out.

3. "Precision review 0,7532 nghia la bat het xung dot."
   - Sai. Recall chi 0,3295.

4. "Lop missing/found co Recall 0,2 nen dung duoc."
   - Sai. Support 5, F1 0,0714, precision rat thap.

## Cau hoi kiem tra

1. Informative fusion F2 bao nhieu?
2. Always-informative F2 bao nhieu?
3. Humanitarian fusion Macro-F1 bao nhieu?
4. Robust test con bao nhieu dong?
5. Bootstrap cho ta biet gi?
6. Ca "no missing child" minh hoa diem yeu nao?

