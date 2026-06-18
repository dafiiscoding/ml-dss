# Buoi 10 - Metric, baseline va mat can bang lop

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> Khong duoc nhin moi Accuracy hay F2. Du an co du lieu mat can bang, nen phai doc ket qua cung baseline, Precision, Recall, F1/F2, Balanced Accuracy, MCC, Macro-F1 va metric theo lop.

## Tu khoa

- Confusion matrix
- TP
- FP
- FN
- TN
- Accuracy
- Precision
- Recall
- F1
- F2
- Balanced Accuracy
- MCC
- Macro-F1
- Weighted-F1
- Baseline
- Majority baseline
- Always-informative baseline

## Vi sao can metric

Model khong chi "dung" hoac "sai". Can biet sai kieu nao.

Trong tham hoa:

- Bo sot mot loi keu cuu co the rat nguy hiem.
- Chuyen nham mot bai it khan cap cho nguoi review cung ton cong nhung it nguy hiem hon.

Vay Recall co the quan trong hon Precision o tang sang loc.

## Confusion matrix

Voi task informative:

| | Du bao informative | Du bao not-informative |
|---|---:|---:|
| Thuc te informative | TP | FN |
| Thuc te not-informative | FP | TN |

Giai thich:

- TP: bai huu ich va model doan huu ich.
- FP: bai khong huu ich nhung model bao huu ich.
- FN: bai huu ich nhung model bo sot.
- TN: bai khong huu ich va model bo qua dung.

## Vi du 100 mau

| | Du bao informative | Du bao not-informative |
|---|---:|---:|
| Thuc te informative | 55 | 5 |
| Thuc te not-informative | 15 | 25 |

Tinh:

```text
Accuracy = (55 + 25) / 100 = 0,80
Precision = 55 / (55 + 15) = 0,79
Recall = 55 / (55 + 5) = 0,92
```

Dien giai:

- Recall cao: bat duoc hau het ca informative.
- Precision thap hon: co mot so ca bao nham informative.

## Accuracy

Accuracy = tong du bao dung / tong mau.

Manh:

- de hieu.

Yeu:

- bi lop lon chi phoi.

Vi du:

Neu 95% mau la lop A, model doan tat ca A co Accuracy 95%, nhung vo dung voi lop B.

## Precision

Precision tra loi:

> Trong nhung ca model bao positive, bao nhieu ca thuc su positive?

Trong du an:

Neu Precision informative thap, hang doi co nhieu bai khong huu ich.

## Recall

Recall tra loi:

> Trong nhung ca positive that, model bat duoc bao nhieu?

Trong du an:

Recall informative cao nghia la bo sot it thong tin huu ich.

## F1 va F2

F1 can bang Precision va Recall.

F2 nhan manh Recall hon Precision.

Vi sao dung F2 cho informative?

> Bo sot thong tin khan cap co the ton kem hon chuyen nham mot bai cho nguoi kiem tra.

Nhung F2 co bay:

Neu informative la lop da so, model luon doan informative co Recall = 1, nen F2 rat cao.

## Baseline la gi

Baseline la moc doi chung don gian.

Hoi:

> Model co that su hon cach ngu ngoc nhat khong?

Trong du an:

- Informative baseline: always-informative.
- Humanitarian baseline: majority class.

## Always-informative baseline

Model nay doan tat ca mau la informative.

Ket qua canonical test:

- F2 = 0,8939.
- MCC = 0.

Vi sao F2 cao?

Vi Recall = 1, bat tat ca informative that, nhung no cung bao nham tat ca not-informative.

Day cau quan trong:

> F2 cao khong co nghia model thong minh neu baseline ngu cung F2 cao.

## MCC la gi

MCC = Matthews Correlation Coefficient.

Khong can bat nguoi hoc tinh cong thuc. Noi:

> MCC do model co phan biet hai lop that khong. Du bao hang co MCC = 0.

Trong du an:

- Always-informative MCC = 0.
- Late Fusion MCC = 0,3325.

Y nghia:

Fusion co phan biet hai lop hon du bao hang.

## Balanced Accuracy

Balanced Accuracy la trung binh Recall cua cac lop.

No giup chong viec lop lon che lop nho.

Trong informative:

- Always-informative Balanced Accuracy = 0,5000.
- Late Fusion Balanced Accuracy = 0,5944.

## Macro-F1

Dung cho 8 lop humanitarian.

Macro-F1:

1. Tinh F1 rieng tung lop.
2. Lay trung binh deu.

Y nghia:

> Lop missing/found it mau van co tieng noi ngang lop not-humanitarian nhieu mau.

## Weighted-F1

Weighted-F1 lay trung binh F1 theo so mau moi lop.

Y nghia:

- Phan anh hieu nang theo ty le thuc te.
- Bi lop lon anh huong hon Macro-F1.

## Metric trong du an dung the nao

Task informative:

- chon theo F2 tren dev;
- bao cao Accuracy;
- Balanced Accuracy;
- Precision;
- Recall;
- F1;
- F2;
- MCC.

Task humanitarian:

- chon theo Macro-F1 tren dev;
- bao cao Accuracy;
- Macro-F1;
- Weighted-F1;
- ket qua theo lop.

## Bai tap

Cho model A:

```text
F2 = 0,90
MCC = 0
```

Hoi: co tot khong?

Dap an:

Chua. MCC = 0 co the la du bao hang. Phai so baseline.

Cho model B:

```text
Accuracy = 0,90
Macro-F1 = 0,10
```

Hoi: van de gi?

Dap an:

Co the model chi doan lop lon, bo qua lop hiem.

## Loi hieu sai can chan

1. "Accuracy cao la tot."
   - Sai neu du lieu mat can bang.

2. "F2 cao la du."
   - Sai. Phai so baseline va xem MCC/Balanced Accuracy.

3. "Macro-F1 thap nghia la model vo dung."
   - Chua chac. Can xem lop hiem, support, muc dich he thong.

4. "Weighted-F1 va Macro-F1 giong nhau."
   - Sai. Weighted-F1 uu tien lop nhieu mau.

## Cau hoi kiem tra

1. TP, FP, FN, TN la gi?
2. Precision tra loi cau hoi nao?
3. Recall tra loi cau hoi nao?
4. Vi sao F2 informative co the gay ao tuong?
5. Baseline la gi?
6. Macro-F1 vi sao hop voi du lieu mat can bang?

