# Buoi 13 - Risk Score, Priority, Routing va ranh gioi policy

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> Risk Score va Priority la chinh sach DSS minh bach, khong phai model da hoc tu ground truth Priority. Chung dung de xep hang va goi y routing cho con nguoi, khong tu dong dieu dong nguon luc.

## Tu khoa

- Risk Score
- Priority
- Routing
- Severity
- Keyword score
- Category confidence
- Weighted scoring model
- Policy
- Ground truth
- Sensitivity analysis
- Override

## Vi sao can tang DSS sau model

Model cho:

- xac suat informative;
- xac suat 8 lop humanitarian;
- category du doan;
- conflict score.

Nhung nguoi dieu phoi can:

- xem cai nao truoc;
- doi nao phu trach;
- co can supervisor khong;
- ly do he thong goi y.

Vay can tang DSS.

## Ground truth Priority co khong?

Khong.

CrisisMMD co nhan:

- informative;
- humanitarian category.

Khong co nhan:

- Low/Medium/High do chuyen gia nghiep vu gan;
- doi nao that su xu ly;
- thoi gian phan ung;
- ket qua ngoai doi.

Do do:

> Risk Score/Priority khong the duoc goi la toi uu thong ke.

## Weighted scoring model

Noi don gian:

> Weighted scoring model la cong thuc cong diem nhieu tieu chi, moi tieu chi co trong so.

Cong thuc trong du an:

```text
Risk = 0,40*S_informative
     + 0,25*S_severity
     + 0,20*S_keyword
     + 0,15*(S_severity * C_category)
```

## Giai thich tung thanh phan

### S_informative

Xac suat fusion informative dua ve thang 0-100.

Hoi:

> Bai nay co kha nang huu ich khong?

Trong so 0,40 la lon nhat.

### S_severity

Diem nghiem trong theo category.

Vi du truc giac:

- injured/dead cao;
- missing/found cao;
- infrastructure kha cao;
- donation/rescue trung binh;
- not humanitarian thap.

Hoi:

> Neu dung category nay, no nghiem trong den muc nao?

### S_keyword

Diem tu khoa khan cap.

Vi du:

- urgent;
- trapped;
- help;
- rescue;
- dead;
- missing.

Can can than:

Tu khoa khong hieu phu dinh. `no missing child` van co tu `missing`.

### C_category

Do tin cay cua category.

Nhan voi severity de tranh truong hop:

- severity cao nhung model khong chac;
- hoac model chac la not-humanitarian nhung khong nen tang risk.

## Vi du tinh tay

Tweet:

```text
"bridge collapsed, people trapped"
```

Gia su:

```text
S_informative = 88
S_severity = 80
S_keyword = 70
C_category = 0,9
```

Tinh:

```text
Risk = 0,40*88 + 0,25*80 + 0,20*70 + 0,15*(80*0,9)
     = 35,2 + 20 + 14 + 10,8
     = 80,0
```

Risk 80 nam trong High.

Nhac lai:

> High la uu tien doc/xac minh, khong phai lenh dieu dong.

## Priority

Nguong:

- 0-39: Low.
- 40-69: Medium.
- 70-100: High.

Trong current policy tren canonical test:

- 624 Low sau override conflict.
- 1.496 Medium.
- 49 High.
- 539 Manual Review.

## Routing

Moi category co doi mac dinh.

Vi du:

| Category | Team goi y |
|---|---|
| injured/dead | Emergency Team |
| missing/found | Emergency Team |
| rescue/donation | Relief Team |
| infrastructure | Infrastructure Team |
| vehicle damage | Infrastructure Team |
| affected | Relief/Coordination tuy policy |
| other relevant | Coordination Team |
| not humanitarian | Coordination/Low |

Neu conflict > 0,54:

- them Supervisor song song.

## Override

Override la quy tac uu tien an toan.

Vi du:

- Case High va conflict cao.
- Khong ha xuong Low chi vi conflict.
- Gui Supervisor + team goc.

Noi:

> Review song song, khong chan phan ung khan cap.

## Sensitivity analysis

Sensitivity analysis = thu xem doi nguong thi ket qua ra sao.

Ban main co 3 policy:

| Policy | Low max | Medium max | Low | Medium | High |
|---|---:|---:|---:|---:|---:|
| Sensitive | 29 | 59 | 651 | 1.186 | 332 |
| Current | 39 | 69 | 769 | 1.351 | 49 |
| Strict | 49 | 79 | 1.212 | 955 | 2 |

Dien giai:

- Sensitive tao nhieu High hon.
- Strict tao rat it High.
- Current la policy hien tai.

Ket luan:

> Priority phu thuoc manh vao lua chon to chuc. Muon khang dinh dung, can du lieu nghiep vu that.

## Ba phan vi du can day

### Phan vi 1 - Phu dinh lam keyword sai

Tweet:

```text
"thank god nobody died"
```

Co tu `died`, keyword co the tang.

Can con nguoi review neu model day len risk.

### Phan vi 2 - Not-humanitarian nhung conflict cao

Neu text va anh mau thuan, khong nen bo qua ngay. Co the review.

### Phan vi 3 - Lop hiem xac suat cao

Missing/found co rat it mau. Xac suat cao khong co nghia dang tin tuyet doi.

Phai xac minh.

## Policy khac model the nao

### Model

- Hoc tu nhan co san.
- Co metric test.
- Vi du: classifier informative, classifier humanitarian.

### Policy

- Do con nguoi thiet ke.
- Khong co ground truth trong dataset.
- Kiem tra bang scenario va sensitivity.
- Vi du: trong so Risk, nguong Priority, mapping Routing.

Bang can nho:

| Noi dung | Da co ground truth? | Co the khang dinh toi uu? |
|---|---|---|
| Informative classifier | Co | Co the danh gia bang metric |
| Humanitarian classifier | Co | Co the danh gia bang metric |
| Conflict detector | Co nhan bat dong chan doan | Co the danh gia han che |
| Risk Score | Khong | Khong |
| Priority | Khong | Khong |
| Routing policy | Khong | Khong |

## Bai tap

Cho:

```text
S_informative = 60
S_severity = 90
S_keyword = 20
C_category = 0,5
```

Tinh Risk:

```text
0,40*60 = 24
0,25*90 = 22,5
0,20*20 = 4
0,15*(90*0,5) = 6,75
Tong = 57,25
```

Priority: Medium.

Hoi:

Neu conflict cao, co them ai?

Dap an:

Them Supervisor.

## Loi hieu sai can chan

1. "Risk Score la model hoc tu du lieu."
   - Sai. La policy trong so.

2. "Priority toi uu thong ke."
   - Sai. Khong co ground truth Priority.

3. "High la lenh dieu dong."
   - Sai. High la uu tien trong hang doi.

4. "Conflict cao thi khong tin nua."
   - Sai. Conflict cao thi review, khong nhat thiet ha risk.

## Cau hoi kiem tra

1. Risk Score gom nhung thanh phan nao?
2. Priority Low/Medium/High chia nguong nao?
3. Vi sao Risk/Priority khong phai toi uu thong ke?
4. Routing la gi?
5. Khi conflict cao, them team nao?
6. Sensitivity analysis dung de lam gi?

