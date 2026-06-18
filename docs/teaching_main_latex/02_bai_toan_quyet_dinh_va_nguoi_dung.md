# Buoi 2 - Bai toan quyet dinh, nguoi dung va nam hanh dong DSS ho tro

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> He thong ho tro 5 viec: loc bai dang, goi y nhom noi dung, xep uu tien, goi y doi tiep nhan va danh dau ca can nguoi kiem tra.

## Tu khoa

- Sàng lọc
- Humanitarian category
- Priority
- Routing
- Manual Review
- Ban dieu phoi
- Emergency Team
- Relief Team
- Infrastructure Team
- Supervisor

## On lai 3 phut

Hoi:

"Buoi truoc ta da biet du an khong thay hotline. Vay no lam gi?"

Dap an:

"No sap xep tweet-anh thanh hang doi co can cu de nguoi xu ly doc truoc."

## Nam quyet dinh he thong ho tro

### 1. Sang loc

Cau hoi:

> Bai nay co dang chu y trong ung pho tham hoa khong?

Trong bao cao goi la `informative`.

Vi du:

- "Bridge collapsed, people trapped" la informative.
- "My heart goes out to everyone" co the it useful hon cho dieu phoi.

Day nguoi hoc:

> Sang loc khong co nghia xoa bai. No chi la giup nguoi doc uu tien bai co thong tin hanh dong duoc.

### 2. Goi y loai nhu cau

Cau hoi:

> Bai dang noi ve loai van de nao?

Cac nhom humanitarian trong du an gom 8 lop:

1. injured/dead: nguoi bi thuong hoac tu vong.
2. missing/found: nguoi mat tich hoac tim thay.
3. rescue/donation: cuu ho, tinh nguyen, quyen gop.
4. infrastructure: ha tang, dien, duong, cau, cong trinh.
5. affected: nguoi bi anh huong.
6. vehicle damage: phuong tien hu hong.
7. other relevant: lien quan khac.
8. not humanitarian: khong mang tinh nhan dao.

### 3. Xep muc uu tien

Cau hoi:

> Bai nao can xem truoc?

Muc uu tien:

- Low
- Medium
- High

Day bang vi du:

"High khong co nghia xe cuu ho tu chay di. High chi co nghia trong dashboard, dong nay nam tren dau hang doi."

### 4. Routing

Cau hoi:

> Neu phai xu ly, ho so nay nen gui den nhom nao?

Vi du:

- Thuong vong, mat tich: Emergency Team.
- Do an, cuu tro, quyen gop: Relief Team.
- Cau, duong, dien, xe hu: Infrastructure Team.
- Noi dung chung, khong ro: Coordination Team.
- Bat dong cao: them Supervisor.

Can noi ro:

> Routing la goi y nhom ho so, khong phai gui lenh ngoai doi.

### 5. Manual Review

Cau hoi:

> Ca nay co can nguoi xem lai khong?

Bat Manual Review khi:

- text va anh mau thuan;
- model khong chac;
- ca co hau qua cao;
- lop hiem;
- tweet co phu dinh/mia mai;
- thieu vi tri.

## Nguoi dung cua he thong

### Ban dieu phoi

Can xem:

- co bao nhieu ca High;
- bao nhieu ca can review;
- workload theo event;
- nhom nao dang qua tai.

### Emergency Team

Can xem:

- injured/dead;
- missing/found;
- Risk cao;
- vi tri neu co;
- noi dung goc va anh.

### Infrastructure Team

Can xem:

- duong sap;
- cau sap;
- mat dien;
- xe hu;
- ha tang cong cong.

### Relief Team

Can xem:

- can do an;
- can nuoc;
- can cho o;
- quyen gop;
- tinh nguyen.

### Supervisor

Can xem:

- conflict score;
- xac suat text;
- xac suat image;
- anh va tweet goc;
- ly do he thong bat review.

## Vi du xuyen suot

Tweet:

```text
"Urgent! People are trapped after the bridge collapsed in rising flood water. Need rescue now."
```

Anh: duong ngap.

Xu ly:

1. Sang loc: informative.
2. Nhom: infrastructure hoac rescue related.
3. Priority: High neu risk cao.
4. Routing: Emergency Team + Infrastructure Team tuy policy.
5. Manual Review: neu text va anh dong thuan thi co the khong review; neu anh khong lien quan thi bat review.

## Cac tinh huong thuc te ban main them vao

### Tinh huong 1 - Text khan cap nhung anh khong lien quan

Vi du:

```text
"Urgent help needed!"
```

Anh selfie.

Phan ung dung:

- Khong bo qua text.
- Giu uu tien neu text khan cap.
- Bat Manual Review vi text-anh mau thuan.

### Tinh huong 2 - Anh thiet hai ro nhung text chung chung

Vi du:

```text
"This is terrible"
```

Anh cau sap.

Phan ung dung:

- Tin rang anh co bang chung rieng.
- Khong phu thuoc hoan toan vao text.
- Cho nhánh anh dong gop qua Late Fusion.

### Tinh huong 3 - Anh cu dang lai

Phan ung dung:

- Dung hash de phat hien anh trung/gần trùng.
- Khong tinh nhu bang chung moi neu da thay o split truoc khi danh gia.

### Tinh huong 4 - Tweet thieu vi tri

Phan ung dung:

- Van co the xep uu tien.
- Nhac con nguoi xac minh vi tri truoc hanh dong.

### Tinh huong 5 - Phu dinh hoac mia mai

Vi du:

```text
"thank god nobody died"
```

TF-IDF thay tu `died`, co the hieu sai.

Phan ung dung:

- Neu nghi ngo, dua vao Manual Review.
- Khong tin tuyet doi keyword.

## Bai tap tai lop

Cho bang:

| Tweet | Anh | Hoi |
|---|---|---|
| "Need rescue now" | anh selfie | co review khong? |
| "Power outage downtown" | anh den duong toi | team nao? |
| "No missing child found here" | anh tre em | co nguy co hieu sai khong? |
| "Donate food at shelter" | anh thung hang | priority nao? |

Dap an:

- Dong 1: review vi conflict.
- Dong 2: Infrastructure Team.
- Dong 3: nguy co phu dinh, review.
- Dong 4: Relief Team, thuong Medium neu khong co tu khan cap.

## Loi hieu sai can chan

1. "Manual Review la model sai."
   - Sai. Manual Review la co che quan tri rui ro.

2. "Priority High la hanh dong ngoai doi."
   - Sai. No la thu tu trong hang doi.

3. "Routing la gui lenh cho doi."
   - Sai. No la goi y nhom tiep nhan.

4. "Text va anh khac nhau thi mot cai sai."
   - Sai. Hai phuong thuc co the noi ve hai mat khac nhau cua cung su kien.

## Cau hoi kiem tra

1. Ke ten 5 quyet dinh he thong ho tro.
2. Ai xu ly ca conflict cao?
3. Vi sao Manual Review khong phai "nhan loi"?
4. Neu tweet la High nhung conflict cao thi co ha uu tien khong?
5. Lop missing/found co du tin de tu dong dieu dong khong?

