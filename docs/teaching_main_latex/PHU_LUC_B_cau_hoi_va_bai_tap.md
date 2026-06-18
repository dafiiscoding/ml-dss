# Phu luc B - Cau hoi kiem tra va bai tap

## De kiem tra nhanh 20 cau

1. Du an co thu thap tweet live khong?
2. CrisisMMD co bao nhieu cap tweet-anh?
3. Khoa dung cua mot mau la gi?
4. `label` dung cho task nao?
5. `label_top` dung cho task nao?
6. Vi sao khong suy `label` tu humanitarian?
7. Train/dev/test dung de lam gi?
8. Data leakage la gi?
9. SHA-256 bat loai trung lap nao?
10. pHash khac SHA-256 o dau?
11. TF-IDF la gi?
12. CLIP trong du an co duoc fine-tune khong?
13. Embedding anh co bao nhieu chieu?
14. EDA thiet ke co dung test khong?
15. Vi sao Accuracy khong du voi du lieu mat can bang?
16. Baseline informative la gi?
17. Late Fusion la gi?
18. Conflict score dung de lam gi?
19. Risk Score co phai model hoc tu ground truth khong?
20. Dashboard co phai production system khong?

## Dap an ngan

1. Khong, du an doc CrisisMMD da luu san.
2. 18.082.
3. `(tweet_id, image_id)`.
4. Informative.
5. Humanitarian 8 lop.
6. Vi se sai 2.649 mau neu suy dien.
7. Train hoc, dev chon, test bao cao cuoi.
8. Thong tin dev/test di vao hoc/chon model.
9. Anh giong het byte.
10. pHash bat anh trong gan giong, SHA-256 bat file giong het.
11. Trong so tu theo tan suat trong tweet va do hiem tren train.
12. Khong, CLIP frozen.
13. 512.
14. Khong.
15. Lop lon co the che lop hiem.
16. Always-informative.
17. Tron xac suat text va anh sau khi du bao rieng.
18. Phat hien bat dong text-anh de Manual Review.
19. Khong, la policy.
20. Khong, la prototype hoc thuat.

## Bai tap tinh toan 1 - TF-IDF

Cho 3 tweet:

```text
d1 = "flood rescue now"
d2 = "flood power outage"
d3 = "donation drive"
```

Hoi:

1. df(flood) bang bao nhieu?
2. df(rescue) bang bao nhieu?
3. Tu nao co IDF cao hon?

Dap an:

1. df(flood) = 2.
2. df(rescue) = 1.
3. rescue co IDF cao hon vi hiem hon.

## Bai tap tinh toan 2 - Fusion informative

Cho:

```text
p_text = 0,85
p_image = 0,30
alpha = 0,70
threshold = 0,38
```

Hoi:

1. p_fused bang bao nhieu?
2. Co du bao informative khong?
3. Conflict informative bang bao nhieu?
4. Neu conflict threshold = 0,54, co review khong?

Dap an:

```text
p_fused = 0,70*0,85 + 0,30*0,30 = 0,685
```

2. Co, vi 0,685 >= 0,38.
3. Conflict = |0,85 - 0,30| = 0,55.
4. Co, vi 0,55 > 0,54.

## Bai tap tinh toan 3 - Risk Score

Cho:

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
     = 80
```

Priority = High.

Hoi them:

High co phai lenh dieu dong khong?

Dap an: Khong, High la uu tien trong hang doi.

## Bai tap dien giai metric

Cho:

```text
Always-informative F2 = 0,8939
Late Fusion F2 = 0,9045
Late Fusion MCC = 0,3325
```

Hoi:

Nen trinh bay ket qua the nao?

Dap an:

Khong khoe F2 rieng. Noi Fusion chi tang F2 0,0106 so voi baseline, nhung cai thien ro hon o MCC, Accuracy, Balanced Accuracy va F1.

## Bai tap tinh huong

### Tinh huong A

Tweet:

```text
"Urgent help needed, people trapped"
```

Anh: selfie.

Hoi:

- Priority co nen ha vi anh khong lien quan khong?
- Co Manual Review khong?

Dap an:

- Khong tu dong ha neu text co nguy co cao.
- Co Manual Review vi conflict.

### Tinh huong B

Tweet:

```text
"No missing child reported"
```

Anh: tre em.

Hoi:

- TF-IDF co the hieu sai tu nao?
- Nen lam gi?

Dap an:

- Tu `missing`.
- Dua vao review neu model du bao missing/found, khong tu dong hanh dong.

### Tinh huong C

Anh giong het anh train theo SHA-256, nhung nam trong test.

Hoi:

- Co tinh metric canonical khong?

Dap an:

Khong, bi canonical mask loai.

## Cau hoi van dap nang cao

1. Neu CLIP image-only co Average Precision cao hon text, vi sao threshold accuracy van gan text?
2. Vi sao row-bootstrap khong thay duoc leave-one-event-out?
3. Vi sao pHash candidate khong tu dong xoa?
4. Vi sao K-Means silhouette thap khong co nghia TF-IDF vo dung?
5. Vi sao CSV phu hop prototype nhung khong phu hop production?

## Goi y tra loi nang cao

1. Average Precision do chat luong xep hang, con threshold accuracy la chat luong tai mot nguong cu the. Hai khai niem khac nhau.
2. Row-bootstrap lay mau lai theo dong, nhung tweet trong cung event co tuong quan. Event moi la mot phan bo khac.
3. Gan trung khong chac trung that. Can review thi giac/nghia truoc khi loai.
4. K-Means la clustering khong giam sat va gia dinh cum dang cau. Classifier co nhan co the hoc ranh gioi tot hon.
5. CSV don gian, bat bien, de tai lap. Production can concurrent write, transaction, auth, audit log va index.

