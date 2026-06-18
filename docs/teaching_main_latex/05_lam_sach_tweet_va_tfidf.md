# Buoi 5 - Lam sach tweet va TF-IDF

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> Model khong doc cau chu truc tiep. Du an lam sach tweet roi bien tweet thanh vector TF-IDF 2.000 chieu. TF-IDF cho trong so cao hon cho tu vua xuat hien trong tweet nay, vua khong qua pho bien o toan bo train.

## Tu khoa

- Tweet cleaning
- URL
- Mention
- Hashtag
- HTML entity
- Token
- Stopword
- OOV
- TF
- IDF
- TF-IDF
- Unigram
- Bigram
- Vector thua
- L2 normalization

## Vi sao phai lam sach tweet

Tweet ngoai doi co nhieu thu model khong nen hoc qua nhieu:

- URL rut gon;
- @mention;
- HTML entity;
- ky tu dac biet;
- hashtag co dau `#`;
- viet hoa/viet thuong lung tung;
- dau cau, emoji, ky tu la.

Neu dua thang vao model, model co the hoc nham:

- `https://t.co/...` la dau hieu thay vi noi dung that;
- mot username la dau hieu cua mot lop;
- cac ky tu rac lam phinh to tu vung.

## Ham lam sach lam gi

Theo ban main, ham lam sach di qua 5 buoc:

1. Chuyen chu thuong.
2. Bo HTML entity, URL va @mention.
3. Bo ky hieu `#` nhung giu tu trong hashtag.
4. Chi giu chu cai Latin, chu so va khoang trang.
5. Gop nhieu khoang trang lien tiep.

## Vi du

Truoc:

```text
URGENT: Flood waters rising! @Police https://t.co/x #Harvey
```

Sau:

```text
urgent flood waters rising harvey
```

Giai thich:

- `URGENT` thanh `urgent`.
- `@Police` bi bo.
- URL bi bo.
- `#Harvey` thanh `harvey`.
- Dau `:` va `!` bi bo.

## Vi sao giu hashtag

Hashtag khong chi la rac. No co the chua:

- ten tham hoa: `#Harvey`, `#Irma`;
- dia danh: `#PuertoRico`;
- chu de: `#flood`, `#earthquake`;
- hoat dong cuu tro: `#donate`, `#rescue`.

Nen bo dau `#`, nhung giu chu.

## Ba muc xu ly nhieu trong ban main

### Da xu ly truc tiep

- URL.
- @mention.
- HTML entity.
- Ky hieu `#`.
- Ky tu khong phai chu/so.

### Giam tac dong

- Token qua hiem bi cat boi gioi han 2.000 dac trung.
- Stopword tieng Anh bi loai.
- Tu ngoai tu vung khong lam model vo, chi khong co feature.

### Chua xu ly tot

- Emoji.
- Mia mai.
- Phu dinh.
- Loi chinh ta.
- Da ngon ngu.
- Spam lap tu.

## Audit that trong bao cao

Ban main co audit text:

- trung binh 40% ky tu moi tweet bi loai khi lam sach;
- khong tweet nao thanh chuoi rong sau lam sach;
- OOV dev/test la 7,6% va 7,8%.

Giai thich OOV:

> OOV la tu khong co trong tu dien da hoc tu train.

Vi du:

Train chua co tu `mudslide`, dev co `mudslide`. Tu nay la OOV, TF-IDF da fit tren train se khong co cot cho no.

## TF-IDF la gi

Noi bang ngon ngu doi thuong:

> TF-IDF la cach cho diem cac tu. Tu nao xuat hien trong tweet hien tai nhieu, nhung khong xuat hien o qua nhieu tweet khac, thi tu do co diem cao hon.

TF = Term Frequency:

- tu nay xuat hien bao nhieu lan trong tweet.

IDF = Inverse Document Frequency:

- tu nay hiem hay pho bien tren toan bo tap train.

TF-IDF = TF nhan IDF.

## Vi sao can TF-IDF

Model nhu Logistic Regression hay SVM khong doc chu. No doc vector so.

Tweet:

```text
need rescue flood water
```

Can bien thanh:

```text
[0, 0.42, 0, 0.71, 0.33, ...]
```

Moi cot la mot tu/cum tu.

## Vector thua la gi

TF-IDF co 2.000 dac trung. Mot tweet ngan chi co vai tu.

Neu tweet co 10 tu, trong 2.000 cot chi co vai cot khac 0. Con lai bang 0.

Vay goi la vector thua.

## Unigram va bigram

Unigram la mot tu:

- `power`
- `outage`
- `missing`
- `people`

Bigram la hai tu lien tiep:

- `power outage`
- `missing people`
- `need shelter`
- `flood water`

Vi sao dung ca hai?

- Unigram giu do phu.
- Bigram giu cum nghia quan trong.

## Stopword

Stopword la tu qua pho bien, it thong tin:

- the
- is
- a
- an
- of
- to

Bo stopword giup model tap trung vao tu co nghia hon.

## Vi du tinh tay

Dung vi du trong ban main.

Ba tweet da lam sach:

```text
d1 = "flood rescue now"
d2 = "flood power outage"
d3 = "donation drive"
```

N = 3.

Tu `flood` xuat hien trong 2 van ban, df = 2.

Tu `rescue` xuat hien trong 1 van ban, df = 1.

Cong thuc IDF smooth cua scikit-learn:

```text
idf(t) = ln((N + 1) / (df(t) + 1)) + 1
```

Tinh:

```text
idf(flood) = ln(4 / 3) + 1 = 1,288
idf(rescue) = ln(4 / 2) + 1 = 1,693
```

Neu ca hai tu xuat hien 1 lan trong d1:

```text
tfidf(flood, d1) = 1,288
tfidf(rescue, d1) = 1,693
```

Ket luan:

> `rescue` hiem hon `flood`, nen co trong so cao hon.

## Ma tran document-term

Chua chuan hoa:

| Document | flood | rescue | now | power | outage | donation | drive |
|---|---:|---:|---:|---:|---:|---:|---:|
| d1 | 1,29 | 1,69 | 1,69 | 0 | 0 | 0 | 0 |
| d2 | 1,29 | 0 | 0 | 1,69 | 1,69 | 0 | 0 |
| d3 | 0 | 0 | 0 | 0 | 0 | 1,69 | 1,69 |

## Vi sao chi fit TF-IDF tren train

Neu fit tren toan corpus, IDF se biet truoc tu nao xuat hien trong dev/test.

Nghe co ve nhe, nhung van la ro ri:

> Dev/test la de thi. Khong duoc dung thong tin de thi de lap tu dien va tan suat.

Quy trinh dung:

```text
train: fit_transform
dev: transform
test: transform
```

## Cau hinh that trong du an

- 2.000 dac trung.
- Word n-gram (1,2).
- Stopword tieng Anh.
- L2 normalization.
- Vectorizer fit tren train.

## Gioi han cua TF-IDF

TF-IDF khong hieu:

- phu dinh: "no missing child";
- mia mai: "great, another flood";
- thu tu xa trong cau;
- ngu canh ngoai van ban;
- hinh anh;
- dia diem that neu viet mo ho.

Vi du:

```text
"thank god nobody died"
```

TF-IDF thay tu `died`, co the day model ve lop injured/dead, du cau that su la phu dinh.

Day la ly do can:

- nhánh anh;
- Manual Review;
- khong tu dong hoa ca hau qua cao.

## Bai tap

Cho cau:

```text
"No one is missing after the flood, thank god"
```

Hoi:

1. Tu nao co the lam TF-IDF hieu nham?
2. Vi sao?
3. He thong nen lam gi neu ca nay duoc du bao missing/found?

Dap an:

1. `missing`, `flood`.
2. TF-IDF khong hieu phu dinh `No one`.
3. Dua vao Manual Review, khong tu dong hanh dong.

## Loi hieu sai can chan

1. "Lam sach xong la tweet hoan hao."
   - Sai. Van con mia mai, phu dinh, loi chinh ta.

2. "TF-IDF hieu nghia cau."
   - Sai. TF-IDF dem tu va trong so tu.

3. "Fit TF-IDF tren ca train/dev/test cung khong sao."
   - Sai. Do la ro ri nhe nhung co that.

4. "Tu hiem luon quan trong."
   - Sai. Tu hiem co the la loi chinh ta hoac rac, nen gioi han feature va min_df giup giam nhieu.

## Cau hoi kiem tra

1. Vi sao bo URL nhung giu noi dung hashtag?
2. TF la gi?
3. IDF la gi?
4. Bigram co ich o dau?
5. OOV la gi?
6. TF-IDF co hieu phu dinh khong?

