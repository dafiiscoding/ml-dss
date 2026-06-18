# Buoi 6 - Anh, CLIP, embedding 512 chieu va cache

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> Anh khong dua truc tiep vao classifier co dien bang pixel. Du an dung CLIP ViT-B/32 da huan luyen san de bien moi anh thanh embedding 512 chieu, giu CLIP frozen va chi huan luyen classifier phia sau.

## Tu khoa

- Pixel
- RGB
- Resize 224x224
- Patch
- Token anh
- Transformer
- CLIP
- ViT-B/32
- Embedding
- 512 chieu
- Frozen feature extractor
- Fine-tuning
- L2 normalization
- Cache

## Vi sao anh kho hon text

Mot anh co rat nhieu pixel. Vi du anh 224x224 RGB co:

```text
224 x 224 x 3 = 150.528 gia tri mau
```

Neu dua thang 150.528 so nay vao model co dien voi 13.608 mau train, model rat de:

- qua tai;
- hoc nham mau sac/nen;
- overfit;
- chay cham.

Nen can mot cach rut gon anh thanh vector co y nghia hon.

## CLIP la gi

Giai thich doi thuong:

> CLIP la mot bo phien dich anh sang day so. No da duoc hoc truoc tu nhieu cap anh-va-mo-ta, nen vector cua anh co chua thong tin ngu nghia.

Trong du an:

- CLIP dung de trich dac trung anh.
- CLIP khong duoc fine-tune.
- Classifier phia sau moi duoc huan luyen tren CrisisMMD.

## ViT-B/32 nghia la gi

ViT = Vision Transformer.

B/32:

- B la ban Base;
- 32 la kich thuoc patch 32x32.

Anh duoc resize ve 224x224.

So patch:

```text
224 / 32 = 7
7 x 7 = 49 patch
```

Nghia la anh duoc chia thanh 49 o vuong.

## Patch va token anh

Giai thich:

- Patch la mot manh nho cua anh.
- Moi patch duoc bien thanh mot token.
- Transformer xem quan he giua cac token.
- Cuoi cung CLIP tra ra vector anh 512 chieu.

Ve bang chu:

```text
Anh 224x224
    ↓ chia 7x7 patch
49 patch
    ↓ bien thanh token
49 token anh
    ↓ Transformer
vector 512 chieu
```

## Contrastive learning cua CLIP

Khong can day qua sau, chi can noi:

> CLIP hoc bang cach keo anh gan dung mo ta cua no, va day xa cac mo ta sai.

Vi du:

- Anh duong ngap nen gan caption "flooded road".
- Anh duong ngap khong nen gan caption "food donation boxes".

Nho vay embedding CLIP co tinh ngu nghia.

## Embedding la gi

Noi:

> Embedding la toa do cua mau trong khong gian so.

Anh duong ngap co the gan anh duong ngap khac.

Anh thung quyen gop co the gan anh cuu tro/quyen gop.

Nhung embedding khong phai:

- nhan;
- xac suat;
- quyet dinh.

Embedding chi la dau vao cho classifier.

## Frozen feature extractor

Frozen nghia la dong bang.

Trong du an:

- CLIP giu nguyen trong so.
- Chi forward pass de lay vector.
- Khong tinh gradient.
- Khong cap nhat CLIP.

Noi bang doi thuong:

> Muon dung CLIP nhu may scan anh co san, khong day lai no.

## Fine-tuning la gi

Fine-tuning la cap nhat mot phan hoac toan bo trong so model tien huan luyen bang du lieu cua minh.

Du an khong fine-tune CLIP.

Vi sao khong fine-tune:

1. Muc tieu hoc phan la classifier co dien va DSS.
2. Train chi 13.608 mau.
3. Lop hiem rat it.
4. Fine-tuning de overfit.
5. Tinh embedding mot lan de so sanh cong bang sau classifier.
6. Chi phi thap hon, tai lap hon.

## Quy trinh trich anh that

Moi anh:

1. Mo bang Pillow.
2. Chuyen RGB.
3. Resize/xu ly theo processor chinh thuc cua CLIP.
4. Chay model o `eval` va `torch.no_grad()`.
5. Lay projected image embedding 512 chieu.
6. Chuan hoa L2.
7. Luu cache theo dung thu tu `(tweet_id, image_id)`.

## Chuan hoa L2 la gi

Noi don gian:

> Chuan hoa L2 dua vector ve do dai 1. Khi so sanh, ta chu y huong cua vector hon do lon tuy y.

Cong thuc:

```text
x_hat = x / ||x||2
```

Sau chuan hoa:

- norm trung binh bang 1;
- khoang cach Euclidean va cosine co quan he tot hon;
- k-NN tren CLIP on dinh hon.

## Cache la gi

Cache la luu ket qua tinh san de khoi tinh lai moi lan.

Trong du an:

- train cache shape: (13.608, 512)
- dev cache shape: (2.237, 512)
- test cache shape: (2.237, 512)

Vi sao can cache:

- CLIP chay cham hon classifier co dien.
- Embedding khong doi neu anh va thu tu khong doi.
- So sanh 6 classifier de cong bang hon.
- Dashboard text-only khong can tai CLIP ngay.

## Audit embedding

Bao cao noi:

- norm trung binh bang 1;
- khong co vector zero;
- metadata khop thu tu annotation;
- do lech chuan toan ma tran khoang 0,04418.

Giai thich:

> Metadata khop thu tu rat quan trong. Neu dong thu 100 cua embedding khong ung voi dong thu 100 cua label, model se hoc sai.

## Vi sao khong noi "CLIP phan lop"

Noi dung:

> CLIP chi tao embedding. Classifier phia sau moi phan lop CrisisMMD.

Sai:

> CLIP du bao humanitarian.

Dung:

> Logistic Regression/k-NN tren embedding CLIP du bao humanitarian.

## Vi du noi mieng

Anh A: duong ngap.

CLIP bien thanh:

```text
[0.12, -0.03, 0.44, ..., 0.07]  # 512 so
```

Classifier doc vector nay va hoc:

```text
vector gan kieu nay thuong la infrastructure/affected
```

## Gioi han

CLIP:

- hoc tu du lieu lon chung, khong rieng anh tham hoa;
- co the khong nhan dung mot so hien truong dac thu;
- khong hieu vi tri;
- khong xac thuc anh cu hay moi;
- khong thay the con nguoi.

## Bai tap

Hoi nguoi hoc:

Neu hai anh la:

1. Cung mot anh nhung resize nhe.
2. Hai anh deu ngap duong nhung khac dia diem.
3. Anh quyen gop va anh duong sap.

Theo CLIP, cap nao co kha nang gan nhau nhat?

Dap an:

- 1 rat gan neu noi dung y chang.
- 2 co the gan vi cung ngu nghia ngap duong.
- 3 thuong xa hon.

## Loi hieu sai can chan

1. "Anh dua vao model bang pixel thang."
   - Sai. Du an dung embedding CLIP 512 chieu.

2. "CLIP duoc train lai tren CrisisMMD."
   - Sai. CLIP frozen.

3. "Embedding la nhan."
   - Sai. Embedding la vector dac trung.

4. "Anh luon dung hon text."
   - Sai. Anh la mot nguon bang chung doc lap, co luc huu ich, co luc mau thuan.

## Cau hoi kiem tra

1. Vi sao can resize anh ve 224x224?
2. ViT-B/32 chia anh thanh bao nhieu patch?
3. Embedding CLIP co bao nhieu chieu?
4. Frozen khac fine-tuning nhu the nao?
5. Vi sao can cache embedding?
6. Ai la thanh phan phan lop: CLIP hay classifier phia sau?

