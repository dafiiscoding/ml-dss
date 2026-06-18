# Buoi 8 - K-Means, Apriori, PCA va t-SNE: kham pha, khong thay classifier

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> K-Means, Apriori, PCA va t-SNE trong du an dung de kham pha va giai thich du lieu, khong dung de chung minh model phan lop tot hay thay the classifier.

## Tu khoa

- Clustering
- K-Means
- Centroid
- Silhouette
- Apriori
- Transaction
- Support
- Confidence
- Lift
- PCA
- t-SNE
- Giam chieu

## K-Means la gi

Noi doi thuong:

> K-Means gom cac diem giong nhau thanh K nhom, ma khong can nhin nhan that.

K-Means la hoc khong giam sat.

Khac voi classifier:

- Classifier hoc tu nhan.
- K-Means khong dung nhan de gom cum.

## Cach K-Means chay

Lap lai 2 buoc:

1. Gan moi diem vao centroid gan nhat.
2. Dich centroid ve trung binh cua cac diem trong cum.

Centroid = tam cum.

Vi du doi thuong:

Co nhieu hoc sinh tren san truong. Dat 3 la co. Moi hoc sinh chay ve la co gan nhat. Sau do dich la co ve giua nhom hoc sinh. Lap lai den khi on dinh.

## K-Means trong du an

Du an dung:

- TF-IDF toi da 2.000 unigram/bigram;
- min_df = 3;
- n_init = 10;
- random seed 42;
- sweep K = 2 den 12;
- K = 8 de doi chieu taxonomy 8 lop.

Can noi ro:

> K = 8 khong phai vi K-Means chung minh co 8 cum tot. K = 8 giu de doi chieu voi 8 nhan humanitarian.

## Silhouette la gi

Silhouette do diem nam gon trong cum hay nam lung chung giua cac cum.

Truc giac:

- Gan 1: cum dep, diem gan ban cung cum va xa cum khac.
- Gan 0: cac cum chong lan.
- Am: co the gan sai cum.

Ket qua du an:

- Silhouette chi khoang 0,014 den 0,024.
- K = 8 co silhouette 0,021.
- Khong co diem gay ro.

Ket luan:

> K-Means co ich de xem chu de, nhung khong du on dinh de thay classifier hoac routing.

## Apriori la gi

Noi doi thuong:

> Apriori tim cac tu/hashtag hay xuat hien cung nhau.

Moi tweet duoc bien thanh transaction, tuc mot gio do gom item.

Vi du:

```text
"#iran #earthquake rescue teams searching"
```

Thanh gio:

```text
{#iran, #earthquake, rescue}
```

## One-hot transaction

Bang:

| Transaction | #iran | #earthquake | rescue | flood |
|---|---:|---:|---:|---:|
| tweet A | 1 | 1 | 1 | 0 |

1 nghia la co item do.

## Support, Confidence, Lift

### Support

`support(A => B)` = ty le gio co ca A va B.

Noi:

> Luat nay pho bien den muc nao?

### Confidence

`confidence(A => B)` = trong cac gio co A, bao nhieu gio cung co B.

Noi:

> Neu thay A, xac suat thay B la bao nhieu?

### Lift

`lift(A => B)` = A va B di cung nhau manh hon doc lap bao nhieu lan.

Noi:

> Hai item co di chung bat thuong hon ngau nhien khong?

Lift = 1: gan nhu doc lap.

Lift > 1: co lien ket duong.

## Luat trong du an

Vi du:

- `#iran => #earthquake`
- support 0,018;
- confidence 0,717;
- lift 18,5.

Dien giai:

Trong cac transaction da giu, 1,8% co ca #iran va #earthquake. Neu co #iran, 71,7% cung co #earthquake. Cap nay di chung cao gap 18,5 lan so voi neu doc lap.

## Gioi han cua Apriori

Lift cao khong co nghia la nhan qua.

Sai:

> #iran gay ra #earthquake.

Dung:

> Hai hashtag cung xuat hien trong cung ngu canh su kien.

Apriori dung de:

- filter;
- drill-down;
- hieu context event.

Khong dung de:

- du bao Priority;
- thay classifier;
- chung minh quan he nhan qua.

## PCA la gi

PCA = Principal Component Analysis.

Noi don gian:

> PCA tim cac truc moi giu nhieu do bien thien nhat cua du lieu.

CLIP embedding co 512 chieu. Nguoi khong nhin duoc 512 chieu, nen phai giam chieu.

Ket qua ban main:

- PC1 giu 7,5% phuong sai.
- PC2 giu 4,8%.
- Hai truc dau chi giu 12,4%.
- Can 32 thanh phan moi dat 50%.

Y nghia:

> Hinh 2D chi la mot lat rat mong cua du lieu 512 chieu.

## t-SNE la gi

t-SNE giam chieu phi tuyen, co gang giu cac lang gieng cuc bo.

Dung de:

- nhin nhom cuc bo;
- hinh dung embedding.

Khong dung de:

- cham diem model;
- chung minh cac lop tach tot;
- so sanh khoang cach toan cuc.

Ban main:

- t-SNE dung 3.000 embedding train;
- perplexity 30;
- khoi tao PCA;
- seed 42.

Ket luan:

> Mot so vung cuc bo xuat hien, nhung cac lop chong lan manh. Gia tri CLIP phai xac nhan bang dev/test o chuong 5.

## K-Means/PCA/t-SNE khac classifier the nao

| Cong cu | Co dung nhan khi hoc? | Muc dich |
|---|---|---|
| K-Means | Khong | Kham pha cum chu de |
| Apriori | Khong | Tim item hay di cung |
| PCA/t-SNE | Khong | Ve hinh giam chieu |
| Classifier | Co | Du bao nhan cho mau moi |

## Bai tap

Hoi:

Neu t-SNE ve thay 2 cum dep, co duoc ket luan model se phan lop tot khong?

Dap an:

Khong. t-SNE la hinh 2D, co the meo khoang cach va phu thuoc tham so. Phai xem dev/test metric.

Hoi:

Neu Apriori thay `#puertorico => #hurricanemaria` lift cao, co dung de xep Priority khong?

Dap an:

Khong. No chi la lien ket hashtag/event, khong noi muc khan cap.

## Loi hieu sai can chan

1. "K-Means ra 8 cum nen trung 8 lop."
   - Sai. K = 8 chi de doi chieu, silhouette rat thap.

2. "Lift cao la nhan qua."
   - Sai. Lift chi la dong xuat hien.

3. "PCA 2D chong lop nen CLIP vo dung."
   - Sai. 2D chi giu 12,4% phuong sai.

4. "t-SNE tach lop la model tot."
   - Sai. t-SNE khong cham diem model.

## Cau hoi kiem tra

1. K-Means co dung nhan humanitarian khong?
2. Silhouette gan 0 nghia la gi?
3. Support khac confidence the nao?
4. Lift > 1 nghia la gi?
5. PCA hai truc dau giu bao nhieu phuong sai?
6. Vi sao t-SNE khong chung minh phan lop tot?

