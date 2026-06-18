# Buoi 9 - Hoc co giam sat va sau thuat toan phan lop

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> Hoc co giam sat la hoc tu cap input da co nhan. Du an co 4 bai toan con: TF-IDF va CLIP embedding, moi loai du bao informative va humanitarian. Sau thuat toan duoc so sanh cong bang tren cung dac trung.

## Tu khoa

- Supervised learning
- Classifier
- Logistic Regression
- Decision Tree
- Naive Bayes
- k-NN
- SVM
- Random Forest
- XGBoost
- Deep learning
- Overfit

## Hoc co giam sat la gi

Noi bang doi thuong:

> Hoc co giam sat giong nhu cho hoc sinh xem bai co dap an. Moi mau co du lieu dau vao va nhan dung. Model hoc quan he giua dau vao va nhan.

Trong du an:

```text
x = vector cua tweet hoac anh
y = nhan informative hoac humanitarian
```

Model hoc ham:

```text
f(x) -> y
```

## Bon bai toan con

Du an khong co mot model duy nhat. Co 4 bai toan:

1. TF-IDF -> informative.
2. TF-IDF -> humanitarian category.
3. CLIP image embedding -> informative.
4. CLIP image embedding -> humanitarian category.

Y nghia:

- Text co model rieng.
- Anh co model rieng.
- Task informative co model rieng.
- Task humanitarian co model rieng.

## Vi sao tach nhu vay

Tach de tra loi hai cau hoi:

1. Bieu dien nao co tin hieu huu ich?
   - TF-IDF hay CLIP?

2. Thuat toan nao khai thac bieu dien do tot?
   - SVM, LR, k-NN, v.v.

Can noi:

> CLIP chi tao dac trung anh. Classifier phia sau moi hoc tu CrisisMMD.

## Logistic Regression

Ten co chu "Regression" nhung dung de phan lop.

Noi don gian:

> Logistic Regression hoc mot duong ranh gioi tuyen tinh va cho xac suat mot mau thuoc lop nao.

Manh:

- nhanh;
- tot voi TF-IDF va embedding;
- co xac suat;
- de kiem soat bang regularization.

Yeu:

- ranh gioi tuyen tinh;
- khong tu tim tuong tac phuc tap.

Trong du an:

- Text humanitarian chon Logistic Regression C=1.
- Image humanitarian chon Logistic Regression C=1.

## Decision Tree

Noi don gian:

> Decision Tree giong nhu cay hoi dap co/khong. Moi nut hoi mot cau, di den la de ra nhan.

Vi du:

```text
co tu "rescue" khong?
  co -> co anh nguoi khong?
  khong -> co tu "donation" khong?
```

Manh:

- de giai thich;
- bat quan he phi tuyen;
- truc quan.

Yeu:

- de overfit;
- tren TF-IDF 2.000 chieu co nhieu tu hiem, cay de hoc chi tiet ngau nhien.

Trong du an:

- Decision Tree dung max_depth=20 o vong so sanh.

## Naive Bayes

Noi don gian:

> Naive Bayes doan lop dua tren xac suat cua cac tu/dac trung, voi gia dinh cac dac trung doc lap khi biet lop.

Tu "naive" nghia la gia dinh hoi don gian, khong phai thuat toan ngu.

Manh:

- nhanh;
- hay tot voi text;
- on voi du lieu nhieu chieu.

Yeu:

- gia dinh doc lap thuong khong dung;
- xac suat co the qua tu tin.

Trong du an:

- Multinomial NB cho TF-IDF khong am.
- Gaussian NB cho embedding dac.

## k-Nearest Neighbors

Noi don gian:

> Mau moi hoi k hang xom gan nhat: "cac ban gan toi nhat thuoc lop nao?"

Neu k=5, 3 hang xom la rescue, 2 hang xom la infrastructure, model doan rescue.

Distance weighting:

- hang xom gan hon co phieu nang hon.

Manh:

- truc quan;
- tot neu khong gian embedding co lang gieng co nghia.

Yeu:

- du bao cham;
- phai luu train;
- tren TF-IDF thua nhieu chieu co the kem.

Trong du an:

- Image informative chon k-NN k=41, distance weighting.

## SVM

SVM = Support Vector Machine.

Noi don gian:

> SVM tim duong ranh gioi co khoang cach an toan lon nhat giua cac lop.

Manh:

- rat manh voi TF-IDF thua nhieu chieu;
- bien quyet dinh tuyen tinh ro;
- tham so C dieu khien bam train hay mem hon.

Yeu:

- LinearSVC ban dau khong xuat xac suat;
- can calibration de dung cho Late Fusion.

Trong du an:

- Text informative chon calibrated Linear SVM C=3.

## Random Forest

Noi don gian:

> Random Forest la nhieu Decision Tree cung bo phieu.

Moi cay hoi hoi khac nhau, hoc tren mau bootstrap va tap con dac trung.

Manh:

- giam overfit so voi mot cay;
- bat phi tuyen;
- on dinh hon Decision Tree.

Yeu:

- lon;
- kho giai thich hon mot cay;
- khong chac tot hon SVM tren TF-IDF tuyen tinh.

Trong du an:

- Random Forest dung 200 cay, balanced_subsample.

## Vi sao khong XGBoost hay mang sau

Ban main co muc rieng.

Ly do:

1. Muc tieu hoc phan la so sanh cac ho thuat toan trong chuong trinh.
2. SVM/LR da la baseline manh tren TF-IDF.
3. Lop hiem qua it, model phuc tap de overfit.
4. Them model moi phai co protocol tuning cong bang.
5. Muc tieu khong phai san Accuracy cao bang moi gia, ma la DSS trung thuc, tai lap.

Cau noi dung:

> Khong dung XGBoost khong phai vi XGBoost kem, ma vi no khong can thiet de tra loi cau hoi hien tai va co the lam so sanh thieu cong bang.

## Overfit la gi

Noi:

> Overfit la hoc thuoc bai tap train den muc gap de moi thi sai.

Vi du:

Model thay tu `harvey` qua nhieu trong train va gan no voi mot lop, nhung event moi khong co `harvey` thi hong.

## Bai tap

Hoi nguoi hoc noi thuat toan phu hop:

1. Muon model text manh tren TF-IDF thua: SVM.
2. Muon model don gian co xac suat: Logistic Regression.
3. Muon xem hang xom trong embedding anh: k-NN.
4. Muon cay hoi dap de giai thich: Decision Tree.
5. Muon nhieu cay bo phieu: Random Forest.
6. Muon text baseline nhanh: Naive Bayes.

## Loi hieu sai can chan

1. "Logistic Regression la hoi quy, khong phan lop."
   - Sai. Trong ngu canh nay la classifier.

2. "CLIP la model chinh cua du an."
   - Sai. CLIP la bo trich dac trung bo tro.

3. "Model phuc tap hon luon tot hon."
   - Sai. Du lieu lop hiem it, phuc tap de overfit.

4. "Random Forest luon hon Decision Tree nen chac thang."
   - Sai. Phai xem metric tren dev.

## Cau hoi kiem tra

1. Bon bai toan con cua du an la gi?
2. SVM manh o loai dac trung nao?
3. k-NN phu hop voi embedding anh vi sao?
4. Overfit la gi?
5. Vi sao khong dung XGBoost trong ban hien tai?
6. CLIP co phai classifier khong?

