# Buoi 11 - Thiet ke thuc nghiem, so sanh model va tuning

## Muc tieu

Sau buoi nay, nguoi hoc phai noi duoc:

> Du an fit tren train, chon model va tham so tren dev, khoa tat ca roi moi bao cao test. Tuning la do siêu tham so trong luoi nho, khong phai cap nhat CLIP hay nhin test.

## Tu khoa

- Experiment design
- Model parameter
- Hyperparameter
- Grid search
- Tuning
- Dev selection
- Locked model
- Canonical test
- Robust test

## Tham so model va sieu tham so

### Model parameter

La cai model hoc tu train.

Vi du:

- trong so `w` cua Logistic Regression;
- he so cua SVM;
- cau truc cay sau khi fit.

### Hyperparameter

La cai con nguoi/dat truoc khi model hoc.

Vi du:

- C cua SVM/LR;
- k cua k-NN;
- max_depth cua Decision Tree;
- so cay cua Random Forest.

Noi:

> Parameter la diem hoc sinh tu lam trong bai. Hyperparameter la cach giao vien quy dinh bai tap truoc khi lam.

## Quy tac train/dev/test

| Split | So hang sach | Dung de lam gi |
|---|---:|---|
| Train | 13.608 | Fit TF-IDF, classifier, calibration noi bo |
| Dev | 2.189 | Chon model, hyperparameter, fusion weight, threshold |
| Test | 2.169 | Danh gia cuoi sau khi khoa moi lua chon |

Can nhac lai:

> Dung test de chon hyperparameter la bien test thanh dev.

## Ba vong thuc nghiem

### Vong 1 - So sanh sau thuat toan

Chay 6 ho model tren 4 bai toan con.

Muc dich:

- biet ho nao hop voi TF-IDF;
- biet ho nao hop voi CLIP embedding;
- khong nhay vao tuning qua som.

### Vong 2 - Tuning ho thang

Chi tune ho thang cua moi bai toan.

Vi sao?

- Giam so phep thu.
- Giam nguy co overfit dev.
- Cong bang va de giai thich.

### Vong 3 - Fusion va threshold

Sau khi co model text/image:

- can xac suat;
- tune fusion weight tren dev;
- tune threshold tren dev;
- tune conflict threshold tren dev voi rang buoc capacity.

## Ket qua vong 1

### Text informative

Chon theo F2.

Thang: Linear SVM.

Dev:

- Linear SVM F2 = 0,8511.
- Naive Bayes F2 = 0,8406.
- Random Forest F2 = 0,8135.

### Text humanitarian

Chon theo Macro-F1.

Thang: Logistic Regression.

Dev:

- Logistic Regression Macro-F1 = 0,3297.
- Linear SVM Macro-F1 = 0,2787.
- Random Forest Macro-F1 = 0,2626.

### Image informative

Chon theo F2.

Thang: k-NN.

Dev:

- k-NN F2 = 0,8146.
- Linear SVM F2 = 0,8045.
- Random Forest F2 = 0,7996.

### Image humanitarian

Chon theo Macro-F1.

Thang: Logistic Regression.

Dev:

- Logistic Regression Macro-F1 = 0,3637.
- Linear SVM Macro-F1 = 0,3631.
- k-NN Macro-F1 = 0,3563.

Can noi:

> Accuracy cao nhat khong nhat thiet la model duoc chon neu tieu chi la Macro-F1.

## Tuning la gi

Tuning = thu mot so gia tri hyperparameter da dinh truoc tren dev.

Khong phai:

- train lai CLIP;
- doi tung tham so sau khi nhin test;
- thu vo han den khi dep.

## Luoi tuning trong du an

| Task | Model | Luoi | Ket qua |
|---|---|---|---|
| Text informative | Linear SVM | C in {0,05; 0,1; 0,3; 1; 3} | C=3, F2=0,8548 |
| Text humanitarian | Logistic Regression | C in {0,05; 0,1; 0,3; 1; 3; 10} | C=1, Macro-F1=0,3297 |
| Image informative | k-NN | k in {5,9,15,25,41}, uniform/distance | k=41 distance, F2=0,8216 |
| Image humanitarian | Logistic Regression | C in {0,05; 0,1; 0,3; 1; 3; 10} | C=1, Macro-F1=0,3637 |

## Y nghia tham so

### C cua SVM/LR

C cao:

- phat loi train manh hon;
- bam train sat hon;
- co nguy co overfit hon.

C thap:

- regularization manh hon;
- bien mem hon;
- co the underfit.

### k cua k-NN

k cao:

- du bao muot hon;
- it nhiu hon;
- co the bo mat chi tiet cuc bo.

k thap:

- nhay voi lang gieng gan;
- de bi nhiu.

### Distance weighting

Lang gieng gan co trong so lon hon lang gieng xa.

## Truoc va sau tuning

| Task | Metric | Truoc | Sau | Chenh |
|---|---:|---:|---:|---:|
| Text informative | F2 | 0,8511 | 0,8548 | +0,0037 |
| Image informative | F2 | 0,8146 | 0,8216 | +0,0070 |
| Text humanitarian | Macro-F1 | 0,3297 | 0,3297 | 0 |
| Image humanitarian | Macro-F1 | 0,3637 | 0,3637 | 0 |

Giai thich:

> Tuning khong bat buoc phai lam diem tang. Co luc tuning chi xac nhan cau hinh mac dinh da tot.

## Quy tac hoa khi Macro-F1 gan nhau

Ban main noi:

- Neu Macro-F1 chenh khong qua 0,001 coi la hoa.
- Khi hoa, uu tien Weighted-F1 roi Accuracy.

Ly do:

> Lop hiem it mau co the lam Macro-F1 dao dong vi vai du bao. Quy tac hoa giup tranh chon ngau nhien.

## Khoa model

Sau dev:

- khoa model;
- khoa hyperparameter;
- khoa fusion weight;
- khoa threshold;
- khoa conflict threshold.

Sau do moi test.

Noi:

> Test chi duoc mo sau khi da ky ten vao cach lam.

## Bai tap

Hoi:

Neu chay test thay threshold 0,40 tot hon 0,38, co duoc doi khong?

Dap an:

Khong, vi threshold da chon tren dev va khoa truoc test.

Hoi:

Neu Logistic Regression humanitarian sau tuning khong tang diem, tuning co vo ich khong?

Dap an:

Khong. No xac nhan C=1 la lua chon on trong luoi.

## Loi hieu sai can chan

1. "Tuning la huan luyen CLIP."
   - Sai. CLIP frozen, tuning chi hyperparameter classifier.

2. "Test dung de chon cau hinh tot nhat."
   - Sai. Test chi de bao cao cuoi.

3. "Sau tuning khong tang nghia la sai."
   - Sai. Co the lưoi da xac nhan mac dinh tot.

4. "Thu cang nhieu gia tri cang tot."
   - Khong chac. Thu qua nhieu co the overfit dev.

## Cau hoi kiem tra

1. Parameter khac hyperparameter the nao?
2. Dev dung de lam gi?
3. Test dung de lam gi?
4. Bon model cuoi duoc chon la gi?
5. Tuning co cap nhat CLIP khong?
6. Vi sao can khoa model truoc test?

