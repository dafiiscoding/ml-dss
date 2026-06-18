# Phu luc A - Bang thuat ngu sieu ngan gon

## Du lieu

**Tweet**: Bai dang ngan tren Twitter/X.

**CrisisMMD**: Bo du lieu da luu san gom tweet, anh va nhan cua 7 tham hoa nam 2017.

**Annotation**: Bang gan nhan do con nguoi tao.

**`tweet_id`, `image_id`**: Dinh danh tweet va anh. Khoa dung cua mau la `(tweet_id, image_id)`.

**`label`**: Nhan chinh thuc cho task informative.

**`label_top`**: Nhan chinh thuc cho task humanitarian 8 lop.

**`label_text_*`**: Nhan rieng dua tren text, chi dung de chan doan.

**`label_image_*`**: Nhan rieng dua tren anh, chi dung de chan doan.

## Split

**Train**: Tap de hoc model va fit feature.

**Dev**: Tap de chon model, hyperparameter, threshold.

**Test**: Tap de bao cao cuoi sau khi khoa lua chon.

## Text

**Token**: Don vi tu/cum tu sau khi tach.

**Stopword**: Tu qua pho bien, it thong tin.

**Unigram**: Mot tu don.

**Bigram**: Hai tu lien tiep.

**TF**: So lan tu xuat hien trong van ban.

**IDF**: Do hiem cua tu tren tap train.

**TF-IDF**: Trong so tu bang TF nhan IDF.

**Vector thua**: Vector co hau het phan tu bang 0.

**OOV**: Tu ngoai tu vung da hoc tu train.

## Anh

**Pixel**: Diem anh.

**RGB**: Ba kenh mau red, green, blue.

**Patch**: Manh nho cua anh.

**Token anh**: Vector bieu dien patch.

**CLIP**: Model tien huan luyen anh-van ban, dung de trich dac trung.

**ViT-B/32**: Vision Transformer Base, patch 32x32.

**Embedding**: Vector so bieu dien noi dung.

**Frozen**: Giu nguyen trong so model, khong fine-tune.

**Fine-tuning**: Cap nhat model tien huan luyen bang du lieu moi.

**L2 normalization**: Dua vector ve do dai 1.

## Ro ri va trung lap

**Data leakage**: Ro ri du lieu, khi thong tin dev/test di vao qua trinh hoc/chon model.

**SHA-256**: Van tay file, bat anh giong het byte.

**pHash**: Van tay thi giac, bat anh gan giong.

**Hamming distance**: So bit khac nhau giua hai hash.

**Canonical mask**: Loai exact duplicate anh/text tren dev/test.

**Robustness mask**: Loai them near-duplicate da review de kiem tra do ben.

## EDA

**EDA**: Phan tich kham pha du lieu.

**Class imbalance**: Mat can bang lop.

**Event shift**: Phan bo thay doi theo tham hoa.

**K-Means**: Gom cum khong giam sat.

**Silhouette**: Diem do cum tach ro hay chong lan.

**Apriori**: Tim item/hashtag hay xuat hien cung nhau.

**Support**: Ty le transaction co ca tien de va ket qua.

**Confidence**: Xac suat co ket qua khi da co tien de.

**Lift**: Muc dong xuat hien manh hon doc lap.

**PCA**: Giam chieu tuyen tinh giu phuong sai lon.

**t-SNE**: Giam chieu phi tuyen de xem lang gieng cuc bo.

## Model va metric

**Classifier**: Mo hinh phan lop.

**Hyperparameter**: Sieu tham so do nguoi dat truoc.

**Tuning**: Thu cac gia tri hyperparameter tren dev.

**Calibration**: Hieu chinh score thanh xac suat.

**Baseline**: Moc doi chung don gian.

**Accuracy**: Ty le dung tong.

**Precision**: Trong cac ca model bao positive, bao nhieu ca dung.

**Recall**: Trong cac ca positive that, bat duoc bao nhieu.

**F1**: Trung binh dieu hoa Precision va Recall.

**F2**: Giong F1 nhung uu tien Recall hon.

**Balanced Accuracy**: Trung binh Recall cua cac lop.

**MCC**: Tuong quan du bao nhi phan, du bao hang co MCC = 0.

**Macro-F1**: F1 trung binh deu tren cac lop.

**Weighted-F1**: F1 trung binh theo so mau moi lop.

## Fusion va DSS

**Late Fusion**: Tron xac suat sau khi text va anh du bao rieng.

**Threshold**: Nguong bien xac suat thanh nhan.

**Conflict score**: Muc bat dong giua text va anh.

**Manual Review**: Co che dua ca kho cho con nguoi xac minh.

**Risk Score**: Diem rui ro theo chinh sach trong so.

**Priority**: Low/Medium/High, thu tu uu tien trong hang doi.

**Routing**: Goi y doi tiep nhan.

**Policy**: Quy tac do con nguoi thiet ke, khong phai model hoc tu nhan.

**Human-in-the-loop**: Con nguoi nam trong cac diem quyet dinh rui ro.

**DSS**: Decision Support System, he ho tro quyet dinh.

