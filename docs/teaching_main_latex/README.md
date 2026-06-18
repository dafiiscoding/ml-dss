# Giao an day du an DSS tham hoa da phuong thuc

Thu muc nay duoc soan rieng cho ban bao cao LaTeX chinh:

- PDF: `reports/latex/main.pdf`
- LaTeX: `reports/latex/main.tex`
- So trang PDF theo `pdfinfo`: 79 trang

Muc tieu cua giao an: day cho mot nguoi mat goc van co the hieu du an tu dau den cuoi, theo kieu giai thich bang doi thuong truoc, cong thuc va metric sau.

## Cach dung nhanh

Neu can day trong 1-2 ngay, doc theo thu tu:

1. `01_boi_canh_va_ranh_gioi.md`
2. `02_bai_toan_quyet_dinh_va_nguoi_dung.md`
3. `03_du_lieu_crisismmd_nhan_va_split.md`
4. `04_ro_ri_du_lieu_trung_lap_va_mask.md`
5. `05_lam_sach_tweet_va_tfidf.md`
6. `06_anh_clip_embedding_va_cache.md`
7. `09_hoc_co_giam_sat_va_sau_thuat_toan.md`
8. `10_metric_baseline_va_mat_can_bang.md`
9. `12_late_fusion_conflict_manual_review.md`
10. `13_risk_priority_routing_policy.md`
11. `14_ket_qua_robustness_va_case_study.md`
12. `16_tong_on_va_kich_ban_bao_ve.md`

Neu day ky, di het 16 buoi.

## Danh sach file

| File | Noi dung |
|---|---|
| `00_cach_su_dung_giao_an.md` | Cach day nguoi mat goc, nguyen tac dung tu, thu tu giai thich |
| `01_boi_canh_va_ranh_gioi.md` | Tham hoa, tweet, vi sao can DSS, ranh gioi he thong, khong thay hotline |
| `02_bai_toan_quyet_dinh_va_nguoi_dung.md` | Nam quyet dinh, nguoi dung, tinh huong thuc te |
| `03_du_lieu_crisismmd_nhan_va_split.md` | CrisisMMD, 18.082 cap tweet-anh, label, label_top, train/dev/test |
| `04_ro_ri_du_lieu_trung_lap_va_mask.md` | Ro ri du lieu, SHA-256, pHash, canonical mask, robustness mask |
| `05_lam_sach_tweet_va_tfidf.md` | Lam sach tweet, audit text, TF-IDF, vi du tinh tay |
| `06_anh_clip_embedding_va_cache.md` | Pixel, patch, CLIP, embedding 512 chieu, frozen, cache |
| `07_eda_phan_bo_event_text_conflict.md` | EDA train-only, phan bo nhan, event shift, text length, bat dong text-image |
| `08_kmeans_apriori_pca_tsne.md` | K-Means, silhouette, Apriori, PCA, t-SNE |
| `09_hoc_co_giam_sat_va_sau_thuat_toan.md` | Supervised learning va 6 classifier |
| `10_metric_baseline_va_mat_can_bang.md` | Accuracy, Precision, Recall, F1/F2, Macro-F1, MCC, baseline |
| `11_thiet_ke_thuc_nghiem_tuning.md` | Quy trinh train/dev/test, tuning, model duoc chon |
| `12_late_fusion_conflict_manual_review.md` | Calibration, late fusion, conflict score, manual review |
| `13_risk_priority_routing_policy.md` | Risk Score, Priority, Routing, policy vs model |
| `14_ket_qua_robustness_va_case_study.md` | Ket qua cuoi, robust test, bootstrap, ca minh hoa |
| `15_dashboard_streamlit_va_trien_khai.md` | Dashboard 5 trang, CSV vs database, kiem thu |
| `16_tong_on_va_kich_ban_bao_ve.md` | Tong on, cau hoi bao ve, cach noi dung va sai |
| `PHU_LUC_A_thuat_ngu.md` | Bang thuat ngu cuc ngan gon |
| `PHU_LUC_B_cau_hoi_va_bai_tap.md` | Cau hoi kiem tra, bai tap, de cuong van dap |

## Nguyen tac khong duoc noi sai

- Khong noi he thong tu dong cuu ho.
- Khong noi du an thu thap tweet truc tiep tu Twitter/X.
- Khong noi Risk Score/Priority la toi uu thong ke.
- Khong khoe F2 informative mot minh.
- Khong so sanh truc tiep metric cua nhan text rieng voi nhan image rieng.
- Khong noi CLIP duoc fine-tune.
- Khong noi K-Means/t-SNE chung minh phan lop tot.
- Khong noi lop hiem nhu missing/found du tin cay de tu dong dieu dong.

## Mot cau tom tat dung

Du an doc du lieu CrisisMMD da luu san, bien tweet va anh thanh vector, dung cac classifier co dien de du bao informative va humanitarian, tron du bao text-anh bang Late Fusion, sau do tao Risk Score, Priority, Routing va Manual Review de con nguoi xu ly theo thu tu co can cu.

