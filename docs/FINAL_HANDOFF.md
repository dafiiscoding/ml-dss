# FINAL HANDOFF

## Trạng thái nộp bài

- GĐ0-GĐ3: hoàn thành và có artefact chạy thật.
- GĐ4-GĐ5: hoàn thành ở mức prototype học thuật đã kiểm thử.
- GĐ6-GĐ7: hoàn thành kỹ thuật; còn dữ liệu danh tính/phân công.
- GĐ8: hoàn thành baseline, leakage audit, robust mask, bootstrap,
  event/class stability và tích hợp báo cáo/dashboard/notebook.

## Nghiệm thu cuối

- 37 unit tests pass.
- Hai notebook execute, 0 error.
- 6/6 Streamlit entrypoint AppTest, 0 exception.
- DSS audit và integration với ảnh thật pass.
- `reports/latex/main.pdf`: 34 trang, XeLaTeX hai lượt, không có
  overfull hoặc undefined warning.
- Canonical model/config không bị retune bởi robust analysis.

## Chỉ còn cần nhóm cung cấp

1. Họ tên và MSSV của ba thành viên còn lại.
2. Phần việc thực tế của cả bốn thành viên.
3. Tỷ lệ đóng góp nếu biểu mẫu môn học yêu cầu.

Chỉ sửa một file:

- `docs/team_info.json`

Sau đó chạy:

```powershell
python -m scripts.finalize_submission
```

Lệnh này đồng bộ bìa, phụ lục, contribution log, biên dịch PDF hai lượt và
chạy preflight GitHub ở chế độ final.

Sau đó commit và push:

```powershell
git add docs/team_info.json docs/contribution_log.md `
  reports/latex/chapters/team_info.tex reports/latex/main.pdf
git commit -m "Finalize team information"
git push
```

## Ngoài phạm vi bản nộp hiện tại

- Leave-one-event-out retraining để đo tổng quát hóa sang thảm họa chưa thấy.
- Thu thập ground truth Priority để học và đánh giá Risk Score.
- Kiểm thử tải, phân quyền và API cho triển khai production.

Đây là hướng phát triển, không phải thiếu sót bị che giấu trong kết quả hiện tại.
