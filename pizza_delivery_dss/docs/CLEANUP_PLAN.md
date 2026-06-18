# Plan dọn sạch project Pizza DSS

Mục tiêu là giữ project đủ tái lập và đủ nộp bài, nhưng bỏ các file môi trường
hoặc file build trung gian không cần thiết. Không áp dụng plan này cho project
CrisisMMD ở thư mục cha.

## Đánh giá hiện tại

Project đã ổn về mặt học thuật:

- `scripts.run_all` pass 18/18.
- Unit tests pass 30/30.
- Có đủ code, dữ liệu, notebook, metrics, figures, report PDF, slide PDF,
  dashboard Streamlit và Power BI data pack.
- Điểm cần nhớ: dữ liệu synthetic mạnh; mọi phân tích forecast/brand/preference
  phải nói là minh họa phương pháp, không phải kết luận kinh doanh thật.

## Nên giữ khi nộp bài hoặc push GitHub

- `app/`: dashboard Streamlit.
- `src/`: toàn bộ pipeline code.
- `scripts/`: runbook tái lập.
- `tests/`: kiểm thử.
- `docs/`: hướng dẫn, roadmap, progress, grading map.
- `data/raw/Enhanced_pizza_sell_data_2024-25.xlsx`: giữ để nộp offline không phụ thuộc Kaggle/network.
- `data/processed/*.csv`: giữ để giảng viên mở nhanh và dashboard chạy nhanh.
- `models/best_delay_model.joblib`: giữ để dashboard/demo không cần train lại.
- `reports/metrics/`: giữ vì là nguồn số liệu kỹ thuật.
- `reports/figures/`: giữ vì report/slide dùng trực tiếp.
- `reports/PIZZA_DSS_REPORT.pdf`, `reports/REPORT_GUIDE.md` và `reports/latex/main.tex`: giữ PDF nộp, hướng dẫn nội dung và source LaTeX.
- `slides/PIZZA_DSS_SLIDE_DECK.pdf`, `slides/SLIDE_GUIDE.md` và `slides/pizza_dss_slides.tex`: giữ slide nộp, speaking notes và source.
- `notebooks/*.ipynb`: giữ vì là minh chứng quy trình theo module.
- `powerbi/`: giữ vì user yêu cầu có dashboard Power BI-ready, gồm cả `POWERBI_BUILD_GUIDE.md`.

## Nên xóa hoặc không đưa vào GitHub

Nhóm này không ảnh hưởng tái lập, có thể xóa trước khi nộp/commit:

- `.venv/`: môi trường local, rất nặng, đang khoảng 695 MB.
- `.streamlit.log`, `.streamlit.err.log`, `.streamlit.pid`: file runtime local.
- `__pycache__/`, `*.pyc`: cache Python.
- `.pytest_cache/`: cache test.
- `reports/latex/main.aux`, `main.log`, `main.out`, `main.toc`, `main.lof`,
  `main.lot`, `main.pdf`: build trung gian. PDF chính đã được copy ra
  `reports/PIZZA_DSS_REPORT.pdf`.
- `slides/pizza_dss_slides.aux`, `.log`, `.nav`, `.out`, `.snm`, `.toc`, `pizza_dss_slides.pdf`: build trung gian. PDF chính đã được copy ra `slides/PIZZA_DSS_SLIDE_DECK.pdf`.
- `data/raw/*.zip`: zip download tạm nếu có.

## Lệnh dọn sạch an toàn

Chạy trong `pizza_delivery_dss/`:

```powershell
$root = (Resolve-Path .).Path
$targets = @(
  ".streamlit.log",
  ".streamlit.err.log",
  ".streamlit.pid",
  "reports/latex/main.aux",
  "reports/latex/main.log",
  "reports/latex/main.out",
  "reports/latex/main.toc",
  "reports/latex/main.lof",
  "reports/latex/main.lot",
  "reports/latex/main.pdf",
  "slides/pizza_dss_slides.aux",
  "slides/pizza_dss_slides.log",
  "slides/pizza_dss_slides.nav",
  "slides/pizza_dss_slides.out",
  "slides/pizza_dss_slides.snm",
  "slides/pizza_dss_slides.toc",
  "slides/pizza_dss_slides.pdf"
)
foreach ($item in $targets) {
  if (Test-Path $item) {
    $resolved = (Resolve-Path $item).Path
    if (-not $resolved.StartsWith($root)) { throw "Refusing to delete outside project: $resolved" }
    Remove-Item -LiteralPath $resolved -Force
  }
}
Get-ChildItem -Path app,scripts,src,tests -Recurse -Directory -Filter __pycache__ |
  ForEach-Object {
    $resolved = (Resolve-Path $_.FullName).Path
    if (-not $resolved.StartsWith($root)) { throw "Refusing to delete outside project: $resolved" }
    Remove-Item -LiteralPath $resolved -Recurse -Force
  }
```

Không xóa `.venv/` nếu vẫn muốn chạy lại ngay. Nếu đóng gói hoặc push GitHub thì
không đưa `.venv/` lên, vì `.gitignore` đã chặn.

## Kiểm tra sau khi dọn

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Get-ChildItem -Recurse -Include *.aux,*.log,*.out,*.toc,*.nav,*.snm reports,slides
git status --short --ignored -- .venv .streamlit.log .streamlit.err.log .streamlit.pid
```
