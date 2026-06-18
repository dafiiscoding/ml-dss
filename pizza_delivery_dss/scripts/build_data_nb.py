import textwrap
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_data_audit_preprocessing.ipynb"


def code_cell(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def md_cell(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.cells = [
        md_cell(
            """
            # Module 01 - Data Audit and Preprocessing

            Notebook này kiểm tra dữ liệu gốc, chuẩn hóa schema, xác định leakage
            columns và tạo split train/dev/test cố định.
            """
        ),
        md_cell(
            """
            > **Đối tượng & cách đọc.** Notebook cho **người mới** trong nhóm. Lời
            > giải thích bằng tiếng Việt; thuật ngữ tiếng Anh (F2, MCC, bootstrap…)
            > được định nghĩa ở mục **"0. Định nghĩa"** và đầy đủ trong
            > `docs/GLOSSARY.md`. Mỗi mục theo mạch *mục tiêu → phân tích → insight*;
            > mỗi bảng/biểu đồ có **một dòng đọc-hiểu ngay bên dưới**.
            """
        ),
        code_cell(
            """
            import sys
            from pathlib import Path

            PROJECT_ROOT = Path.cwd()
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

            import pandas as pd
            from pizza_dss.business_analysis import dtype_audit, redundant_feature_audit, synthetic_data_audit
            from pizza_dss.config import FEATURE_COLUMNS, LEAKAGE_COLUMNS
            from pizza_dss.data_loader import (
                audit_dataset,
                export_processed_splits,
                load_dataset,
                validate_feature_contract,
            )
            from pizza_dss.data_forensics import infer_delay_threshold
            """
        ),
        md_cell(
            """
            ## Bức tranh tổng (đọc trước khi đi vào chi tiết)

            Sơ đồ dưới là toàn bộ luồng dự án, từ dữ liệu thô đến quyết định vận
            hành. Mỗi ô là một notebook. Người mới nên xem hình này trước rồi đi
            lần lượt 01 → 06.
            """
        ),
        code_cell(
            """
            import sys as _sys
            from IPython.display import Image
            _sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.build_overview_figure import build as _build_overview
            Image(filename=str(_build_overview()))
            """
        ),
        md_cell(
            """
            ## 0. Định nghĩa thuật ngữ tiền xử lý

            - **Schema**: tập cột + kiểu dữ liệu. **snake_case**: chuẩn hóa tên cột.
            - **Leakage (rò rỉ)**: dùng thông tin chỉ biết *sau* khi giao hàng làm
              feature dự báo → kết quả ảo. Ở đây gồm duration/delay/efficiency/
              restaurant_avg_time vì chúng định nghĩa hoặc suy ra nhãn.
            - **Stratified split**: chia train/dev/test giữ nguyên tỷ lệ trễ ở mỗi
              phần, để dev/test đại diện đúng cho phân phối nhãn.
            - **Feature contract**: ràng buộc code chặn mọi feature nằm trong danh
              sách leakage (raise lỗi nếu vi phạm).
            """
        ),
        md_cell("## 1. Load and schema"),
        code_cell(
            """
            df = load_dataset()
            df.shape, df.head()
            """
        ),
        md_cell(
            """
            **Vì sao bước này?**
            - Làm gì: Đổi tên cột sang định dạng `snake_case` và sửa lỗi chính tả "Marco’s Pizza" (trong `load_dataset`).
            - Vì sao: Để code dễ đọc, nhất quán theo chuẩn Python, và gộp đúng các nhóm nhà hàng thay vì bị tách lẻ do dấu nháy khác nhau.
            - Kỹ thuật: Renaming & String replacement.
            - Bằng chứng dẫn tới: Dữ liệu thô dùng `Title Case` có khoảng trắng (vd `Order ID`), cản trở truy xuất `df.order_id`.
            """
        ),
        code_cell(
            """
            dtype_audit(df)
            """
        ),
        md_cell("**Insight (schema).** 1.004 đơn, 37 cột sau FE, không có cột nào sai kiểu rõ rệt; bảng này là tham chiếu kiểu/dtype cho các bước sau."),
        md_cell(
            """
            ### 1b. Audit lẫn int/float

            Dữ liệu trộn kiểu int và float. Cell dưới chỉ ra cột nào *lưu dạng float
            nhưng giá trị thực ra là số nguyên* — dấu hiệu generator sinh số rồi ép
            kiểu, và là lý do ta mã hóa/ép kiểu nhất quán khi xây feature.
            """
        ),
        code_cell(
            """
            numeric = df.select_dtypes("number")
            audit = pd.DataFrame({
                "column": numeric.columns,
                "dtype": [str(numeric[c].dtype) for c in numeric.columns],
                "is_integer_valued": [bool((numeric[c].dropna() % 1 == 0).all()) for c in numeric.columns],
            })
            audit["stored_as_float_but_integer"] = audit["dtype"].str.startswith("float") & audit["is_integer_valued"]
            audit
            """
        ),
        md_cell("**Insight (int/float).** Các cột `stored_as_float_but_integer=True` (vd toppings, order_hour) bản chất là số nguyên bị lưu float — vô hại cho mô hình sau khi StandardScaler, nhưng được ghi nhận như một dấu hiệu synthetic."),
        md_cell("## 2. Feature engineering cho size/time/complexity"),
        md_cell(
            """
            **Vì sao bước này?**
            - Làm gì: Tạo các cột biến đổi (feature engineering) từ dữ liệu gốc, bao gồm `pizza_size_score` (ordinal), `time_segment` (nhóm giờ), `complexity_band`, `distance_band` (khoảng), và `order_period`.
            - Vì sao: 
              - `pizza_size` có thứ tự (Small < Medium < Large < XL) nên dùng Ordinal Encoding (`pizza_size_score` 1..4) giúp mô hình bắt được xu hướng tăng dần. Cột `code` và `label` dùng cho Dashboard hiển thị đẹp.
              - Nhóm giờ (`time_segment`), độ phức tạp (`complexity_band`), khoảng cách (`distance_band`) giúp giảm nhiễu (noise) và bắt được các pattern gộp (vd: "đơn xa trên 8km") hữu ích cho cây quyết định. Các ngưỡng chia dựa trên phân phối thực tế của dữ liệu.
            - Kỹ thuật: Ordinal Encoding, Binning (pd.cut).
            - Bằng chứng dẫn tới: `pizza_size` mang tính thứ bậc tự nhiên. Các ngưỡng `time_segment` (0,11,15,17,21,24) khớp với các ca hoạt động thực tế (Lunch, Afternoon, Dinner, Late).
            """
        ),
        md_cell(
            """
            **Vì sao FE `pizza_size_score`?**
            - Làm gì: Ánh xạ Small/Medium/Large/XL thành 1/2/3/4.
            - Vì sao: Size có thứ tự tự nhiên; XL lớn hơn Large, Large lớn hơn Medium.
              Ordinal score giữ được quan hệ tăng dần để mô hình tuyến tính đọc được.
            - Kỹ thuật: Ordinal encoding.
            - Bằng chứng dẫn tới: `SIZE_SCORE` trong `data_loader.py` và
              `generator_deterministic_formulas.csv` cho thấy `pizza_complexity =
              toppings_count * pizza_size_score`, tức generator cũng xem size là
              thang thứ bậc.
            """
        ),
        md_cell(
            """
            **Vì sao FE `pizza_size_code` / `pizza_size_label`?**
            - Làm gì: Tạo mã `01`..`04` và nhãn `01_Small`..`04_XL`.
            - Vì sao: Dashboard/notebook cần sort đúng thứ tự size; nếu sort chữ
              đơn thuần thì Large/Medium/Small/XL dễ bị xếp sai ngữ nghĩa.
            - Kỹ thuật: Label engineering phục vụ báo cáo/BI, không phải feature
              chính cho model compact.
            - Bằng chứng dẫn tới: `size_mix.csv` và Power BI pack cần hiển thị size
              theo thứ tự Small → Medium → Large → XL.
            """
        ),
        md_cell(
            """
            **Vì sao FE `time_segment`?**
            - Làm gì: Gom `order_hour` vào Other/Lunch/Afternoon/Dinner/Late bằng
              các mốc 0, 11, 15, 17, 21, 24.
            - Vì sao: Điều phối nhân sự thường ra quyết định theo ca, không theo
              từng giờ rời rạc; dinner peak là ngữ cảnh vận hành quan trọng.
            - Kỹ thuật: Binning bằng `pd.cut`.
            - Bằng chứng dẫn tới: `hourly_staffing_plan.csv` chỉ ra peak hour 19h,
              nằm trong nhóm Dinner.
            """
        ),
        md_cell(
            """
            **Vì sao FE `complexity_band` và `distance_band`?**
            - Làm gì: Gom complexity thành Low/Medium/High/Very High và distance
              thành 0-3/3-6/6-8/8+ km.
            - Vì sao: Band giúp EDA, association rules và dashboard đọc rủi ro theo
              nhóm vận hành thay vì nhìn số liên tục khó diễn giải.
            - Kỹ thuật: Binning có ngưỡng cố định.
            - Bằng chứng dẫn tới: `delay_rate_by_distance_band.csv` và
              `delay_rate_by_complexity_band.csv` cho thấy delay rate thay đổi theo
              các dải này; `risk_calibration.csv` dùng cùng tư duy band để kiểm tra
              Risk Score.
            """
        ),
        md_cell(
            """
            **Vì sao FE `order_period` / `order_weekday`?**
            - Làm gì: Tách tháng dạng `YYYY-MM`, thứ trong tuần dạng số và tên ngày.
            - Vì sao: Forecast cần chuỗi theo tháng; customer behavior và staffing
              cần biết weekday/weekend hoặc ngày trong tuần.
            - Kỹ thuật: Datetime feature extraction.
            - Bằng chứng dẫn tới: `monthly_demand_forecast.csv`,
              `preference_trend_forecast.csv` và `hourly_staffing_plan.csv` đều cần
              feature thời gian đã chuẩn hóa.
            """
        ),
        code_cell(
            """
            df[[
                "pizza_size", "pizza_size_score", "pizza_size_code", "pizza_size_label",
                "time_segment", "complexity_band", "distance_band", "order_period",
                "order_weekday", "order_weekday_name"
            ]].head()
            """
        ),
        code_cell(
            """
            df.groupby(["pizza_size_score", "pizza_size_label"]).size().reset_index(name="orders")
            """
        ),
        md_cell("## 3. Leakage and synthetic-data audit"),
        md_cell(
            """
            **Vì sao bước này?**
            - Làm gì: Phát hiện và loại bỏ các cột Leakage (rò rỉ tương lai) và các cột Redundant (dư thừa, công thức tất định).
            - Vì sao: 
              - Cột **Leakage** (`delivery_duration_min`, `delay_min`, `delivery_efficiency_min_per_km`, `restaurant_avg_time`, `delivery_time`) chứa thông tin chỉ có *sau khi* giao xong. Dùng chúng mô hình sẽ "học vẹt" đáp án, accuracy 100% khi train nhưng vô dụng khi chạy thực tế (lúc vừa nhận đơn chưa có thời gian giao).
              - Cột **Redundant** (`estimated_duration_min`, `topping_density`, `pizza_complexity`, `traffic_impact`) bị loại trong tập feature gọn (`COMPACT_FEATURE_COLUMNS`) để tránh đa cộng tuyến.
            - Kỹ thuật: Feature Selection (Leakage detection, Deterministic formula discovery).
            - Bằng chứng dẫn tới: `redundant_feature_audit.csv` và `generator_deterministic_formulas.csv` cho thấy sai số max_abs_error ≈ 0 khi phục hồi công thức. Nghĩa là chúng được tính toán trực tiếp từ các cột khác.
            """
        ),
        md_cell(
            """
            **Vì sao cấm từng leakage column?**
            - Làm gì: Chặn 5 cột hậu nghiệm khỏi `FEATURE_COLUMNS`.
            - Vì sao: Khi đơn vừa được đặt, hệ thống chưa biết thời gian giao thật,
              thời gian kết thúc, hay thống kê trung bình tính từ kết quả giao.
            - Kỹ thuật: Leakage audit + feature contract.
            - Bằng chứng dẫn tới:
              `delivery_duration_min` phục dựng nhãn với mismatch 0;
              `delay_min = delivery_duration_min - estimated_duration_min`;
              `delivery_efficiency_min_per_km = delivery_duration_min / distance_km`;
              `restaurant_avg_time` là thống kê hậu nghiệm theo nhà hàng;
              `delivery_time` chỉ có sau khi giao xong. Các quan hệ này nằm trong
              `data_quality_summary.json`, `redundant_feature_audit.csv` và
              `generator_deterministic_formulas.csv`.
            """
        ),
        code_cell(
            """
            pd.DataFrame([
                {"column": "delivery_duration_min", "why_blocked": "Actual duration after delivery; reconstructs is_delayed with 0 mismatch."},
                {"column": "delay_min", "why_blocked": "Formula from actual duration minus estimated duration."},
                {"column": "delivery_efficiency_min_per_km", "why_blocked": "Formula from actual duration divided by distance."},
                {"column": "restaurant_avg_time", "why_blocked": "Post-hoc restaurant aggregate derived from delivery outcomes."},
                {"column": "delivery_time", "why_blocked": "Timestamp known only after the order has been delivered."},
            ])
            """
        ),
        md_cell(
            """
            **Vì sao bỏ cột deterministic khỏi compact model?**
            - Làm gì: Compact feature set giữ biến gốc và bỏ các biến công thức như
              `estimated_duration_min`, `topping_density`, `pizza_complexity`,
              `traffic_impact`.
            - Vì sao: Các cột này không thêm thông tin mới; giữ lại dễ làm mô hình
              phụ thuộc vào bản sao của cùng tín hiệu và khó giải thích hơn.
            - Kỹ thuật: Redundant feature removal.
            - Bằng chứng dẫn tới: `redundant_feature_audit.csv` ghi max_abs_error
              xấp xỉ 0 cho các công thức; `feature_set_comparison.csv` cho thấy
              compact vẫn đạt F2 ngang hoặc tốt hơn full.
            """
        ),
        code_cell(
            """
            audit_dataset(df)
            """
        ),
        md_cell(
            """
            `is_delayed` là nhãn có sẵn trong file, nhưng file không giải thích
            SLA bao nhiêu phút thì trễ. Vì vậy ta suy luận ranh giới nhãn bằng
            cách thử các luật ngưỡng trên `delivery_duration_min` và đếm số dòng
            mismatch.
            """
        ),
        code_cell(
            """
            infer_delay_threshold(df).head(8)
            """
        ),
        code_cell(
            """
            synthetic_data_audit(df)
            """
        ),
        code_cell(
            """
            redundant_feature_audit(df)
            """
        ),
        code_cell(
            """
            {
                "feature_columns": FEATURE_COLUMNS,
                "blocked_leakage_columns": LEAKAGE_COLUMNS,
                "feature_contract_passed": validate_feature_contract(),
            }
            """
        ),
        md_cell(
            """
            ## 3b. Hình minh hoạ dữ liệu và ranh giới nhãn

            **Lý do.** Hai biểu đồ dưới cho thấy lớp trễ là thiểu số và nhãn nằm
            đúng trên lưới 5 phút: nhóm on-time tối đa 30 phút, nhóm delayed tối
            thiểu 35 phút.
            """
        ),
        code_cell(
            """
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            df["is_delayed"].value_counts().rename({False: "On-time", True: "Delayed"}).plot.bar(
                ax=axes[0], color=["#27ae60", "#c0392b"]
            )
            axes[0].set_title("Class balance (is_delayed)")
            axes[0].set_ylabel("Orders")
            axes[0].tick_params(axis="x", rotation=0)
            for label, group in df.groupby("is_delayed"):
                axes[1].hist(
                    group["delivery_duration_min"],
                    bins=range(13, 56, 5),
                    alpha=0.6,
                    label=("Delayed" if label else "On-time"),
                )
            axes[1].axvspan(30, 35, color="grey", alpha=0.25, label="inferred boundary")
            axes[1].set_title("Delivery duration by label")
            axes[1].set_xlabel("Delivery duration (min)")
            axes[1].set_ylabel("Orders")
            axes[1].legend()
            fig.tight_layout()
            plt.show()
            """
        ),
        md_cell(
            """
            **Insight.** Tỷ lệ trễ ~21% nên Accuracy đơn lẻ sẽ đánh lừa; cần F2/
            Recall cùng Balanced Accuracy/MCC. Ranh giới nhãn là suy luận từ dữ
            liệu, củng cố quyết định chặn duration/delay khỏi feature.
            """
        ),
        md_cell("## 4. Processed splits"),
        md_cell(
            """
            **Vì sao bước này?**
            - Làm gì: Chia tập dữ liệu thành Train/Dev/Test theo tỷ lệ 60/20/20 bằng phương pháp Stratified (giữ nguyên tỷ lệ nhãn `is_delayed` trong từng tập).
            - Vì sao: Để đảm bảo mọi tập dữ liệu đều có tỷ lệ trễ ~21%. Nếu chia theo thời gian (chronological split), phần đuôi (last-20%) hoàn toàn không có đơn bị trễ, khiến tập Test trở nên vô nghĩa không thể đánh giá khả năng bắt lỗi trễ.
            - Kỹ thuật: Stratified Split.
            - Bằng chứng dẫn tới: Báo cáo `data_quality_summary.json` ghi nhận `chronological_last20_delayed = 0`, chứng tỏ dữ liệu mất cân bằng theo trục thời gian ở đoạn cuối.
            """
        ),
        code_cell(
            """
            export_processed_splits()
            """
        ),
        code_cell(
            """
            splits = {
                name: pd.read_csv(PROJECT_ROOT / "data" / "processed" / f"{name}.csv")
                for name in ["train", "dev", "test"]
            }
            pd.DataFrame([
                {
                    "split": name,
                    "rows": len(frame),
                    "delayed": int(frame["is_delayed"].sum()),
                    "delayed_rate": frame["is_delayed"].mean(),
                }
                for name, frame in splits.items()
            ])
            """
        ),
        md_cell(
            """
            ### 4b. Kiểm tra split không rò rỉ

            Hai điều phải đúng: (1) không `order_id` nào xuất hiện ở hơn một split;
            (2) tỷ lệ trễ được giữ gần như nhau giữa các split (stratify).
            """
        ),
        code_cell(
            """
            ids = {name: set(frame["order_id"]) for name, frame in splits.items()}
            overlaps = {
                "train∩dev": len(ids["train"] & ids["dev"]),
                "train∩test": len(ids["train"] & ids["test"]),
                "dev∩test": len(ids["dev"] & ids["test"]),
            }
            pd.DataFrame([
                {"check": "row_id_overlaps_between_splits", **overlaps},
            ])
            """
        ),
        md_cell("**Insight (rò rỉ split).** Mọi giao tập `order_id` = 0 → không đơn nào lọt sang split khác; cùng với tỷ lệ trễ ~21% giữ đều (bảng mục 4), split là leakage-safe và đại diện."),
        md_cell("## 5. Insight → quyết định tiền xử lý"),
        code_cell(
            """
            pd.DataFrame([
                {
                    "question": "Có missing/duplicate không?",
                    "evidence": "missing_total=0, duplicate_order_ids=0 trong audit_dataset.",
                    "decision": "Không cần imputation; nhưng zero-missing là dấu hiệu synthetic, ghi vào audit.",
                },
                {
                    "question": "Nhãn is_delayed định nghĩa thế nào?",
                    "evidence": "is_delayed == (delivery_duration_min > 30), mismatch=0.",
                    "decision": "Chặn duration/delay/efficiency/restaurant_avg_time khỏi feature.",
                },
                {
                    "question": "Cột nào trùng thông tin?",
                    "evidence": "estimated_duration=2.4*distance; complexity=toppings*size; density=toppings/distance.",
                    "decision": "Dùng compact feature set, bỏ cột công thức, giữ pizza_size_score.",
                },
                {
                    "question": "Split thế nào?",
                    "evidence": "Chronological last-20% có 0 delayed.",
                    "decision": "Dùng stratified train/dev/test 602/201/201, ghi rõ giới hạn.",
                },
            ])
            """
        ),
        md_cell(
            """
            Kết luận: dataset gốc đủ nhỏ để xử lý offline bằng pandas/scikit-learn.
            Pipeline giữ raw Excel trong thư mục dự án con và sinh processed CSV
            để notebook, dashboard và PowerBI dùng chung.
            """
        ),
    ]
    return nb


def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    ).execute()
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Saved executed notebook -> {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
