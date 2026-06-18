import sys
import textwrap
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "06_data_forensics.ipynb"


def code_cell(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def md_cell(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def img_cell(name):
    return code_cell(f'Image(filename=str(PROJECT_ROOT / "reports" / "figures" / "{name}"))')


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
            # Module 06 - Data Forensics và Reverse Engineering

            **Mục tiêu.** Dataset có nhiều dấu hiệu được sinh tự động. Notebook
            này ghi rõ cách phát hiện các công thức tất định, cách suy luận
            ngưỡng nhãn `is_delayed`, và cách quyết định biến nào chỉ dùng để
            audit/chẩn đoán thay vì đưa vào mô hình dự báo.
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
            from IPython.display import Image
            from pizza_dss.data_loader import load_dataset
            from pizza_dss.data_forensics import (
                bootstrap_brand_delta_f2,
                brand_ablation,
                brand_homogeneity_tests,
                build_forensics_artifacts,
                deterministic_formula_audit,
                duration_generator_reconstruction,
                duration_model_recovery,
                feature_information_audit,
                infer_delay_threshold,
                mi_permutation_audit,
                uniformity_tests,
            )
            """
        ),
        md_cell("Sinh trước các hình forensics để hiển thị inline ở từng mục."),
        code_cell("_ = build_forensics_artifacts()"),
        md_cell(
            """
            ## 0. Định nghĩa thuật ngữ forensics

            - **Tất định (deterministic)**: cột tính được chính xác từ cột khác bằng
              một công thức; sai số tối đa ~0.
            - **Lượng tử hóa (quantization)**: giá trị bị "ép" về một lưới rời rạc
              (ở đây duration chỉ nhận bội số 5 phút).
            - **Mutual Information (MI)**: lượng thông tin (bit) mà một biến cho biết
              về nhãn; 0 = độc lập. **MI có điều kiện theo distance band** = MI còn
              lại *sau khi đã biết quãng đường*.
            - **Permutation test**: xáo trộn ngẫu nhiên nhãn nhiều lần để dựng "mức
              nền do may rủi"; nếu giá trị quan sát không vượt mức nền thì đó là
              **nhiễu ước lượng**, không phải tín hiệu thật. Dùng để kiểm tra MI có
              bị thổi phồng bởi biến nhiều mức / mẫu nhỏ hay không.
            - **Ablation**: bỏ một feature rồi đo mức sụt metric, để định lượng đóng
              góp thực của feature đó.
            - **Bootstrap CI**: lấy mẫu lại có hoàn lại để ước lượng khoảng tin cậy;
              nếu CI chứa 0 thì hiệu ứng nằm trong nhiễu.
            """
        ),
        md_cell(
            """
            ## 1. Nguyên tắc truy ngược

            **Vì sao bước này?**
            - Làm gì: Đặt ra quy trình 5 bước nghiêm ngặt để xác định xem một cột có phải là "công thức tất định" (được sinh tự động từ cột khác) hay không.
            - Vì sao: Để tránh kết luận vội vàng. Một cột chỉ bị coi là "tất định" khi tìm ra ĐÚNG công thức toán học tạo ra nó với sai số (max_abs_error) bằng 0.
            - Kỹ thuật: Reverse Engineering, Symbolic Regression (thủ công/heuristic).
            - Bằng chứng dẫn tới: Nếu dự báo bằng một cột vốn dĩ là công thức từ nhãn thì mô hình sẽ bị data leakage 100%. Cần nguyên tắc này làm ranh giới loại bỏ feature.
            
            Cách làm không phải đoán bằng cảm tính. Với mỗi cột nghi là công
            thức, nhóm:

            1. nhìn tên cột và quan hệ nghiệp vụ có thể có;
            2. đề xuất biểu thức ứng viên;
            3. tính biểu thức cho toàn bộ 1.004 dòng;
            4. đo `max_abs_error` giữa cột gốc và biểu thức;
            5. chỉ gọi là tất định nếu sai số tối đa bằng 0 hoặc xấp xỉ 0.
            """
        ),
        code_cell(
            """
            df = load_dataset()
            df.shape
            """
        ),
        md_cell("## 2. Suy luận ngưỡng nhãn delayed"),
        md_cell(
            """
            **Vì sao bước này?**
            - Làm gì: Tìm luật (logic) được dùng để gán nhãn `is_delayed` = True/False từ cột thời gian `delivery_duration_min`.
            - Vì sao: Dataset không có file giải thích (SLA) báo bao nhiêu phút thì bị tính là trễ. Phải quét ngưỡng (threshold sweep) để tự tìm ra.
            - Kỹ thuật: Threshold Sweep, Logic Inference.
            - Bằng chứng dẫn tới: Mọi đơn `is_delayed = False` đều có `duration <= 30`, mọi đơn `True` đều có `duration >= 35`. Nghĩa là ngưỡng cắt (cut-off) thực sự nằm ở 30 phút.
            """
        ),
        code_cell('pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "delay_threshold_inference.csv").head(12)'),
        md_cell(
            """
            **Insight.** File chỉ cho nhãn `is_delayed`, không cho SLA. Threshold
            sweep cho thấy `duration > 30` và `duration >= 35` đều khớp nhãn
            tuyệt đối vì duration nằm trên lưới 5 phút. Do đó báo cáo phải nói
            đây là ranh giới suy luận từ dữ liệu, không phải quy định thực tế.
            """
        ),
        img_cell("delay_threshold_inference.png"),
        md_cell("## 3. Bảy công thức/target tất định"),
        code_cell(
            """
            deterministic_formula_audit(df)[
                ["column", "recovered_formula", "max_abs_error", "how_found", "verification_method", "matches_exactly"]
            ]
            """
        ),
        md_cell(
            """
            **Quyết định.** Các cột hậu nghiệm như `delivery_duration_min`,
            `delay_min`, `delivery_efficiency_min_per_km` và
            `restaurant_avg_time` bị chặn khỏi feature model. Các cột trùng
            thông tin như `estimated_duration_min`, `topping_density`,
            `pizza_complexity`, `traffic_impact` được giữ cho audit/giải thích
            nhưng không dùng trong feature set compact.
            """
        ),
        img_cell("generator_deterministic_formula_errors.png"),
        md_cell("## 4. Luật lượng tử hóa duration"),
        code_cell(
            """
            duration_model_recovery(df).round(4)
            """
        ),
        md_cell(
            """
            **Insight.** Duration chỉ có các mốc 15, 20, ..., 50 phút. Mô hình
            tuyến tính theo distance/traffic/complexity giải thích được phần lớn
            duration nhưng residual vẫn phản ánh việc generator snap về bội số
            5 phút.
            """
        ),
        img_cell("duration_model_residual_histogram.png"),
        md_cell(
            """
            ### 4b. Tái dựng generator của duration

            Nếu duration thật sự là `round_to_5(base + noise)`, ta fit `base` theo
            distance+traffic rồi đo: làm tròn dự đoán về lưới 5 phút có khớp duration
            thật bao nhiêu %, và nhiễu lớn cỡ nào.
            """
        ),
        code_cell('pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "duration_generator_reconstruction.csv")'),
        md_cell(
            """
            **Insight.** R² cao + tỷ lệ `round5(dự đoán)` khớp duration lớn ⇒ duration
            gần như được sinh bởi một hàm xác định theo distance/traffic rồi làm tròn
            5 phút, phần nhiễu nhỏ. Đây là lý do mô hình dự báo trễ "quá dễ": tín
            hiệu gần tất định, không phản ánh độ khó thực tế.
            """
        ),
        md_cell("## 5. Categorical artifact và tính đồng đều"),
        code_cell(
            """
            feature_information_audit(df).round(6)
            """
        ),
        code_cell(
            """
            uniformity_tests(df).round(6)
            """
        ),
        md_cell(
            """
            **Insight.** Một số categorical như location và pizza type còn tín
            hiệu sau khi kiểm soát distance band. Vì vậy không thể nói mọi biến
            categorical chỉ là noise độc lập; phải ghi rõ chúng có artifact từ
            generator.
            """
        ),
        img_cell("feature_information_audit.png"),
        img_cell("uniformity_tests.png"),
        md_cell(
            """
            ### 5b. Permutation test cho MI (chứng minh mức nhiễu nền)

            **Vì sao bước này?**
            - Làm gì: Tính toán Mutual Information (MI) để đo sự liên quan của biến tới nhãn, kết hợp với Permutation Test (xáo trộn nhãn 200 lần) để tìm "mức nhiễu nền" (noise floor).
            - Vì sao: MI dễ bị "ảo" (chỉ số cao giả tạo) với các cột có nhiều giá trị độc nhất (high cardinality như `location`, `restaurant_name`) đặc biệt khi dataset quá nhỏ. Permutation test tạo ra 1 base line: nếu MI gốc không lớn hơn MI sinh ra do xáo trộn ngẫu nhiên thì MI đó là vô nghĩa (artifact).
            - Kỹ thuật: Conditional Mutual Information, Permutation Test.
            - Bằng chứng dẫn tới: `is_weekend` hay `restaurant_name` tưởng chừng có ảnh hưởng nhưng p-value của MI lại rớt vào vùng nhiễu nền. Xác minh được các cột này chỉ sinh ra "rác" nếu đưa vào model dự báo.
            
            MI plug-in bị thổi phồng với biến nhiều mức và mẫu nhỏ. Cell dưới xáo
            trộn nhãn để dựng mức nền do may rủi, rồi so với MI quan sát.
            """
        ),
        code_cell('pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "mi_permutation_audit.csv").round(6)'),
        md_cell(
            """
            **Insight.** Những feature có `p_value` lớn (vd `restaurant_name`,
            `is_weekend`) nằm **trong mức nhiễu nền** — conditional MI dương của
            chúng là *artifact ước lượng*, không phải tín hiệu thật. Chỉ feature có
            `verdict = signal_beyond_chance` mới thực sự liên quan nhãn sau khi kiểm
            soát distance. Đây là bằng chứng định lượng cho caveat MI-bias.
            """
        ),
        md_cell("## 6. Brand có gộp thành một chuỗi được không?"),
        md_cell(
            """
            **Vì sao bước này?**
            - Làm gì: Đo lường xem cột Brand (`restaurant_name`) có tác dụng phân loại thực sự hay không. Dùng **Homogeneity test** (đo độ đồng nhất) và **Ablation test** kết hợp **Bootstrap** (đo độ sụt giảm F2 khi vứt bỏ cột Brand).
            - Vì sao: Có 5 nhà hàng (brand), người ta thường muốn so sánh nhà hàng nào tệ hơn. Nhưng nếu sự khác biệt giữa các nhà hàng chỉ nằm trong mức độ nhiễu ngẫu nhiên (chênh lệch vài ba đơn do mẫu quá nhỏ), thì việc đánh giá "brand nào tốt/xấu" sẽ dẫn đến sai lầm kinh doanh.
            - Kỹ thuật: Ablation Test, Bootstrap CI, Homogeneity Test.
            - Bằng chứng dẫn tới: `ci_includes_zero = True` trong phần Bootstrap (F2 thay đổi không vượt quá mốc 0) chứng tỏ việc bỏ cột brand không làm giảm năng lực dự báo trễ của model, brand chỉ là biến nhiễu/sinh ngẫu nhiên, không mang tín hiệu dự báo đáng kể.
            """
        ),
        code_cell(
            """
            brand_homogeneity_tests(df).round(4)
            """
        ),
        code_cell(
            """
            brand_ablation().round(4)
            """
        ),
        md_cell(
            """
            **Insight.** Có thể gộp brand để báo cáo cấp chuỗi, nhưng kiểm định
            homogeneity không ủng hộ kết luận các brand giống hệt nhau. Vì vậy
            dashboard nên cho phép lọc theo brand, còn báo cáo không kết luận
            chất lượng thật của từng hãng.
            """
        ),
        img_cell("brand_delay_rate_homogeneity.png"),
        md_cell(
            """
            **Caveat phương pháp.** MI có điều kiện theo distance band bị thiên
            lệch dương với biến nhiều mức (location 84, type 12), và `restaurant_name`
            có conditional MI tăng so với raw MI — dấu hiệu nhiễu mẫu nhỏ. Vì vậy
            verdict "artifact" của các biến cardinality cao cần đọc thận trọng; với
            brand, ưu tiên bằng chứng ablation ΔF2 hơn là MI. Kết luận brand cũng
            chỉ trong giới hạn dev nhỏ (~42 ca trễ).
            """
        ),
        md_cell(
            """
            ### 6b. Bootstrap khoảng tin cậy cho ΔF2 của brand

            Lấy mẫu lại dev để xem hiệu ứng F2 khi thêm `restaurant_name` có khác 0
            một cách đáng kể không.
            """
        ),
        code_cell('pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "brand_delta_f2_bootstrap.csv")'),
        md_cell(
            """
            **Insight.** Nếu `ci_includes_zero = True`, mức ΔF2 của brand **nằm trong
            nhiễu** ở cỡ dev này — không đủ bằng chứng để nói brand giúp dự báo. Kết
            hợp với homogeneity test, brand chỉ dùng để *lọc/mô tả* trên dashboard,
            tuyệt đối không kết luận "hãng tốt/xấu".
            """
        ),
        md_cell(
            """
            ## 7. Xác nhận artifact

            CSV/figure forensics đã sinh **một lần** ở Bước 0
            (`build_forensics_artifacts`). Cell dưới chỉ liệt kê, không tính lại.
            """
        ),
        code_cell(
            """
            import os
            forensic_files = sorted(
                f for f in os.listdir(PROJECT_ROOT / "reports" / "metrics")
                if f.startswith(("generator_", "duration_", "feature_information", "uniformity", "brand_", "mi_", "delay_threshold"))
            )
            {"n_forensic_metric_files": len(forensic_files), "files": forensic_files}
            """
        ),
    ]
    return nb


def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    NotebookClient(
        nb,
        timeout=420,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    ).execute()
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Saved executed notebook -> {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
