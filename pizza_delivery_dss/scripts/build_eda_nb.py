import sys
import textwrap
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_eda.ipynb"


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
            # Module 02 - Exploratory Data Analysis

            **Mục tiêu.** Notebook này không chỉ vẽ thống kê mô tả. Mỗi phần
            đều đi theo logic: đặt câu hỏi vận hành, chọn phép phân tích, đọc
            kết quả, rồi chuyển insight thành quyết định cho mô hình/DSS.

            **Nguyên tắc đọc.**

            1. Các cột sau giao hàng được dùng để hiểu dữ liệu, không dùng làm
               feature dự báo.
            2. `location` có dạng "City, ST"; `state_code` trong notebook được
               suy ra từ chuỗi này, không phải cột bang chính thức.
            3. Vì dữ liệu có dấu hiệu synthetic, insight được dùng cho báo cáo
               học thuật và dashboard, không kết luận nhân quả kinh doanh.
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
            from pizza_dss.business_analysis import customer_preference_tables, hypothesis_tests
            from pizza_dss.data_loader import audit_dataset, load_dataset
            from pizza_dss.eda import (
                add_distance_band,
                add_location_parts,
                association_rules_to_delay,
                build_eda_artifacts,
                delay_rate_by,
                delay_rate_with_ci,
                kmeans_cluster_profile,
                delay_severity_distribution,
                duration_delay_profile,
                duration_grid_by_delay,
                favorite_item_summary,
                kmeans_silhouette_sweep,
                location_dependency_summary,
                restaurant_dependency_summary,
                restaurant_mix_matrix,
                state_dependency_summary,
                state_mix_matrix,
            )
            """
        ),
        md_cell(
            """
            **Sinh trước các hình.** Cell dưới tạo toàn bộ PNG dùng chung cho
            notebook, slide và báo cáo, để các mục sau hiển thị biểu đồ inline.
            """
        ),
        code_cell("_ = build_eda_artifacts()"),
        md_cell(
            """
            ## 0. Định nghĩa thuật ngữ EDA

            - **Delay rate** = tỷ lệ đơn trễ trong một nhóm.
            - **Khoảng tin cậy Wilson (95%)**: dải giá trị hợp lý của delay rate khi
              nhóm ít đơn; ưu tiên hơn xấp xỉ chuẩn vì không vỡ ở tỷ lệ cực đoan/n
              nhỏ. Hai nhóm có CI chồng nhau ⇒ chưa đủ bằng chứng khác biệt.
            - **Chi-square độc lập**: kiểm định biến phân loại có liên hệ với nhãn
              trễ không; `p < 0.05` = có bằng chứng liên hệ.
            - **Cramér's V**: độ mạnh liên hệ trong [0,1]; quy ước thô: <0.1 không
              đáng kể, 0.1-0.3 yếu, 0.3-0.5 vừa, >0.5 mạnh.
            - **Silhouette**: chất lượng phân cụm trong [-1,1]; cao = cụm tách tốt.
            - **Association rule**: `support` = tần suất xuất hiện chung; `confidence`
              = P(trễ | điều kiện); `lift` = confidence / tỷ lệ trễ chung (>1 = điều
              kiện làm tăng khả năng trễ).
            """
        ),
        md_cell(
            """
            ## 1. Load dữ liệu và câu hỏi EDA

            Trước khi modeling, cần hiểu dữ liệu trả lời được gì:

            - Tỷ lệ trễ và mức độ trễ phân bố như thế nào?
            - Traffic, distance, complexity, size/type liên quan gì đến trễ?
            - Món/size nào được đặt nhiều nhất?
            - Sở thích có khác theo restaurant, location hoặc state suy ra không?
            - Insight nào đủ an toàn để đưa vào DSS, dashboard hoặc feature set?
            """
        ),
        code_cell(
            """
            df = add_location_parts(add_distance_band(load_dataset()))
            audit_dataset(df)
            """
        ),
        code_cell(
            """
            df.head()
            """
        ),
        md_cell(
            """
            ## 2. Phân tích nhãn và độ trễ

            **Lý do làm.** Nếu chỉ biết `is_delayed`, ta chưa biết độ trễ nặng
            hay nhẹ. Vì vậy phần này xem duration, estimated duration và
            `delay_min` như biến chẩn đoán hậu nghiệm để hiểu generator và mức
            độ rủi ro. Những cột này không được dùng trong mô hình dự báo.
            """
        ),
        code_cell(
            """
            duration_delay_profile(df).round(3)
            """
        ),
        code_cell(
            """
            duration_grid_by_delay(df).round(3)
            """
        ),
        code_cell(
            """
            delay_severity_distribution(df).round(3)
            """
        ),
        md_cell(
            """
            **Insight.** Nhóm on-time có duration tối đa 30 phút, nhóm delayed
            có duration tối thiểu 35 phút. Duration chỉ nhận các giá trị bội số
            5 nên ranh giới nhãn nằm giữa 30 và 35 phút. Điều này dẫn tới quyết
            định chống leakage: duration/delay chỉ dùng để audit, không đưa vào
            feature dự báo.
            """
        ),
        img_cell("duration_grid_by_delay.png"),
        img_cell("delay_severity_distribution.png"),
        md_cell("## 3. Delay rate theo yếu tố vận hành"),
        md_cell(
            """
            **Vì sao bước này?**
            - Làm gì: Tính toán Delay Rate (tỷ lệ trễ) theo từng biến phân loại (traffic, distance, complexity, order_hour) và đính kèm Khoảng tin cậy Wilson (Wilson CI).
            - Vì sao: 
              - Khách hàng/Business thường hỏi "Đơn xa thì trễ nhiều hơn không?", bảng này trả lời trực tiếp câu hỏi đó. 
              - Dùng **Wilson CI** thay vì xấp xỉ chuẩn (Normal Approximation) vì có những nhóm rất nhỏ (ví dụ: giờ khuya, hoặc traffic đặc biệt). Xấp xỉ chuẩn có thể sinh ra CI < 0 hoặc > 1 (vô lý với tỷ lệ), trong khi Wilson CI đáng tin cậy hơn cho tỷ lệ mẫu nhỏ hoặc cực đoan.
            - Kỹ thuật: Groupby Aggregation, Proportion Confidence Interval (Wilson Score Interval).
            - Bằng chứng dẫn tới: Nhiều phân khúc (ví dụ: distance 8+ km) có lượng sample khá nhỏ, cần một thước đo chắc chắn để tránh kết luận sai lầm về nguyên nhân gây trễ.
            """
        ),
        code_cell(
            """
            delay_rate_by(df, "traffic_level").round(4)
            """
        ),
        code_cell(
            """
            delay_rate_by(df, "distance_band").round(4)
            """
        ),
        code_cell(
            """
            delay_rate_by(df, "complexity_band").round(4)
            """
        ),
        code_cell(
            """
            delay_rate_by(df, "order_hour").round(4)
            """
        ),
        md_cell("**Delay rate kèm khoảng tin cậy Wilson** (đọc kèm `orders` để không tin nhóm nhỏ):"),
        code_cell('pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "delay_rate_ci_by_distance_band.csv").round(4)'),
        code_cell('pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "delay_rate_ci_by_traffic_level.csv").round(4)'),
        md_cell(
            """
            **Insight (bảng CI).** Delay rate tăng đơn điệu theo distance band và
            theo mức traffic, và các CI **không chồng nhau** giữa nhóm thấp/cao →
            khác biệt là thật chứ không phải nhiễu mẫu. Nhóm nào `small_sample=True`
            thì đọc thận trọng.
            """
        ),
        img_cell("delay_rate_by_traffic.png"),
        img_cell("delay_rate_by_distance_band.png"),
        img_cell("delay_rate_by_complexity_band.png"),
        img_cell("correlation_heatmap.png"),
        md_cell(
            """
            **Insight.** Traffic, distance và một phần complexity làm thay đổi
            delay rate rõ hơn payment/weekend. Vì vậy các biến này được ưu tiên
            trong mô hình và công thức Risk Score. Heatmap tương quan cho thấy
            `distance_km` và `estimated_duration_min` gần như trùng nhau (chứng
            cứ redundancy), nên compact feature set chỉ giữ một biến đại diện.
            """
        ),
        md_cell(
            """
            ## 4. Món yêu thích và preference

            **Lý do làm.** DSS không chỉ dự báo trễ. Quản lý còn cần biết cơ cấu
            nhu cầu: loại pizza nào bán nhiều, size nào phổ biến, combo nào vừa
            phổ biến vừa rủi ro. Phần này phục vụ dashboard, Power BI và
            recommendation rule.
            """
        ),
        code_cell(
            """
            favorites = favorite_item_summary(df)
            favorites.query("dimension == 'pizza_type'").round(4)
            """
        ),
        code_cell(
            """
            favorites.query("dimension == 'pizza_size'").round(4)
            """
        ),
        code_cell(
            """
            favorites.query("dimension == 'type_size_combo'").head(15).round(4)
            """
        ),
        code_cell(
            """
            prefs = customer_preference_tables(df)
            prefs["type_size_matrix"]
            """
        ),
        md_cell(
            """
            **Insight.** Medium là size phổ biến nhất và có delay rate thấp hơn
            Large/XL. Non-Veg là type có số đơn cao nhất, còn Cheese Burst và
            một vài nhóm nhỏ có delay rate cao hơn. Vì nhiều nhóm nhỏ rất sparse,
            các tỷ lệ cực đoan chỉ dùng để cảnh báo/khám phá, không kết luận thị
            hiếu thật.
            """
        ),
        img_cell("top_pizza_types.png"),
        img_cell("pizza_type_size_heatmap.png"),
        md_cell("## 5. Phụ thuộc theo restaurant/brand"),
        code_cell(
            """
            restaurant_dependency_summary(df).round(4)
            """
        ),
        code_cell(
            """
            restaurant_mix_matrix(df, "pizza_type").round(4)
            """
        ),
        code_cell(
            """
            restaurant_mix_matrix(df, "pizza_size").round(4)
            """
        ),
        md_cell(
            """
            **Insight.** Brand không chỉ khác nhau ở count; mỗi restaurant có
            mix type/size, distance và traffic khác nhau. Vì vậy có thể phân
            tích brand cho dashboard, nhưng không kết luận "hãng tốt/xấu" nếu
            không có bằng chứng dữ liệu vận hành thật.
            """
        ),
        img_cell("restaurant_delay_rate.png"),
        img_cell("restaurant_pizza_type_mix_heatmap.png"),
        md_cell(
            """
            ## 6. Phụ thuộc theo location và state suy ra

            **Lý do làm.** User vận hành thường hỏi khu vực nào nhiều rủi ro.
            Dataset không có cột bang chính thức, nhưng `location` chứa dạng
            "City, ST", nên notebook suy ra `state_code` để phân tích mô tả.
            Những nhóm ít đơn được gắn cờ small sample.
            """
        ),
        code_cell(
            """
            location_dependency_summary(df).head(25).round(4)
            """
        ),
        code_cell(
            """
            state_dependency_summary(df).head(25).round(4)
            """
        ),
        code_cell(
            """
            state_mix_matrix(df, "pizza_type").head(20).round(4)
            """
        ),
        md_cell(
            """
            **Insight.** Một số state/city có delay rate cao nhưng count nhỏ.
            Vì vậy dashboard nên hiển thị cả `orders` lẫn `delay_rate`; không
            nên xếp hạng khu vực chỉ bằng tỷ lệ trễ khi mẫu quá ít.
            """
        ),
        img_cell("state_delay_rate_top15.png"),
        md_cell("## 7. Kiểm định giả thiết"),
        md_cell(
            """
            **Vì sao bước này?**
            - Làm gì: Tính toán hệ số Cramér's V và P-value (Chi-square test) để đo mức độ tương quan giữa các biến phân loại (traffic, distance, payment...) và nhãn `is_delayed`.
            - Vì sao: Để biết biến nào THỰC SỰ có ảnh hưởng (đóng góp vào việc sinh ra nhãn trễ) một cách có ý nghĩa thống kê, loại bỏ nhiễu ngẫu nhiên.
              - **Chi-square (p-value)** trả lời: "Có mối liên hệ hay không?".
              - **Cramér's V** trả lời: "Mối liên hệ đó mạnh cỡ nào?" (0 là không, 1 là hoàn hảo).
            - Kỹ thuật: Chi-square Test of Independence, Cramér's V.
            - Bằng chứng dẫn tới: Cần bằng chứng toán học (thay vì chỉ nhìn biểu đồ) để quyết định chọn features nào đưa vào mô hình Logistic Regression sau này.
            """
        ),
        code_cell(
            """
            hypothesis_tests(df).round(6)
            """
        ),
        md_cell(
            """
            **Cách đọc.** Chi-square kiểm tra biến categorical có độc lập với
            `is_delayed` hay không. `p < 0.05` = có bằng chứng liên hệ. Cột
            `cramers_v` cho biết **độ mạnh**: traffic_level cao nhất (liên hệ vừa/
            mạnh), kế đến pizza_type/payment/complexity; `payment_category` không
            đáng kể.

            **Insight.** p-value nhỏ + Cramér's V vừa của traffic/distance khớp với
            bảng delay-rate ở mục 3. Nhưng vì dữ liệu có artifact, đây là bằng chứng
            *liên hệ* để ưu tiên phân tích/dashboard, **không** phải quan hệ nhân quả.
            """
        ),
        md_cell("## 8. Clustering và association rules"),
        md_cell(
            """
            **Vì sao bước này?**
            - Làm gì: Chạy thuật toán K-Means để gom nhóm dữ liệu (Clustering), đo chất lượng bằng Silhouette score. Sau đó khai thác luật kết hợp (Association Rules).
            - Vì sao: 
              - **K-Means**: Dùng các feature không rò rỉ (distance_km, pizza_size_score, order_hour) để xem khách hàng tự nhiên chia thành các nhóm nào (ví dụ: nhóm chuyên gọi khuya, nhóm gọi gần...). Chọn K qua **Silhouette score** (đo lường độ "sắc nét" của cụm: trong cùng cụm thì giống nhau, giữa các cụm thì khác nhau).
              - **Association Rules**: Để tìm ra các tổ hợp điều kiện cùng xuất hiện dẫn đến rễ. Các ngưỡng binning (ví dụ binning distance_km thành `distance_band`) giúp quy dữ liệu liên tục về dạng rời rạc (itemset) để Apriori chạy được.
              - **Support**: Tần suất xuất hiện chung (A và B cùng xảy ra bao nhiêu lần).
              - **Confidence**: Xác suất trễ khi biết điều kiện A đã xảy ra.
              - **Lift**: Tỷ lệ giữa Confidence và tỷ lệ trễ chung (>1 nghĩa là điều kiện A làm TĂNG nguy cơ trễ).
            - Kỹ thuật: K-Means Clustering, Silhouette Analysis, Apriori Algorithm (Association Rules).
            - Bằng chứng dẫn tới: Mô hình supervised (LogReg) cho thấy distance và traffic quan trọng. Unsupervised learning (Clustering/Apriori) tái xác nhận điều này một cách độc lập khi nhóm có distance/traffic cao lại chính là nhóm có delay rate cao nhất.
            """
        ),
        code_cell('pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "kmeans_silhouette.csv").round(4)'),
        md_cell("**Profile từng cụm** (cụm *nghĩa là gì*, không chỉ silhouette):"),
        code_cell('pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "kmeans_cluster_profile.csv").round(4)'),
        md_cell(
            """
            **Insight (cụm).** Các cụm tách chủ yếu theo distance/traffic: cụm có
            `avg_distance_km` lớn và `dominant_traffic=High` chính là cụm
            `delay_rate` cao. Điều này khớp mô hình supervised và xác nhận distance/
            traffic là trục rủi ro chính, không phải brand hay payment.
            """
        ),
        code_cell(
            """
            association_rules_to_delay(df).head(15).round(4)
            """
        ),
        md_cell(
            """
            **Insight.** K-Means và association rules dùng để khám phá nhóm điều
            kiện thường đi với delayed. Chúng không thay thế mô hình supervised
            và không tự động sinh quyết định điều phối.
            """
        ),
        img_cell("kmeans_silhouette.png"),
        img_cell("association_lift_top.png"),
        md_cell("## 9. Insight dẫn đến quyết định DSS"),
        code_cell(
            """
            pd.DataFrame([
                {
                    "question": "Bao nhiêu phút thì nhãn chuyển sang trễ?",
                    "evidence": "On-time max=30, delayed min=35; threshold rules mismatch=0.",
                    "decision": "Chặn duration/delay khỏi feature; trình bày ngưỡng là suy luận từ nhãn.",
                },
                {
                    "question": "Yếu tố nào cần đưa vào Risk Score?",
                    "evidence": "Traffic, distance và complexity làm delay rate thay đổi rõ.",
                    "decision": "Dùng traffic/distance/peak/complexity trong Risk Score.",
                },
                {
                    "question": "Món nào được yêu thích?",
                    "evidence": "Medium nhiều nhất; Non-Veg nhiều nhất; combo type-size có chênh lệch lớn.",
                    "decision": "Đưa size/type mix vào dashboard, Power BI và recommendation rule.",
                },
                {
                    "question": "Có nên phân tích theo brand/location/state?",
                    "evidence": "Brand/location/state có mix và delay rate khác nhau nhưng nhiều artifact/small sample.",
                    "decision": "Báo cáo mô tả kèm caveat, không kết luận chất lượng thật của hãng/khu vực.",
                },
                {
                    "question": "Metric modeling nên chọn gì?",
                    "evidence": "Delayed là lớp thiểu số khoảng 21%.",
                    "decision": "Dùng F2/Recall cùng Balanced Accuracy/MCC, không dùng Accuracy đơn lẻ.",
                },
            ])
            """
        ),
        md_cell(
            """
            ## 10. Xác nhận artifact

            Toàn bộ CSV/figure đã được sinh **một lần** ở Bước 0 (`build_eda_artifacts`).
            Cell dưới chỉ liệt kê để xác nhận, không tính lại — tách bạch giữa *sinh
            artifact* (một nơi) và *trình bày* (các mục trên).
            """
        ),
        code_cell(
            """
            import os
            figs = sorted(f for f in os.listdir(PROJECT_ROOT / "reports" / "figures") if f.endswith(".png"))
            {"n_figures": len(figs), "examples": figs[:6]}
            """
        ),
    ]
    return nb


def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Saved executed notebook -> {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
