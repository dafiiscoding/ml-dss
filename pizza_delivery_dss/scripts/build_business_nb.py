import textwrap
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_business_forecasting_recommendation.ipynb"


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
            # Module 05 - Customer Behavior, Demand Forecasting, and Recommendation

            Notebook này mở rộng phần phân tích dữ liệu rác/synthetic thành nội
            dung học thuật: audit tính giả lập, kiểm định giả thiết, xu hướng
            nhu cầu, staffing theo giờ, preference size/type/restaurant/location,
            recommendation rule và sales-volume reporting.
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
            from pizza_dss.business_analysis import (
                build_business_artifacts,
                customer_preference_tables,
                forecast_method_comparison,
                forecast_metrics,
                forecast_monthly_demand,
                hourly_staffing_plan,
                hypothesis_tests,
                monthly_demand,
                monthly_product_trends,
                preference_trend_tests,
                recommendation_rules,
                redundant_feature_audit,
                synthetic_data_audit,
            )
            from pizza_dss.data_loader import load_dataset
            from IPython.display import Image
            """
        ),
        md_cell("Sinh trước hình business để hiển thị inline."),
        code_cell("_ = build_business_artifacts()"),
        md_cell(
            """
            ## 0. Định nghĩa thuật ngữ business

            - **MAE**: sai số tuyệt đối trung bình (đơn vị: số đơn). **MAPE**: sai
              số phần trăm tuyệt đối trung bình; >40% = forecast yếu.
            - **Seasonal-naive**: dự báo tháng t bằng giá trị cùng kỳ năm trước.
              **Moving average**: trung bình vài tháng gần nhất.
            - **Mann-Kendall**: kiểm định xu hướng phi tham số; `tau` là cường độ xu
              hướng, `p < 0.05` mới kết luận có trend (tăng/giảm).
            - **Confidence/Support** của recommendation rule: P(gợi ý | bối cảnh) và
              tần suất bối cảnh.
            - **order_share**: tỷ trọng đơn của một size/type trên tổng.
            """
        ),
        md_cell("## 1. Data realism / synthetic audit"),
        code_cell(
            """
            df = load_dataset()
            synthetic_data_audit(df)
            """
        ),
        code_cell(
            """
            redundant_feature_audit(df)
            """
        ),
        md_cell(
            """
            Kết luận: dữ liệu có nhiều cột deterministic. Vì vậy dự án không cố
            che điểm yếu này mà đưa nó vào audit, đồng thời tạo feature set
            compact để so sánh với feature set đầy đủ.
            """
        ),
        img_cell("synthetic_data_flags.png"),
        img_cell("complexity_distribution.png"),
        md_cell("## 2. Hypothesis tests"),
        code_cell(
            """
            hypothesis_tests(df)
            """
        ),
        md_cell(
            """
            Kiểm định chi-square trả lời các câu hỏi kiểu: payment method,
            pizza size, traffic, restaurant, location top-12 có độc lập với
            trạng thái delayed hay không. Với dữ liệu synthetic, p-value chỉ là
            minh chứng phân tích, không được diễn giải như bằng chứng nhân quả.
            """
        ),
        md_cell("## 3. Customer preferences"),
        code_cell(
            """
            prefs = customer_preference_tables(df)
            prefs["size_mix"]
            """
        ),
        code_cell(
            """
            prefs["type_mix"].head(12)
            """
        ),
        code_cell(
            """
            prefs["top_restaurant_by_type"]
            """
        ),
        code_cell(
            """
            prefs["location_summary"].head(15)
            """
        ),
        img_cell("size_preference.png"),
        md_cell("## 4. Time series demand and staffing"),
        code_cell(
            """
            monthly_demand(df).tail(15)
            """
        ),
        code_cell(
            """
            forecast = forecast_monthly_demand(df, horizon=6)
            forecast.tail(12)
            """
        ),
        code_cell(
            """
            forecast_metrics(forecast)
            """
        ),
        md_cell("**So sánh phương pháp forecast** (backtest seasonal-naive vs moving-average):"),
        md_cell(
            """
            **Vì sao bước này?**
            - Làm gì: Chạy và so sánh độ lỗi (MAE, MAPE) giữa 2 phương pháp dự báo cơ bản (Baseline Forecast): Seasonal-Naive (lấy cùng kỳ năm ngoái) và Moving-Average (lấy trung bình các tháng gần đây).
            - Vì sao: Do dữ liệu được sinh random nên dự báo chuỗi thời gian khó có quy luật dài hạn (trend/seasonality) ổn định. Chọn Seasonal-Naive vì tính đặc thù ngành hàng (nhu cầu F&B lặp lại theo mùa/tháng) thay vì dùng mô hình phức tạp (như ARIMA, Prophet) dễ bị overfit trên dữ liệu ngắn và nhiễu (vài tháng). Việc so sánh MAPE cho thấy dự báo bằng mọi giá trị lịch sử ở đây mang độ sai số rất lớn (>40%).
            - Kỹ thuật: Time Series Backtesting, Seasonal Naive Forecast, Mean Absolute Percentage Error (MAPE).
            - Bằng chứng dẫn tới: Dataset gồm chưa tới 3 năm (thiếu dữ liệu để fit mô hình chuỗi thời gian xịn), tháng cuối bị cắt cụt (partial month). So sánh MAPE chứng tỏ tín hiệu xu hướng trong dataset khá yếu.
            """
        ),
        code_cell('pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "forecast_method_comparison.csv")'),
        md_cell(
            """
            **Insight (forecast).** Cả hai phương pháp đều có MAPE cao trên dữ liệu
            nhỏ/synthetic + tháng partial; không phương pháp nào thắng rõ. Kết luận:
            forecast chỉ là **demo quy trình planning**, không cam kết độ chính xác
            sản xuất. Báo cáo phải nêu giới hạn này thay vì khoe MAPE.
            """
        ),
        code_cell(
            """
            hourly_staffing_plan(df).query("orders > 0").sort_values("scenario_orders_per_day", ascending=False)
            """
        ),
        img_cell("monthly_demand_forecast.png"),
        img_cell("hourly_staffing_plan.png"),
        md_cell("## 5. Trend by product and recommendation"),
        code_cell(
            """
            type_trend, size_trend = monthly_product_trends(df)
            type_trend.sort_values(["order_period", "orders"], ascending=[False, False]).head(20)
            """
        ),
        md_cell("**Kiểm định xu hướng sở thích (Mann-Kendall)** — có trend thật về sau không?"),
        md_cell(
            """
            **Vì sao bước này?**
            - Làm gì: Sử dụng phép thử phi tham số Mann-Kendall để kiểm định xem tỷ trọng từng phân khúc (ví dụ: share của Pizza Size Medium) có xu hướng tăng hay giảm thực sự theo thời gian hay không.
            - Vì sao: Thay vì nhìn biểu đồ lên xuống thất thường (vẽ bằng mắt thường dễ bị thiên lệch), Mann-Kendall trả lời chắc chắn bằng p-value rằng "sự lên xuống này là nhiễu ngẫu nhiên, hay đang có trend dài hạn?". Phép thử này miễn nhiễm với các ngoại lệ (outliers) hoặc dữ liệu không phân phối chuẩn.
            - Kỹ thuật: Mann-Kendall Trend Test (Non-parametric test).
            - Bằng chứng dẫn tới: Đa số các loại pizza/size báo cáo `trend = no_trend` (p-value >= 0.05), xác nhận bản chất dữ liệu không có trend, giải thích vì sao forecast có sai số cao.
            """
        ),
        code_cell('pd.read_csv(PROJECT_ROOT / "reports" / "metrics" / "preference_trend_tests.csv")'),
        md_cell(
            """
            **Insight (trend).** Hầu hết category có `trend = no_trend` (p ≥ 0.05):
            share size/type **không** có xu hướng tăng/giảm có ý nghĩa. Đây là câu
            trả lời trung thực cho "dự đoán trend về sau?": *forecast được về mặt cơ
            học, nhưng dữ liệu không cho thấy trend thực để dự báo* — đúng bản chất
            dữ liệu random.
            """
        ),
        code_cell(
            """
            recommendation_rules(df).head(20)
            """
        ),
        img_cell("preference_size_share_forecast.png"),
        img_cell("preference_type_share_forecast.png"),
        md_cell("## 6. Insight → quyết định"),
        code_cell(
            """
            pd.DataFrame([
                {
                    "module": "Customer preference",
                    "insight": "Medium và Non-Veg có order share cao nhất; một số combo type-size có delay rate cao.",
                    "decision": "Dùng trong dashboard mix sản phẩm và rule recommendation, không dùng làm kết luận thị trường thật.",
                },
                {
                    "module": "Demand forecast",
                    "insight": "Seasonal-naive forecast có MAPE cao do dữ liệu nhỏ/synthetic và tháng partial.",
                    "decision": "Chỉ dùng forecast như minh họa staffing/planning, không cam kết dự báo sản xuất.",
                },
                {
                    "module": "Staffing",
                    "insight": "Peak order hour là 19h trong kịch bản hiện tại.",
                    "decision": "Tạo staffing scenario 100 đơn/ngày để minh họa phân bổ nhân sự.",
                },
                {
                    "module": "Recommendation",
                    "insight": "Rule dựa trên popularity/context dễ giải thích nhưng chưa cá nhân hóa.",
                    "decision": "Dùng làm lớp gợi ý minh bạch trong DSS, không coi là recommender production.",
                },
            ])
            """
        ),
        md_cell(
            """
            ## 7. Xác nhận artifact

            CSV/figure business đã sinh **một lần** ở Bước 0 (`build_business_artifacts`).
            Cell dưới chỉ liệt kê, không tính lại.
            """
        ),
        code_cell(
            """
            import os
            metrics = sorted(f for f in os.listdir(PROJECT_ROOT / "reports" / "metrics") if f.endswith((".csv", ".json")))
            {"n_metric_files": len(metrics), "examples": metrics[:8]}
            """
        ),
    ]
    return nb


def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    ).execute()
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Saved executed notebook -> {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
