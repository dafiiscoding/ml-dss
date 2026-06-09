"""Build and execute the model/fusion/DSS evidence notebook."""
import os

import nbformat as nbf
from nbclient import NotebookClient

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "notebooks", "03_modeling.ipynb")

nb = nbf.v4.new_notebook()
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    cells.append(nbf.v4.new_code_cell(source))


md("""# 03 - So sánh mô hình, Late Fusion và tầng quyết định DSS

Protocol:

1. Dùng nhãn informative chính thức, join theo `(tweet_id, image_id)`.
2. Fit sáu model trên train.
3. Chọn model, trọng số fusion và threshold trên dev leakage-safe.
4. Báo cáo một lần trên test leakage-safe.
5. Text/image/fusion dùng cùng ground truth chung.
6. Luôn đối chiếu dummy baseline; không diễn giải F2 riêng lẻ.
7. Robust mask và bootstrap chỉ là hậu kiểm với cấu hình đã khóa, không tune lại.""")

code("""import json, os, sys
sys.path.insert(0, os.path.abspath(".."))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.dashboard_data import load_dashboard_data

METRICS = os.path.join(REPORTS_DIR, "metrics")
with open(os.path.join(MODELS_DIR, "fusion_config.json"), encoding="utf-8") as f:
    fusion_config = json.load(f)
with open(os.path.join(METRICS, "data_quality_summary.json"), encoding="utf-8") as f:
    quality = json.load(f)
print("Evaluation rows:", fusion_config["evaluation_rows"])
print("Old derived-label mismatches:",
      quality["protocol"]["old_derived_informative_mismatches"])
fusion_config""")

md("## 1. Sáu classifier trên dev")
code("""files = {
    "Text - informative": "text_informative_validation.csv",
    "Text - humanitarian": "text_humanitarian_validation.csv",
    "Image - informative": "image_informative_validation.csv",
    "Image - humanitarian": "image_humanitarian_validation.csv",
}
tables = {}
for title, filename in files.items():
    table = pd.read_csv(os.path.join(METRICS, filename))
    tables[title] = table
    print("\\n", title)
    display(table)""")

md("""**Lựa chọn trên dev.**

- Text informative: Linear SVM theo F2.
- Text humanitarian: Logistic Regression theo Macro-F1.
- Image informative: k-NN theo F2.
- Image humanitarian: Logistic Regression theo Macro-F1.

k-NN rất yếu trên TF-IDF thưa nhưng cạnh tranh trên embedding CLIP; đây là ví
dụ cùng thuật toán có hành vi khác khi biểu diễn đặc trưng thay đổi.""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4))
tables["Text - informative"].set_index("Model")["F2"].sort_values().plot.barh(
    ax=axes[0], color="#22577a"
)
axes[0].set_title("Text informative - F2 dev")
tables["Image - humanitarian"].set_index("Model")["Macro F1"].sort_values().plot.barh(
    ax=axes[1], color="#c0392b"
)
axes[1].set_title("Image humanitarian - Macro-F1 dev")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "model_validation_comparison.png"), bbox_inches="tight")
plt.show()""")

md("## 2. Dummy baseline và kết quả cuối trên test leakage-safe")
code("""binary = pd.read_csv(os.path.join(METRICS, "fusion_informative_test.csv"))
category = pd.read_csv(os.path.join(METRICS, "fusion_humanitarian_test.csv"))
review = pd.read_csv(os.path.join(METRICS, "manual_review_test.csv"))
dummy_binary = pd.read_csv(
    os.path.join(METRICS, "baseline_informative_test.csv")
)
dummy_category = pd.read_csv(
    os.path.join(METRICS, "baseline_humanitarian_test.csv")
)
print("Dummy informative baselines")
display(dummy_binary)
print("Dummy humanitarian baseline")
display(dummy_category)
print("Final systems")
display(binary)
display(category)
display(review)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
informative_plot = pd.concat([
    dummy_binary[dummy_binary["Model"] == "Always informative"][
        ["Model", "Accuracy", "F1", "F2"]
    ],
    binary[["Model", "Accuracy", "F1", "F2"]],
])
informative_plot.set_index("Model").plot.bar(ax=axes[0], rot=15)
axes[0].set_title("Informative - test sạch")
category_plot = pd.concat([
    dummy_category[["Model", "Macro F1", "Weighted F1"]],
    category[["Model", "Macro F1", "Weighted F1"]],
])
category_plot.set_index("Model").plot.bar(
    ax=axes[1], rot=0
)
axes[1].set_title("Humanitarian - test sạch")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fusion_test_comparison.png"), bbox_inches="tight")
plt.show()""")

md("""**Diễn giải có đối chứng.**

- Dummy luôn dự báo `informative` đã đạt F2 = 0,8939 vì positive rate là
  62,75%. Late Fusion đạt 0,9035, chỉ hơn 0,0096 F2; không được dùng F2 riêng
  làm bằng chứng hệ thống mạnh.
- Giá trị của informative fusion rõ hơn ở Accuracy 0,7072 so với 0,6275, F1
  0,8079 so với 0,7711, Balanced Accuracy 0,6136 và MCC 0,3602.
- Với humanitarian, majority baseline Macro-F1 chỉ 0,0686 trong khi fusion đạt
  0,4005. Đây là bằng chứng mạnh hơn cho giá trị phân loại tám lớp.
- Ảnh tạo mức cải thiện rõ hơn ở nhiệm vụ tám lớp; text đóng vai trò lớn hơn
  trong informative với trọng số dev-tuned 0,70/0,30.""")

md("## 3. Phân tích theo lớp và lỗi")
code("""class_report = pd.read_csv(
    os.path.join(METRICS, "fusion_humanitarian_classification_report.csv"),
    index_col=0,
)
display(class_report.iloc[:8])

predictions = load_dashboard_data()
safe = predictions[predictions["evaluation_eligible"].astype(bool)].copy()
labels = list(class_report.index[:8])
cm = confusion_matrix(
    safe["true_category"], safe["fused_category"], labels=labels
)
short = [label.replace("_", " ")[:18] for label in labels]
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=short, yticklabels=short)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Late-fusion confusion matrix - 2.169 test rows")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_humanitarian.png"), bbox_inches="tight")
plt.show()""")

md("""**Giới hạn quan trọng.** `missing_or_found_people` chỉ có 5 mẫu test sạch
và F1 = 0,0714; `vehicle_damage` có 18 mẫu và F1 = 0,2169. Macro-F1 phản ánh
đúng điểm yếu này. Hệ thống chưa đủ bằng chứng để tự động hóa quyết định sinh
tử cho lớp hiếm; các case đó cần người xác minh.""")

md("## 4. Robustness: near-duplicate, bootstrap, event và class")
code("""robust_comparison = pd.read_csv(
    os.path.join(METRICS, "robustness_metric_comparison.csv")
)
bootstrap = pd.read_csv(
    os.path.join(METRICS, "robust_bootstrap_intervals.csv")
)
event_stability = pd.read_csv(
    os.path.join(METRICS, "robust_event_stability.csv")
)
class_stability = pd.read_csv(
    os.path.join(METRICS, "robust_class_stability.csv")
)

core = robust_comparison[
    robust_comparison["Task"].isin(
        ["Informative Fusion", "Humanitarian Fusion", "Manual Review"]
    )
]
display(core)
display(bootstrap)
display(event_stability[[
    "Event", "Rows", "Informative F2",
    "Humanitarian Macro F1 (present classes)", "Review F1"
]])
display(class_stability[[
    "Class", "Canonical F1", "Robust F1", "F1 Delta",
    "Robust Support", "Stability Flag"
]])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
event_stability.sort_values(
    "Humanitarian Macro F1 (present classes)"
).plot.barh(
    x="Event",
    y="Humanitarian Macro F1 (present classes)",
    legend=False,
    color="#c0392b",
    ax=axes[0],
)
axes[0].set_xlim(0, 1)
axes[0].set_title("Robust test theo event")
class_stability.plot.barh(
    x="Class",
    y=["Canonical F1", "Robust F1"],
    ax=axes[1],
)
axes[1].set_xlim(0, 1)
axes[1].set_title("F1 theo lớp: canonical vs robust")
plt.tight_layout()
plt.show()""")

md("""**Diễn giải hậu kiểm.**

- Sau khi loại thêm 137 test rows near-duplicate đã xác minh, informative F2
  giảm từ 0,9035 xuống 0,9006; humanitarian Macro-F1 giảm từ 0,4005 xuống
  0,3908. Cấu hình không được tune lại.
- Trên 2.000 bootstrap phân tầng, F2 robust có CI 95% [0,8938; 0,9068].
  Paired F2 gain so với always-informative là 0,0100, CI [0,0032; 0,0161]:
  khác 0 theo giả định row-bootstrap nhưng độ lớn thực tiễn nhỏ.
- Humanitarian Macro-F1 robust có CI [0,3636; 0,4208], gain so với majority
  baseline có CI [0,2936; 0,3508], là bằng chứng mạnh hơn.
- Theo event, informative F2 thấp nhất ở Sri Lanka floods (0,8377);
  humanitarian Macro-F1 thấp nhất ở Iraq-Iran earthquake (0,3458, 61 rows).
  Đây là tín hiệu domain shift, không phải ước lượng deployment chắc chắn.
- Ba lớp injured/dead, missing/found và vehicle damage vẫn có support dưới 30.
  Không có lớp support đủ lớn nào dịch F1 quá 0,05 giữa hai mask.""")

md("## 5. Manual Review")
code("""print("Conflict threshold:", fusion_config["manual_review"]["conflict_threshold"])
print("Dev capacity:", fusion_config["manual_review"]["max_review_rate_on_dev"])
display(review)""")

md("""Ngưỡng conflict được chọn trên dev với capacity 25%. Trên test, review
rate là 25,63%, precision 0,7536 và recall 0,3401. Đây là trade-off vận hành:
hàng đợi tập trung vào xung đột mạnh, không cố thu hồi toàn bộ xung đột.""")

md("## 6. Kiểm tra logic Risk Score và routing")
code("""scenarios = pd.read_csv(os.path.join(METRICS, "dss_scenarios.csv"))
sensitivity = pd.read_csv(
    os.path.join(METRICS, "dss_threshold_sensitivity.csv")
)
display(scenarios[[
    "scenario", "risk_score", "base_priority", "priority",
    "assigned_team", "manual_review"
]])
display(sensitivity)""")

md("""Risk Score không có nhãn ground truth trong CrisisMMD, nên trọng số và
ngưỡng Priority là **policy minh bạch**, không được tuyên bố là tối ưu thống kê.
Scenario test xác nhận: post thường có score thấp; case sinh tử cao; conflict
High được review song song với Emergency Team và không bị hạ xuống Medium.
Bảng sensitivity cho thấy số case High thay đổi mạnh khi policy đổi, do đó
ngưỡng phải được trung tâm vận hành phê duyệt.""")

md("""## 7. Kết luận nghiệm thu

- Mô hình: có so sánh đủ sáu thuật toán ở bốn nhánh/task.
- Evaluation: tách train/dev/test và loại duplicate hash xuyên split.
- Fusion: trọng số và threshold chọn trên dev, test chỉ dùng báo cáo.
- Baseline: có dummy đối chứng; F2 informative không được diễn giải độc lập.
- Robustness: có mask review thủ công, sensitivity, bootstrap, event/class stability.
- Error analysis: có confusion matrix và per-class metrics.
- DSS: có scenario test, routing đủ tám lớp, sensitivity và human-in-the-loop.""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}

print("Executing 03_modeling ...")
NotebookClient(
    nb,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": REPO}},
).execute()
with open(OUT, "w", encoding="utf-8") as stream:
    nbf.write(nb, stream)
print(f"Saved -> {OUT}")
