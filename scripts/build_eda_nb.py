"""Build and execute the leakage-safe EDA evidence notebook."""
import os

import nbformat as nbf
from nbclient import NotebookClient

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "notebooks", "02_eda.ipynb")

nb = nbf.v4.new_notebook()
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    cells.append(nbf.v4.new_code_cell(source))


md("""# 02 - Phân tích khám phá dữ liệu CrisisMMD

Notebook này trình bày phân tích khám phá có thể tái lập. EDA phục vụ lựa chọn
phương pháp chỉ dùng **13.608 mẫu train**; dev/test được giữ ngoài quá trình khám phá. Kiểm kê
chất lượng file vẫn bao phủ toàn bộ corpus vì đây là kiểm tra dữ liệu đầu vào,
không phải lựa chọn mô hình.

Kỹ thuật: thống kê mô tả, phân tích theo sự kiện, Text Mining, K-Means,
Apriori, kiểm tra ảnh và phân tích mâu thuẫn đa phương thức.""")

code("""import json, os, sys
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display, Image as NotebookImage
%matplotlib inline
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110})

from src.data_loader import load_dataset
from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
import src.eda_analysis as eda

train_df, val_df, test_df, IMG_BASE = load_dataset(use_sample_if_missing=False)
df = eda.add_text_features(train_df)
METRICS = os.path.join(REPORTS_DIR, "metrics")
print(f"EDA train-only: {len(df):,} | dev giữ lại: {len(val_df):,} | test giữ lại: {len(test_df):,}")""")

md("## 1. Chất lượng dữ liệu và tính toàn vẹn split")
code("""with open(os.path.join(METRICS, "data_quality_summary.json"), encoding="utf-8") as f:
    quality = json.load(f)
display(pd.DataFrame(quality["splits"]).T)
print("Sai lệch nếu suy diễn nhãn informative từ humanitarian:",
      quality["protocol"]["old_derived_informative_mismatches"])
print("Dev/test leakage-safe:",
      quality["splits"]["val"]["evaluation_eligible_rows"],
      quality["splits"]["test"]["evaluation_eligible_rows"])
print("Ảnh hợp lệ / file trên đĩa:",
      quality["images"]["valid_images"], "/", quality["images"]["files_on_disk"])
print("Hash ảnh duy nhất:", quality["images"]["referenced_unique_hashes"])
print("Embedding integrity:")
display(pd.DataFrame(quality["embeddings"]).T)""")

md("""**Kết luận chất lượng.** Không thiếu ảnh được tham chiếu, không trùng
`(tweet_id, image_id)` và không giao nhau theo ID. Tuy nhiên có ảnh giống hệt
nhau theo SHA-256 giữa các split. Vì vậy model được chọn/đánh giá trên mask
leakage-safe: 2.189 mẫu dev và 2.169 mẫu test.""")

md("## 2. Phân bố nhãn và mất cân bằng")
code("""inf_dist = eda.label_distribution(df, "label")
cat_dist = eda.label_distribution(df, "label_top")
display(inf_dist)
display(cat_dist)
ratio = eda.imbalance_ratio(df, "label_top")
print(f"Tỉ lệ lớp lớn nhất/nhỏ nhất trên train: {ratio}:1")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(ax=axes[0], x="label", data=df,
              order=df["label"].value_counts().index, hue="label", legend=False)
axes[0].set_title("Nhãn informative - train")
order = df["label_top"].value_counts().index
sns.countplot(ax=axes[1], y="label_top", data=df, order=order,
              hue="label_top", legend=False)
axes[1].set_xscale("log")
axes[1].set_title("Tám lớp humanitarian - train (log scale)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "label_distribution.png"), bbox_inches="tight")
plt.show()""")

md("""**Insight 1.** Lớp `not_humanitarian` có 5.260 mẫu, trong khi
`missing_or_found_people` chỉ có 24 mẫu, chênh khoảng **219:1**. Vì vậy Accuracy
không đủ; phần mô hình dùng class weight, F2 cho informative và Macro-F1 cho
tám lớp.""")

md("## 3. Phân tích theo sự kiện")
code("""event_counts = df["event_name"].value_counts()
event_profile = pd.crosstab(df["event_name"], df["label_top"], normalize="index")
display(event_counts.to_frame("rows"))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.barplot(ax=axes[0], x=event_counts.values, y=event_counts.index,
            hue=event_counts.index, legend=False)
axes[0].set_title("Số mẫu train theo sự kiện")
sns.heatmap(event_profile, ax=axes[1], cmap="YlOrRd",
            cbar_kws={"label": "tỉ lệ trong sự kiện"})
axes[1].set_title("Cơ cấu humanitarian theo sự kiện")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "event_distribution.png"), bbox_inches="tight")
plt.show()
display(event_profile.round(3))""")

md("""**Insight 2.** Phân bố thay đổi đáng kể theo sự kiện: Sri Lanka floods có
69,3% `not_humanitarian`, còn Iraq--Iran earthquake chỉ 22,8%; Mexico earthquake
có 27,9% rescue/donation. Đây là dấu hiệu domain shift theo sự kiện. Hệ thống
không tự đổi ngưỡng theo sự kiện ở phiên bản hiện tại, nhưng dashboard phải cho
phép lọc/giám sát theo event và việc triển khai mới cần hiệu chỉnh lại.""")

md("## 4. Độ dài và từ khóa văn bản")
code("""print(df.groupby("label")[["char_length", "word_count"]].mean().round(2))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(ax=axes[0], data=df, x="word_count", hue="label",
             bins=30, element="step", stat="density", common_norm=False)
axes[0].set_title("Số từ theo informative")
top_all = eda.top_tokens(df[df["label"] == "informative"]["tweet_text"], top_n=20)
sns.barplot(ax=axes[1], data=top_all, x="frequency", y="token",
            hue="token", legend=False)
axes[1].set_title("Top token trong tweet informative")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "text_length_distribution.png"), bbox_inches="tight")
plt.savefig(os.path.join(FIGURES_DIR, "top_keywords.png"), bbox_inches="tight")
plt.show()""")

md("""**Insight 3.** Số từ trung bình của informative và not-informative gần như
giống nhau (11,75 so với 11,68), nên độ dài không phải tín hiệu phân loại đủ
mạnh. Nội dung từ/ngữ và ngữ cảnh mới là đầu vào phù hợp cho TF-IDF.""")

code("""from wordcloud import WordCloud
text_blob = " ".join(
    eda.clean_tweet_text(text)
    for text in df[df["label"] == "informative"]["tweet_text"]
)
wc = WordCloud(
    width=1000, height=400, background_color="white",
    stopwords=set(eda.CLUSTER_STOPWORDS), collocations=False
).generate(text_blob)
plt.figure(figsize=(13, 5))
plt.imshow(wc); plt.axis("off"); plt.title("WordCloud - informative train")
plt.savefig(os.path.join(FIGURES_DIR, "wordcloud.png"), bbox_inches="tight")
plt.show()""")

md("## 5. K-Means: kiểm tra nhiều k và đối chiếu taxonomy")
code("""sweep = pd.read_csv(os.path.join(METRICS, "kmeans_silhouette_by_k.csv"))
display(sweep)
plt.figure(figsize=(8, 4))
sns.lineplot(data=sweep, x="k", y="silhouette", marker="o")
plt.title("Silhouette theo k trên TF-IDF train")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "kmeans_silhouette_sweep.png"), bbox_inches="tight")
plt.show()

clus = eda.kmeans_topic_clustering(df, n_clusters=8, sample_size=8000)
for cluster, terms in clus["cluster_terms"].items():
    print(f"Cluster {cluster}: {', '.join(terms[:8])}")
plt.figure(figsize=(11, 5))
sns.heatmap(clus["crosstab"], cmap="Blues")
plt.title("K-Means k=8 x nhãn humanitarian")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "kmeans_clusters.png"), bbox_inches="tight")
plt.show()""")

md("""**Insight 4.** Silhouette chỉ khoảng 0,014--0,024 cho k=2..12 và không có
điểm gãy rõ. `k=8` được giữ như một phân tích đối chiếu với taxonomy tám lớp,
không được tuyên bố là số cụm tối ưu. Clustering cho thấy chủ đề chồng lấn, nên
không dùng cụm để routing; routing dựa trên phân loại có giám sát.""")

md("## 6. Apriori trên keyword/hashtag")
code("""itemsets, rules = eda.association_rules_analysis(df)
display(rules.head(20))
print("Các transaction loại bỏ cặp trùng nghĩa kiểu #earthquake/earthquake để tránh luật hiển nhiên.")""")

md("""**Insight 5.** Các luật còn lại phản ánh liên kết địa danh--sự kiện thật,
ví dụ `#lka` với `#floodsl` (lift khoảng 40) và `#iran` với `#earthquake`
(lift khoảng 18,5). Chúng hữu ích làm bộ lọc drill-down theo sự kiện, nhưng
không thay thế classifier vì chủ yếu mô tả context.""")

md("## 7. Mâu thuẫn text--ảnh")
code("""mc = eda.multimodal_conflict_analysis(df)
display(mc["overall_pct"].to_frame("percent"))
display(mc["by_category"])
bc = mc["by_category"].sort_values("disagree_pct")
plt.figure(figsize=(10, 5))
sns.barplot(x=bc["disagree_pct"], y=bc.index,
            hue=bc.index, legend=False)
plt.xlabel("% cặp có nhãn category text/image khác nhau")
plt.title("Mâu thuẫn đa phương thức trên train")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "multimodal_conflict.png"), bbox_inches="tight")
plt.show()""")

md("""**Insight 6.** 55,0% mẫu train có nhãn category text và ảnh khác nhau;
tỉ lệ ở missing/injured/affected lần lượt khoảng 91,7%/91,0%/88,9%. Đây không
chứng minh phương thức nào đúng hơn. Nó chứng minh cần tune fusion trên ground
truth chung và dành một hàng đợi Manual Review có giới hạn năng lực.""")

md("## 8. EDA ảnh: gallery, kích thước và không gian CLIP")
code("""display(NotebookImage(filename=os.path.join(FIGURES_DIR, "image_gallery.png")))
display(pd.DataFrame({
    "metric": [
        "valid images", "invalid metadata files", "unique image hashes",
        "duplicate extra rows", "dev eligible", "test eligible"
    ],
    "value": [
        quality["images"]["valid_images"], quality["images"]["invalid_files"],
        quality["images"]["referenced_unique_hashes"],
        quality["images"]["referenced_duplicate_extra_rows"],
        quality["splits"]["val"]["evaluation_eligible_rows"],
        quality["splits"]["test"]["evaluation_eligible_rows"],
    ],
}))""")

code("""from sklearn.manifold import TSNE
E = np.load(os.path.join(MODELS_DIR, "X_train_img_emb.npy"))
meta = pd.read_csv(os.path.join(MODELS_DIR, "img_train_meta.csv"))
rng = np.random.default_rng(42)
idx = rng.choice(len(E), size=min(3000, len(E)), replace=False)
Z = TSNE(
    n_components=2, init="pca", perplexity=30, random_state=42
).fit_transform(E[idx])
plot_df = pd.DataFrame({
    "x": Z[:, 0], "y": Z[:, 1],
    "category": meta.iloc[idx]["label_top"].to_numpy(),
})
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=plot_df, x="x", y="y", hue="category",
    s=12, palette="tab10", linewidth=0
)
plt.title("t-SNE CLIP embedding train theo ground truth chung")
plt.legend(bbox_to_anchor=(1.02, 1), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "tsne_clip.png"), bbox_inches="tight")
plt.show()""")

md("""**Kết luận EDA ảnh.** Ảnh thật đa dạng kích thước/mode và embedding CLIP
không rỗng, chuẩn hóa L2. Không gian t-SNE có vùng cục bộ nhưng các lớp chồng
lấn; do đó CLIP là đặc trưng phụ trợ, hiệu quả cuối phải được xác nhận bằng
dev/test thay vì suy luận từ hình chiếu 2D.""")

md("""## 9. Chuỗi suy luận EDA -> quyết định

| Bằng chứng train | Suy luận hợp lệ | Quyết định thiết kế |
|---|---|---|
| Mất cân bằng 219:1 | Accuracy che khuất lớp hiếm | F2, Macro-F1, class weight |
| Event profile thay đổi | Có domain shift theo sự kiện | Bộ lọc event và tái hiệu chỉnh khi triển khai |
| Độ dài hai lớp gần nhau | Length không đủ phân loại | TF-IDF unigram/bigram |
| Silhouette thấp, không có elbow | Cụm không đủ cho routing | Dùng clustering cho khám phá, classifier cho quyết định |
| Luật hashtag/địa danh lift cao | Hashtag mô tả context sự kiện | Drill-down phụ trợ |
| 55% nhãn text/ảnh khác nhau | Không thể tin tuyệt đối một modality | Tune late fusion và Manual Review |
| Hash ảnh trùng xuyên split | Metric ảnh có nguy cơ lạc quan | Lọc dev/test theo manifest leakage-safe |""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}

print("Executing 02_eda ...")
NotebookClient(
    nb,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": REPO}},
).execute()
with open(OUT, "w", encoding="utf-8") as stream:
    nbf.write(nb, stream)
print(f"Saved -> {OUT}")
