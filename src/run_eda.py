import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_dataset
from src.text_preprocessing import clean_tweet_text
from src.config import FIGURES_DIR

def generate_eda():
    print("Running EDA and generating figures...")

    # Load dataset
    train_df, val_df, test_df, _ = load_dataset()
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Set seaborn style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    # 1. Label Distribution (Informative vs Non-informative)
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x="label", data=df, palette="viridis")
    plt.title("Distribution of Informativeness Labels")
    plt.xlabel("Label")
    plt.ylabel("Count")
    # Add count values on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "label_distribution.png"), dpi=150)
    plt.close()

    # 2. Humanitarian Category Distribution
    plt.figure(figsize=(12, 6))
    order = df["label_top"].value_counts().index
    ax = sns.countplot(y="label_top", data=df, order=order, palette="magma")
    plt.title("Distribution of Humanitarian Categories")
    plt.xlabel("Count")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "category_distribution.png"), dpi=150)
    plt.close()

    # 3. Distribution of Events
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x="event_name", data=df, palette="coolwarm")
    plt.title("Distribution of Disaster Events")
    plt.xlabel("Event Name")
    plt.ylabel("Count")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "event_distribution.png"), dpi=150)
    plt.close()

    # 4. Text Length Distribution by Label
    df["text_length"] = df["tweet_text"].apply(lambda x: len(str(x)))
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="text_length", hue="label", kde=True, multiple="stack", palette="crest")
    plt.title("Tweet Text Length Distribution by Label")
    plt.xlabel("Text Length (Characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "text_length_distribution.png"), dpi=150)
    plt.close()

    # 5. Top keywords in high-priority posts
    # Let's count some words
    from collections import Counter
    import re

    informative_tweets = df[df["label"] == "informative"]["tweet_text"]
    all_words = []
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'for', 'in', 'on', 'at', 'by', 'of', 'with', 'from', 'this', 'that', 'it', 'i', 'you', 'we', 'they', 'our', 'my'}

    for tweet in informative_tweets:
        cleaned = clean_tweet_text(tweet)
        words = [w for w in cleaned.split() if w not in stopwords and len(w) > 2]
        all_words.extend(words)

    word_counts = Counter(all_words).most_common(15)
    if word_counts:
        words_df = pd.DataFrame(word_counts, columns=["Word", "Frequency"])
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Frequency", y="Word", data=words_df, palette="rocket")
        plt.title("Top Keywords in Informative Tweets")
        plt.xlabel("Frequency")
        plt.ylabel("Word")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "top_keywords.png"), dpi=150)
        plt.close()

    print(f"All EDA figures successfully saved to {FIGURES_DIR}")

if __name__ == "__main__":
    generate_eda()
