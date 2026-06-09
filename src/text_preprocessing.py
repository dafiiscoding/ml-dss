import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import MODELS_DIR

def clean_tweet_text(text):
    """
    Cleans raw tweet text to prepare it for vectorization.
    - Lowercase conversion
    - Remove URLs, @mentions, HTML characters
    - Remove special characters but keep words and numbers
    - Strip whitespace
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML entities like &amp;
    text = re.sub(r'&\w+;', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags symbol but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)

    # Keep only letters, numbers and basic spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Replace multiple spaces with a single space and strip
    text = re.sub(r'\s+', ' ', text).strip()

    return text

class TextVectorizerWrapper:
    """
    Wrapper for TfidfVectorizer to simplify saving, loading, and clean preprocessing.
    """
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False

    def fit(self, texts):
        cleaned_texts = [clean_tweet_text(t) for t in texts]
        self.vectorizer.fit(cleaned_texts)
        self.is_fitted = True
        return self

    def transform(self, texts):
        cleaned_texts = [clean_tweet_text(t) for t in texts]
        return self.vectorizer.transform(cleaned_texts)

    def fit_transform(self, texts):
        cleaned_texts = [clean_tweet_text(t) for t in texts]
        X = self.vectorizer.fit_transform(cleaned_texts)
        self.is_fitted = True
        return X

    def save(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "text_vectorizer.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Text Vectorizer saved to {filepath}")

    @staticmethod
    def load(filepath=None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "text_vectorizer.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file not found at {filepath}")
        with open(filepath, "rb") as f:
            wrapper = pickle.load(f)
        return wrapper

if __name__ == "__main__":
    # Test preprocessing
    sample_tweet = "URGENT: Flood waters rising fast near the bridge! @LocalPolice http://disaster.org/help #NepalEarthquake"
    cleaned = clean_tweet_text(sample_tweet)
    print(f"Original: {sample_tweet}")
    print(f"Cleaned:  {cleaned}")

    # Test vectorizer
    texts = [
        "Flood in California, homes destroyed",
        "Need donations and volunteers for wildfire rescue",
        "Check out this beautiful sunset today"
    ]
    vec = TextVectorizerWrapper(max_features=50)
    X = vec.fit_transform(texts)
    print(f"Vectorizer fitted. Vocabulary size: {len(vec.vectorizer.vocabulary_)}")
    vec.save()
