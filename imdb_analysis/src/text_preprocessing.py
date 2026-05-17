import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Ensure required NLTK packages are downloaded once before running:
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4")
# nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - {"not", "no", "very", "too", "just"}


def pipeline(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s.!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = text.lower()
    word_tokens = word_tokenize(text)

    cleaned_tokens = [
        lemmatizer.lemmatize(word, pos="v")
        for word in word_tokens
        if word not in stop_words
    ]

    return " ".join(cleaned_tokens)


def preprocess_reviews(data):
    # This function is the clearest place to see function-to-function flow:
    # raw dataframe comes in, processed dataframe goes out.
    # The returned dataframe is then passed to training functions.
    processed_data = data.copy()
    processed_data["clean_review"] = processed_data["review"].astype(str).apply(pipeline)
    return processed_data
