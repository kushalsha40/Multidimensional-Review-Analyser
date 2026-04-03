import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure required NLTK packages are downloaded (run once):
# nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'not','no','very','too','just'}

def pipeline(text):
    # 1. Cleaning: HTML, URLs, Mentions, Hashtags, Non-alphabetic
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[@#]\w+', ' ', text)
    text = re.sub(r"[^a-zA-Z\s.!?']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Tokenization & Lowercasing
    text = text.lower()
    word_tokens = word_tokenize(text)

    # 3. Lemmatization & Stopword Removal
    # We keep the words in a list to be joined back into a clean string
    cleaned_tokens = [
        lemmatizer.lemmatize(word, pos='v') 
        for word in word_tokens 
        if word not in stop_words
    ]

    return " ".join(cleaned_tokens)

