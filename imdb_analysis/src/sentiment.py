from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
SENTIMENT_LABELS = {"negative": 0, "positive": 1}
REVERSE_SENTIMENT_LABELS = {0: "negative", 1: "positive"}


def prepare_sentiment_training_data(data, preprocess_fn):
    # We keep raw review text and create a new clean_review column.
    # This makes the handoff explicit: the processed dataframe returned here
    # is what we pass into every later training function.
    processed_data = data.copy()
    processed_data["clean_review"] = (
        processed_data["review"].astype(str).apply(preprocess_fn)
    )
    processed_data["sentiment"] = processed_data["sentiment"].map(SENTIMENT_LABELS)
    return processed_data


def train_sentiment_model(data, preprocess_fn=None, text_column="clean_review"):
    working_data = data.copy()

    # Backward compatibility for notebook usage:
    # if clean_review does not exist yet, build it from the raw review column.
    if text_column not in working_data.columns:
        if preprocess_fn is None:
            raise ValueError(
                f"Column '{text_column}' not found. Pass preprocess_fn or create "
                "the processed column before training."
            )
        working_data = prepare_sentiment_training_data(working_data, preprocess_fn)

    if working_data["sentiment"].dtype == object:
        working_data["sentiment"] = working_data["sentiment"].map(SENTIMENT_LABELS)

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=5,
        max_df=0.8,
        sublinear_tf=True,
    )

    X = tfidf.fit_transform(working_data[text_column])
    y = working_data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("========================================================================")
    print("CONFUSION MATRIX :\n", confusion_matrix(y_test, y_pred))
    print("========================================================================")
    print("Precision Score:", precision_score(y_test, y_pred, pos_label=1))

    return tfidf, model


def save_sentiment_artifacts(tfidf, model, model_dir=MODELS_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, model_dir / "tfidf.pkl")
    joblib.dump(model, model_dir / "sentiment_model.pkl")


def load_sentiment_artifacts(model_dir=MODELS_DIR):
    tfidf = joblib.load(model_dir / "tfidf.pkl")
    model = joblib.load(model_dir / "sentiment_model.pkl")
    return tfidf, model


def predict_sentiment(cleaned_review, tfidf, model):
    # In inference we pass one cleaned review string into this function.
    # The same TF-IDF object learned during training converts that text into
    # the exact feature space expected by the trained classifier.
    vector = tfidf.transform([cleaned_review])
    prediction = int(model.predict(vector)[0])
    return REVERSE_SENTIMENT_LABELS[prediction]
