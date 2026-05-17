from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "imdb_analysis"

# Add imdb_analysis to sys.path so `from src...` imports work when this file
# is run as `python scripts/train_models.py` from the project root.
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.sentiment import save_sentiment_artifacts, train_sentiment_model
from src.similarity import save_similarity_artifacts, train_similarity_model
from src.text_preprocessing import preprocess_reviews


DATA_PATH = APP_DIR / "data" / "imdb_dataset.csv"


def main():
    print("Loading dataset...")
    data = pd.read_csv(DATA_PATH)

    # Step 1:
    # preprocess_reviews returns a new dataframe with a clean_review column.
    # That returned dataframe is stored in processed_data and then passed
    # directly into the next training functions.
    print("Preprocessing reviews...")
    processed_data = preprocess_reviews(data)

    # Step 2:
    # train_sentiment_model receives the processed dataframe and learns the
    # TF-IDF vocabulary plus the Logistic Regression weights.
    print("Training sentiment model...")
    tfidf, sentiment_model = train_sentiment_model(processed_data)
    save_sentiment_artifacts(tfidf, sentiment_model)
    print("Saved sentiment artifacts to imdb_analysis/models")

    # Step 3:
    # train_similarity_model uses the same processed dataframe to create BERT
    # embeddings. Here BERT is used as a pretrained encoder, so we save the
    # generated embeddings, not newly trained BERT weights.
    print("Generating BERT embeddings...")
    embeddings, reviews = train_similarity_model(processed_data)
    save_similarity_artifacts(embeddings, reviews)
    print("Saved similarity artifacts to imdb_analysis/models")

    print("Training workflow completed successfully.")


if __name__ == "__main__":
    main()
