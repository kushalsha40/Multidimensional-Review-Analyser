from functools import lru_cache
from pathlib import Path

try:
    from src.aspect import classify_aspect
    from src.emotion import get_emotion_score
    from src.sentiment import load_sentiment_artifacts, predict_sentiment
    from src.similarity import get_similar_reviews, load_similarity_artifacts
    from src.style import get_writing_style
    from src.text_preprocessing import pipeline
except ModuleNotFoundError:
    # Fallback for direct file execution such as:
    # python imdb_analysis/src/pipeline.py
    from aspect import classify_aspect
    from emotion import get_emotion_score
    from sentiment import load_sentiment_artifacts, predict_sentiment
    from similarity import get_similar_reviews, load_similarity_artifacts
    from style import get_writing_style
    from text_preprocessing import pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


@lru_cache(maxsize=1)
def load_runtime_artifacts():
    # These objects are trained once, saved to disk, and then loaded here
    # during API/runtime prediction.
    tfidf, sentiment_model = load_sentiment_artifacts(MODELS_DIR)

    embeddings = None
    reviews = None
    if (MODELS_DIR / "bert_embeddings.pkl").exists() and (MODELS_DIR / "reviews.pkl").exists():
        embeddings, reviews = load_similarity_artifacts(MODELS_DIR)

    return tfidf, sentiment_model, embeddings, reviews


def analyze_review(review):
    # Step 1: preprocess one incoming review string.
    cleaned_review = pipeline(review)

    # Step 2: load the saved training artifacts.
    tfidf, sentiment_model, embeddings, reviews = load_runtime_artifacts()

    # Step 3: pass the processed review into each analysis function.
    sentiment = predict_sentiment(cleaned_review, tfidf, sentiment_model)
    emotion = get_emotion_score(cleaned_review)
    style = get_writing_style(cleaned_review)
    aspect = classify_aspect(cleaned_review, tfidf, sentiment_model)

    similar_reviews = []
    if embeddings is not None and reviews is not None:
        try:
            similar_reviews = get_similar_reviews(

                top_n=3,
                embeddings=embeddings,
                reviews=reviews,
            )
        except RuntimeError:
            # Similarity is optional at runtime. If sentence-transformers or torch
            # is unavailable, return the rest of the analysis instead of failing.
            similar_reviews = []

    return {
        "review": review,
        "cleaned_review": cleaned_review,
        "sentiment": sentiment,
        "emotion": emotion,
        "style": style,
        "aspect": aspect,
        "similar_reviews": similar_reviews,
    }
print("Pipeline functions defined successfully.")