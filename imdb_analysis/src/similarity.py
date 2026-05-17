from pathlib import Path

import joblib
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
BERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILE = "bert_embeddings.pkl"
REVIEWS_FILE = "reviews.pkl"


def _build_sentence_transformer():
    # Import SentenceTransformer only when similarity features are actually used.
    # This keeps the rest of the pipeline importable even if torch/transformers
    # is not installed correctly in the current Python environment.
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers or torch could not be imported in the current "
            "Python environment. Activate the correct virtual environment or "
            "reinstall torch/sentence-transformers before using similarity features."
        ) from exc

    return SentenceTransformer(BERT_MODEL_NAME)


def train_similarity_model(data, text_column="clean_review", review_column="review"):
    model_bert = _build_sentence_transformer()

    # We use the processed review text to create embeddings, but keep the
    # original review text for display when showing similar examples later.
    review_text = data[text_column].astype(str).tolist()
    original_reviews = data[review_column].astype(str).tolist()

    embeddings = model_bert.encode(
        review_text,
        batch_size=32,
        show_progress_bar=True,
    )

    return embeddings, original_reviews


def save_similarity_artifacts(embeddings, reviews, model_dir=MODELS_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)

    # bert_embeddings.pkl stores the dense vectors that will be compared with
    # the query review embedding at runtime.
    joblib.dump(embeddings, model_dir / EMBEDDINGS_FILE)

    # reviews.pkl stores the original review texts in the same order as the
    # embedding rows, so we can map nearest-neighbor indices back to text.
    joblib.dump(reviews, model_dir / REVIEWS_FILE)


def load_similarity_artifacts(model_dir=MODELS_DIR):
    embeddings_path = model_dir / EMBEDDINGS_FILE
    reviews_path = model_dir / REVIEWS_FILE

    if not embeddings_path.exists() or embeddings_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Similarity embeddings file is missing or empty: {embeddings_path}"
        )

    if not reviews_path.exists() or reviews_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Similarity reviews file is missing or empty: {reviews_path}"
        )

    embeddings = joblib.load(embeddings_path)
    reviews = joblib.load(reviews_path)
    return embeddings, reviews


def get_similar_reviews(review, top_n=3, embeddings=None, reviews=None, model_dir=MODELS_DIR):
    if embeddings is None or reviews is None:
        embeddings, reviews = load_similarity_artifacts(model_dir)

    model_bert = _build_sentence_transformer()
    query_vec = model_bert.encode([review])
    scores = cosine_similarity(query_vec, embeddings)[0]
    similar_idx = scores.argsort()[-top_n:][::-1]
    return [reviews[i] for i in similar_idx]
