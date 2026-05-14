# Loading Pretrained BERT
import pandas as pd

# Similarity funtion

from sklearn.metrics.pairwise import cosine_similarity


def get_similar_reviews(review, embeddings, top_n=3):

    query_vec = model_bert.encode([review]).reshape(1,-1)

    score = cosine_similarity(query_vec , embeddings)[0]


    similar_idx = score.argsort()[-(top_n+1):-1][::-1]

    return similar_idx