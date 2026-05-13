# Loading Pretrained BERT
import pandas as pd

from sentence_transformers import SentenceTransformer   
model_bert = SentenceTransformer('all-MiniLM-L6-v2')

# Converting the reviews to Embedings
data = pd.read_csv('../data/imdb_dataset.csv')
embeddings = model_bert.encode(data['review'].tolist())

# Similarity funtion

from sklearn.metrics.pairwise import cosine_similarity


def get_similar_reviews(review, embeddings, top_n=3):

    query_vec = model_bert.encode([review]).reshape(1,-1)

    score = cosine_similarity(query_vec , embeddings)[0]


    similar_idx = score.argsort()[-(top_n+1):-1][::-1]

    return similar_idx