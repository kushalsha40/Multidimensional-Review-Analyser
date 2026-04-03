# src/sentiment.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score


def train_sentiment_model(data, pipeline):
    # Step 1: Clean text
    data['review'] = data['review'].apply(pipeline)
    data['sentiment'] = data['sentiment'].map({ 'negative': 0, 'positive': 1 })
    # Step 2: TF-IDF
    tfidf = TfidfVectorizer(
        ngram_range=(1,2),   # 3-grams removed (better stability)
        max_features=5000,
        min_df=5,
        max_df=0.8,
        sublinear_tf=True
    )

    X = tfidf.fit_transform(data['review'])
    y = data['sentiment']

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4: Train model
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

    # Step 5: Predict
    y_pred = model.predict(X_test)

    # Step 6: Evaluation
    print(classification_report(y_test, y_pred))
    print('========================================================================')
    print('CONFUSION MATRIX :\n', confusion_matrix(y_test, y_pred))
    print('========================================================================')
    print("Precision Score (macro):", precision_score(y_test, y_pred, pos_label='positive'))

    return tfidf, model