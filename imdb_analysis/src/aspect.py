import nltk


aspects = {
    "story": ["story", "plot", "script"],
    "acting": ["acting", "actor", "performance"],
    "visuals": ["visual", "cinematography", "vfx"],
    "music": ["music", "song", "background"],
    "direction": ["director", "direction"],
}

label_map = {0: "negative", 1: "positive"}


def classify_aspect(review, tfidf, model):
    review = review.lower()
    sentences = nltk.sent_tokenize(review)

    result = {}
    for aspect, keywords in aspects.items():
        aspect_sentiments = []

        for sentence in sentences:
            if any(keyword in sentence for keyword in keywords):
                vec = tfidf.transform([sentence])
                pred = int(model.predict(vec)[0])
                aspect_sentiments.append(label_map[pred])

        if not aspect_sentiments:
            result[aspect] = "neutral"
        else:
            result[aspect] = max(set(aspect_sentiments), key=aspect_sentiments.count)

    return result
