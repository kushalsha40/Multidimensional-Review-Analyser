import nltk


aspects = {
    "story": ["story", "plot", "script"],
    "acting": ["acting", "actor", "performance"],
    "visuals": ["visual", "cinematography", "vfx"],
    "music": ["music", "song", "background"],
    "direction": ["director", "direction"]
}

def classify_aspect(review, tfidf, model):
    review = review.lower()
    sentence = nltk.sent_tokenize(review)

    result = {}
    aspect_sentiments = []
    for aspect , keywords in aspects.items():
        
        for sent in sentence:
            for word in keywords:
                if word in sent:
                    vec = tfidf.transform([sent])
                    pred = model.predict(vec)[0]
                    aspect_sentiments.append(pred)

        if len(aspect_sentiments) == 0:
            result[aspect] = 'neutral'
        else:
            result[aspect] = max(set(aspect_sentiments), key = aspect_sentiments.count)
    
    
    return result