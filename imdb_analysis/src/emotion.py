from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

emotion_dict = {
    "joy": [
        "happy",
        "great",
        "amazing",
        "love",
        "fantastic",
        "awesome",
        "fun",
        "enjoyed",
        "brilliant",
        "wonderful",
        "excellent",
    ],
    "anger": [
        "hate",
        "worst",
        "terrible",
        "angry",
        "bad",
        "annoying",
        "frustrating",
        "furious",
    ],
    "sadness": [
        "sad",
        "boring",
        "dull",
        "disappointing",
        "slow",
        "waste",
        "poor",
        "depressing",
    ],
    "surprise": [
        "unexpected",
        "shocking",
        "surprising",
        "twist",
        "surprised",
        "shock",
    ],
    "disgust": [
        "awful",
        "disgusting",
        "horrible",
        "pathetic",
        "ridiculous",
        "gross",
    ],
}


def _normalize_word(word):
    # The review text is already lemmatized by the preprocessing pipeline.
    # We normalize emotion keywords the same way so exact matches work.
    return lemmatizer.lemmatize(word.lower(), pos="v")


normalized_emotion_dict = {
    emotion: {_normalize_word(word) for word in words}
    for emotion, words in emotion_dict.items()
}


def get_emotion_score(text):
    tokens = [_normalize_word(token) for token in text.split()]
    emotion_score = {key: 0.0 for key in normalized_emotion_dict}

    for word in tokens:
        for emotion, words in normalized_emotion_dict.items():
            if word in words:
                emotion_score[emotion] += 1.0

    total = sum(emotion_score.values())

    if total > 0:
        for key in emotion_score:
            emotion_score[key] = round(emotion_score[key] / total, 4)

    return emotion_score
