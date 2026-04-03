emotion_dict = {
    "joy": ["happy", "great", "amazing", "love", "fantastic", "awesome", "fun", "enjoyed", "brilliant"],
    "anger": ["hate", "worst", "terrible", "angry", "bad", "annoying", "frustrating"],
    "sadness": ["sad", "boring", "dull", "disappointing", "slow", "waste", "poor"],
    "surprise": ["unexpected", "shocking", "surprising", "twist"],
    "disgust": ["awful", "disgusting", "horrible", "pathetic", "ridiculous"]
}


def get_emotion_score(text):
    tokens = text.split()

    emotion_score = {key: 0 for key in emotion_dict}

    for word in tokens:
        for emotion, words in emotion_dict.items():
            if word in words:
                emotion_score[emotion] += 1

    total = sum(emotion_score.values())

    if total > 0:
        for key in emotion_score:
            emotion_score[key] /= total

    return emotion_score
