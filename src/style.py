import nltk

# Ensure required NLTK data is downloaded (run once):
# nltk.download('averaged_perceptron_tagger')


def get_writing_style(text):
    tokens = text.split()
    
    pos_tags = nltk.pos_tag(tokens)
    total_words = len(tokens)
    noun_count = 0
    verb_count = 0  
    adj_count = 0
    adv_count = 0

    for word ,tag in pos_tags:
        if tag.startswith('NN'):
            noun_count+=1
        elif  tag.startswith('VB'):
            verb_count+=1
        elif tag.startswith('JJ'):
            adj_count+=1
        elif tag.startswith('RB'):
            adv_count+=1

    # avoid division by zero
    if total_words == 0:
        return{}
    
    return {
        "noun_ratio": noun_count/total_words,
        "verb_ratio": verb_count / total_words,
        "adj_ratio": adj_count / total_words,
        "adv_ratio": adv_count / total_words,
        "avg_word_length": sum(len(w) for w in tokens) / total_words,
        "sentence_length": total_words
    }