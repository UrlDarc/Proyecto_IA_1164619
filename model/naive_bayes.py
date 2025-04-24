import math

def predict(text, log_probs, class_priors, vocab, word_counts):
    words = text.split()
    scores = {}

    for label in class_priors:
        scores[label] = class_priors[label]
        for word in words:
            if word in vocab:
                scores[label] += log_probs[label].get(
                    word,
                    math.log(1 / (sum(word_counts[label].values()) + len(vocab)))
                )

    return max(scores, key=scores.get)
