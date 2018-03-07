import numpy as np

# From common
from common import preprocess, calculateSentenceWeight, splitWords

def calculateTermFrequency(corpus):
    features = []
    # Extract feature list
    for document in corpus:
        d = splitWords(document)
        for word in d:
            if len(word) > 0:
                if not word in features:
                    features.append(word)
    features.sort()

    result = np.zeros((len(corpus), len(features)))
    # Count
    for i in range(0, len(corpus)):
        document = corpus[i]
        d = splitWords(document)
        for word in d:
            index = features.index(word)
            result[i][index] = result[i][index] + 1

    return result, features

def tf(text):
    article = preprocess(text)
    frequency, terms = calculateTermFrequency(article)
    frequencySum = np.sum(frequency, axis=0)

    # Calculate sentence weights
    weightArray = []
    for i in range(0, len(article)):
        s = article[i]
        weight = calculateSentenceWeight(s, frequencySum, terms)
        weightArray.append((
            i,
            weight
        ))
    weightArray.sort(
        key=lambda a: a[1]
    )
    weightArray.reverse()
    return weightArray

