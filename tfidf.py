import numpy as np

from tf import calculateTermFrequency
from common import preprocess, calculateSentenceWeight

def transformFrequencyTdidf(frequency):
    numDocs = frequency.shape[0]
    docContainsWord = frequency
    for i in range(0, numDocs):
        docContainsWord[i] = docContainsWord[i].astype(bool)
    docFrequency = np.sum(docContainsWord, axis=0)
    for i in range(0, numDocs):
        for j in range(0, frequency.shape[1]):
            # Calculate our IDF score for this word
            idf = np.log10(numDocs/(1 + docFrequency[j]))
            # Reassign the frequency
            frequency[i][j] = frequency[i][j] * idf
    return frequency

def tfidf(text):
    article = preprocess(text)
    frequency, terms = calculateTermFrequency(article)

    frequency = transformFrequencyTdidf(frequency)
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

