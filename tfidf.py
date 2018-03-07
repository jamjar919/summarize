import numpy as np

from tf import calculateTermFrequency
from common import preprocess, calculateSentenceWeight

def tfidf(text):
    article = preprocess(text)
    frequency, terms = calculateTermFrequency(article)

    # Apply TDIF
    tfidf = TfidfTransformer()
    frequency = tfidf.fit_transform(frequency).todense()

    frequencySum = np.sum(frequency, axis=0)

    # Needs an extra unpack because tdif uses np 
    frequencySum = frequencySum.tolist()[0]

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

