# Tf stuff 
from sklearn.feature_extraction.text import CountVectorizer

# Other external libs
import numpy as np

# From common
from common import loadArticle, splitSentences, preprocess, calculateSentenceWeight, summarize

# CONSTANTS
SUMMARY_LENGTH = 500

def calculateTermFrequency(corpus, max_features=None, ngrams=1, vocabulary=None):
    '''
    Returns a list of terms and their count given a list of strings
    '''
    vectorizer = CountVectorizer(
        analyzer = "word",
        tokenizer = None,
        preprocessor = None,
        stop_words = None,
        max_features = max_features,
        ngram_range=(ngrams,ngrams),
        vocabulary=vocabulary
    )
    features = vectorizer.fit_transform(corpus)
    return (features.toarray(), vectorizer.get_feature_names())

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

for i in range(1, 9):
    original = loadArticle(i)
    summary = summarize(original, tf, length=SUMMARY_LENGTH)

    print()
    print('Article', i)
    print(original.replace("\n", " "))
    print()
    print('Summary:')
    print(summary)
    print(len(summary), 'chars')
    print()