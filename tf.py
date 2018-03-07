# Deeplearning 
from keras.preprocessing.text import text_to_word_sequence
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords

# Other external libs
import numpy as np

# Standard libs
from functools import reduce
import os 

loc = os.path.dirname(os.path.realpath(__file__))

# CONSTANTS
FILE_ROOT_PATH = loc+'/files'

def loadArticle(id):
    a = open(FILE_ROOT_PATH+'/Doc '+str(id)+".txt", 'r', encoding='utf8');
    return a.read()

def preprocess(text):
    # Split to sentences
    text = text.replace("\n", " ")
    sentences = text.split('. ')
    
    stops = set(stopwords.words("english"))
    
    # Remove stops
    for i in range(0, len(sentences)):
        sentences[i] = text_to_word_sequence(sentences[i])
        sentences[i] = [w for w in sentences[i] if not w in stops]
        sentences[i] = " ".join(sentences[i])

    return sentences

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

def printFrequencyCount(frequency, terms):
    dist = np.sum(frequency, axis=0)
    for tag, count in zip(terms, dist):
        print(count, tag)

for i in range(1, 9):
    original = loadArticle(i)
    article = preprocess(original)
    frequency, terms = calculateTermFrequency(article)
    frequencySum = np.sum(frequency, axis=0)
    print(article)