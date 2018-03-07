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
SUMMARY_LENGTH = 500
DATE_STRING_LIST = [
    "jan", "january", "feb", "febuary", "mar", "march", "apr", "april", "jun", "june", "jul", "july", "aug", "august", "sep", "september",
    "oct", "october", "nov", "november", "dec", "december"
]

def loadArticle(id):
    a = open(FILE_ROOT_PATH+'/Doc '+str(id)+".txt", 'r', encoding='utf8');
    return a.read()+ ' '

def splitSentences(text):
    text = text.replace("\n", " ")
    text = text.replace("\"", "")
    return text.split('. ')

def preprocess(text):
    # Split to sentences
    sentences = splitSentences(text)
    
    stops = set(stopwords.words("english"))
    
    # Remove stops
    for i in range(0, len(sentences)):
        sentences[i] = text_to_word_sequence(sentences[i])
        sentences[i] = [w for w in sentences[i] if not w in stops]
        sentences[i] = " ".join(sentences[i])

    # Remove empty values
    sentences = list(filter(len, sentences))

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

def calculateSentenceWeight(sentence, frequency, terms):
    # Split into array
    sentence = text_to_word_sequence(sentence)

    # Calculate weights
    weight = 0
    for word in sentence:
        try:
            # Lookup term
            index = terms.index(word)
            weight += frequency[index]

            # Increase weight based on word content

            # Numbers are probably important
            if word.isdigit():
                weight += 1
                # Years are probably even more important
                if len(word) == 4:
                    weight += 1

            # Dates are probably important as well
            if word in DATE_STRING_LIST:
                weight += 1
        except ValueError:
            # Word was not in our vocab
            pass
    
    # Normalise by sentence size
    return weight/len(sentence)

def formatSummary(weightArray, originalText, length=500):
    sentences = splitSentences(originalText)
    result = ''
    i = 0
    while(
        len(result) + len(sentences[weightArray[i][0]]) < length
    ):
        result += sentences[weightArray[i][0]] + '. '
        i += 1

    return result

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

def summarize(original):
    # Init weights and sentences
    weights = tf(original)
    sentences = splitSentences(original)

    # Build our summary
    summarySentences = [] 
    while(
        reduce(lambda a, b: a + len(b[1]), summarySentences, 0) + len(sentences[weights[0][0]]) < SUMMARY_LENGTH
    ):
        # Pick the highest weight sentence from the article
        summarySentences.append((
            weights[0][0],
            sentences[weights[0][0]]
        ))

        # Remove that sentence
        sentences.pop(weights[0][0])

        # Regenerate article
        article = ". ".join(sentences)
        weights = tf(article)
        sentences = splitSentences(article)

    # Fill any more unused space
    i = 0
    while(
        reduce(lambda a, b: a + len(b[1]), summarySentences, 0) + len(sentences[weights[i][0]]) < SUMMARY_LENGTH
    ):
        summarySentences.append((
            weights[0][0],
            sentences[weights[0][0]]
        ))

    # Sort sentences by the position they appeared in the document
    summarySentences.sort(
        key=lambda a: a[0]
    )

    print(summarySentences)

    # Build the summary
    summary = reduce(
        lambda a, b: a + b[1] + '. ',
        summarySentences,
        ""
    )
    return summary

for i in range(1, 9):
    original = loadArticle(i)
    summary = summarize(original)

    print()
    print('Article', i)
    print(original.replace("\n", " "))
    print()
    print('Summary:')
    print(summary)
    print(len(summary), 'chars')
    print()