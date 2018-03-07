from nltk.corpus import stopwords
from keras.preprocessing.text import text_to_word_sequence
from functools import reduce
import os 

loc = os.path.dirname(os.path.realpath(__file__))

DATE_STRING_LIST = [
    "jan", "january", "feb", "febuary", "mar", "march", "apr", "april", "jun", "june", "jul", "july", "aug", "august", "sep", "september",
    "oct", "october", "nov", "november", "dec", "december"
]
FILE_ROOT_PATH = loc+'/files'


def loadArticle(id):
    a = open(FILE_ROOT_PATH+'/Doc '+str(id)+".txt", 'r', encoding='utf8');
    return a.read()+ ' '

def splitSentences(text):
    text = text.replace("\n", " ")
    text = text.replace("\"", "")
    sentences = text.split(". ")
    # Remove empty values
    sentences = list(filter(len, sentences))
    return sentences

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

def numWords(text):
    '''Count the number of words in some text'''
    return len(text.split(" "))

def summarize(original, weightFunction, length=100):
    # Init weights and sentences
    weights = weightFunction(original)
    sentences = splitSentences(original)

    # Build our summary
    summarySentences = [] 
    while(
        (len(sentences) > 1) and
        (reduce(lambda a, b: a + numWords(b[1]), summarySentences, 0) + numWords(sentences[weights[0][0]]) < length)
    ):
        # Pick the highest weight sentence from the article
        summarySentences.append((
            weights[0][0],
            sentences[weights[0][0]]
        ))

        # Remove that sentence
        sentences.pop(weights[0][0])

        # Regenerate article
        if (len(sentences) > 1):
            article = ". ".join(sentences)
            weights = weightFunction(article)
            sentences = splitSentences(article)

    # Fill any more unused space
    i = 0
    while(
        (len(sentences) > 1) and
        reduce(lambda a, b: a + numWords(b[1]), summarySentences, 0) + numWords(sentences[weights[i][0]]) < length
    ):
        summarySentences.append((
            weights[0][0],
            sentences[weights[0][0]]
        ))
        i += 1

    # Sort sentences by the position they appeared in the document
    summarySentences.sort(
        key=lambda a: a[0]
    )

    # Build the summary
    summary = reduce(
        lambda a, b: a + b[1] + '. ',
        summarySentences,
        ""
    )
    return summary