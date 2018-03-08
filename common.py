from functools import reduce

DATE_STRING_LIST = [
    "jan", "january", "feb", "febuary", "mar", "march", "apr", "april", "jun", "june", "jul", "july", "aug", "august", "sep", "september",
    "oct", "october", "nov", "november", "dec", "december"
]
CHARS_TO_STRIP = ["!","\"","#","$","%","&","(",")","*","+",",","-",".","/",":",";","<","=",">","?","@","[","\\","]","^,","_","`","{","|","}",
    "~","\t","\n"]
STOPWORDS = {'at', 'doesn', 'whom', 'had', 'those', 'few', 'me', 'each', 'their', 'him', 'some', 'the', 'that', "she's", 'being', 'through',
    'both', 'can', 'were', 'with', 'by', 'down', 'into', 'why', 'you', 'weren', 'where', 'then', 'he', 'between', 'mustn', 'until', "wouldn't",
    'further', 'yourselves', "mightn't", 'theirs', 'there', 'and', 'again', 'not', "shouldn't", 'here', 'in', 'd', 'during', 'does', 'more',
    'couldn', 't', 'haven', 'shouldn', 'its', 'too', 'myself', 'after', 'this', 'of', "needn't", 'about', 'o', "that'll", "won't", 'because',
    'themselves', 'them', 'ourselves', 'these', 'but', 'wouldn', 'ain', 'been', 'than', 'to', 'or', "you'll", 'should', 'hers', 'needn', 'll',
    'now', 'doing', 'yours', 'which', 'mightn', "hadn't", "you're", 'my', 'only', 'same', 'is', 'do', 're', 'his', 'ma', 'all', 'having', 'as',
    'when', "wasn't", 'nor', 'her', "isn't", 'has', "you'd", 'isn', 'they', 'such', 'ours', 'how', 'itself', 'be', 'for', 'out', 'it', 'from',
    'on', 'shan', 'won', 'will', 'our', 'up', 'below', 'was', "didn't", 'above', 'very', "aren't", 'once', 'a', 'i', 'himself', "you've", 'who',
    "mustn't", "should've", "don't", 'am', 'other', 'any', 'own', "hasn't", 'over', 'we', 'if', 's', "haven't", 've', 'under', 'just', 'hasn',
    'are', 'off', 'no', 'wasn', "couldn't", "doesn't", "weren't", 'y', 'what', 'against', 'm', 'yourself', 'aren', "shan't", 'herself', 'she',
    'did', 'don', 'didn', 'an', 'most', "it's", 'so', 'have', 'your', 'hadn', 'while', 'before'}

def loadArticle(id, path):
    a = open(path+'/Doc '+str(id)+".txt", 'r', encoding='utf8');
    return a.read()+ ' '

def splitSentences(text):
    text = text.replace("\n", " ")
    text = text.replace("\"", "")
    sentences = text.split(". ")
    # Remove empty values
    sentences = list(filter(len, sentences))
    return sentences

def splitWords(text):
    text = text.strip()
    text = text.lower()
    stripped = ""
    for c in text:
        if not c in CHARS_TO_STRIP:
            stripped += c

    stripped = stripped.split(" ")
    for i in range(0, len(stripped)):
        hasAp = stripped[i].find("'")
        if hasAp != -1:
            stripped[i] = stripped[i][0:hasAp] 
    
    while '' in stripped:
        stripped.remove('')

    return stripped

def preprocess(text):
    # Split to sentences
    sentences = splitSentences(text)
    
    stops = STOPWORDS
    
    # Remove stops
    for i in range(0, len(sentences)):
        sentences[i] = splitWords(sentences[i])
        sentences[i] = [w for w in sentences[i] if not w in stops]
        sentences[i] = " ".join(sentences[i])

    # Remove empty values
    sentences = list(filter(len, sentences))

    return sentences


def calculateSentenceWeight(sentence, frequency, terms):
    # Split into array
    sentence = splitWords(sentence)

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
    if len(sentence) < 4:
        return 0
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