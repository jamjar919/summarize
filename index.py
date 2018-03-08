from common import loadArticle, summarize, numWords
from tf import tf
from tfidf import tfidf
import sys, getopt
import os 

loc = os.path.dirname(os.path.realpath(__file__))

# CONSTANTS
SUMMARY_LENGTH = 100
USE_TFIDF = False
FILE_ROOT_PATH = loc+'/files'
PARSE_ARTICLES_INDIVIDUALLY = False

def main():
    # Choose parsing function
    if USE_TFIDF:
        print("Using TF-IDF mode")
        func = tfidf
    else:
        print("Using TF mode (default)")
        func = tf

    if (PARSE_ARTICLES_INDIVIDUALLY):
        print("Summarising per article")
        for i in range(1, 9):
            original = loadArticle(i, FILE_ROOT_PATH)
            summary = summarize(original, func, length=SUMMARY_LENGTH)

            print()
            print('Article', i)
            print(original.replace("\n", " "))
            print()
            print('Summary:')
            print(summary)
            print(numWords(summary), 'words')
            print()
    else:
        print("Summarising articles as one corpus (default)")
        text = ''
        for i in range(1, 9):
            original = loadArticle(i, FILE_ROOT_PATH)
            text += original + ' '
        summary = summarize(text, func, length=SUMMARY_LENGTH)
        print()
        print('Text')
        print(text.replace("\n", " "))
        print()
        print('Summary:')
        print(summary)
        print(numWords(summary), 'words')
        print()

if __name__ == "__main__":

    # Parse options
    try:
        opts, args = getopt.getopt(sys.argv[1:],"l:f:ip")
    except getopt.GetoptError as e:
        print (str(e))
        print("Usage: "+sys.argv[0]+" -l Integer length for the number of words, -i enables tf-idf mode, -f designates read directory")
        sys.exit(2)
    
    for o, a in opts:
        if o == '-l':
            SUMMARY_LENGTH=int(a)
        if o == '-i':
            USE_TFIDF = True
        if o == '-f':
            FILE_ROOT_PATH = loc+a
        if o == '-p':
            PARSE_ARTICLES_INDIVIDUALLY = True

    
    main()