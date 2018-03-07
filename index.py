from common import loadArticle, summarize, numWords
from tf import tf
from tfidf import tfidf
import sys, getopt

# CONSTANTS
SUMMARY_LENGTH = 100
USE_TFIDF = False

def main():
    # Choose parsing function
    if USE_TFIDF:
        print("Using TF-IDF mode")
        func = tfidf
    else:
        print("Using TF mode (default)")
        func = tf

    for i in range(1, 9):
        original = loadArticle(i)
        summary = summarize(original, func, length=SUMMARY_LENGTH)

        print()
        print('Article', i)
        print(original.replace("\n", " "))
        print()
        print('Summary:')
        print(summary)
        print(numWords(summary), 'words')
        print()

if __name__ == "__main__":

    # Parse options
    try:
        opts, args = getopt.getopt(sys.argv[1:],"l:i")
    except getopt.GetoptError as e:
        print (str(e))
        print("Usage: "+sys.argv[0]+" -l Integer length for the number of words, -i enables tf-idf mode")
        sys.exit(2)
    
    for o, a in opts:
        if o == '-l':
            SUMMARY_LENGTH=int(a)
        if o == '-i':
            USE_TFIDF = True
    
    main()