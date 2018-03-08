# summarize
This program summarises text using the TF and TF-IDF measures.

## How to run
To run the program, you need to invoke “index.py” through the command line. It accepts these arguments:

    python index.py -l 200 -i -p -f /files

    -l Designates the maximum length of the summarisation to be generated. Default: 150
    -i Specifies that TF-IDF is to be applied. Does not require an argument. Default: False
    -f Designates the directory for which the files are read from. Default: /files
    -p Use this option to parse each article as its own text. Does not require an argument. Default: False

The program will automatically try to find files of the name “Doc X.txt” and run the summariser on the
content in the directory specified.