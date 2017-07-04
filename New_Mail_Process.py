from nltk.stem.snowball import SnowballStemmer
import string
import os
import sys
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

def parseOutText(f):
    

    f.seek(0)  # go back to beginning of file
    all_text = f.read()

    # split off metadata
    content = all_text.split("abcde")
    words = ""
    # remove punctuation
    text_string = content[0].translate(string.maketrans("", ""), string.punctuation)

    # split the text string into individual words, stem each word.
    # space between each stemmed word
    stemmer = SnowballStemmer("english")
    split = text_string.split()
    text = [stemmer.stem(word) for word in split]
    words = ' '.join(text)

    return words


def main():
    ff = open("../Mail_Classifier/New_Mail.txt", "r")
    text = parseOutText(ff)
    g = open("../Mail_Classifier/NewParsedtext.txt", "a")
    g.write(text)
    #g.write("\n")
    g.close()

def NVectorizer() :

    main()

    f = open("../Mail_Classifier/NewParsedtext.txt", "r")

    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')

    processed_input = vectorizer.fit_transform(f)
    
    return processed_input