import os
import sys
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

def Vectorizer() :
	sys.path.append( "../Mail_Classifier/" )
	from Parsing import main

	main()

	f = open("../Mail_Classifier/Parsedtext.txt", "r")

	g =open("../Mail_Classifier/Train_Author.txt", "r")

	vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')

	processed_input = vectorizer.fit_transform(f)
	h = g.read()
	processed_labels = h.split()

	return processed_input, processed_labels