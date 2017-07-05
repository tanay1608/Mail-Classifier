iimport os
import sys
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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

	feature_train, feature_test, labels_train, labels_test = train_test_split(processed_input, processed_labels, test_size = 0.2, random_state = 42)
	return feature_train, feature_test, labels_train, labels_test
