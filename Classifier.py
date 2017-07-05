import sys
from time import time
sys.path.append("../Mail_Classifier/")
from Vectorizer import Vectorizer

sys.path.append("../Mail_Classifier/")
from New_Mail_Process import NVectorizer

features_train, features_test, labels_train, labels_test = Vectorizer()

features_new = NVectorizer()

from sklearn import svm
clf = svm.SVC(kernel = 'rbf', C = 10000.0)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
pred = clf.predict(features_test)
print accuracy_score(labels_test, pred)

print clf.predict(features_new)
