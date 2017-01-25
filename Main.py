import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from nltk.stem.porter import *
import numpy as np
from imblearn.over_sampling import RandomOverSampler

# from nltk.classify import maxent
# import datetime

stemmer = PorterStemmer()
f = open('output.txt', 'w')

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

print("Creating Bag-Of-words\n")
trainFile = pd.read_csv("corpus/input.csv")


testFile = pd.read_csv("testFile.csv")

f.write("Input file:\n")
f.write(str(testFile))

i = 0
for raw_song in testFile["song"]:
    letters_only = re.sub("[^\D]", " ", raw_song)  # The text to search
    testFile["song"][i] = letters_only
    i += 1

analyzer = CountVectorizer().build_analyzer()
print("Creating Word vectors\n")
vectorizer = CountVectorizer(analyzer=stemmed_words,
                             tokenizer=None,
                             lowercase=True,
                             preprocessor=None,
                             max_features=5000
                             )

train_data_features = vectorizer.fit_transform([r for r in trainFile["song"]])
test_data_features = vectorizer.transform([r for r in testFile["song"]])

train_data_features = train_data_features.toarray()
test_data_features = test_data_features.toarray()

print("Resampling corpus\n")
rs = RandomOverSampler()
X_resampledRe, y_resampledRe = rs.fit_sample(train_data_features,trainFile["genre"])

print("fitting for Naive bayes\n")
clf = MultinomialNB()
clf.fit(train_data_features, trainFile["genre"])
f.write("\nOutput from Naive Bayes Multi Normal:\n")
predicted = clf.predict(test_data_features)

f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['genre'])))
f.write("\n")

print("fitting for Naive bayes with resampled corpus\n")
clf = MultinomialNB()
clf.fit(X_resampledRe, y_resampledRe)
f.write("\nOutput from Naive Bayes Multi RE:\n")
predicted = clf.predict(test_data_features)

f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['genre'])))
f.write("\n")

print("fitting for SVC\n")
clf = SVC()
clf.fit(train_data_features, trainFile["genre"])
f.write("\nOutput from SVC Normal:\n")
predicted = clf.predict(test_data_features)
f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['genre'])))
f.write("\n")

print("fitting for SVC with resampled corpus\n")
clf = SVC()
clf.fit(X_resampledRe, y_resampledRe)
f.write("\nOutput from SVC RE:\n")
predicted = clf.predict(test_data_features)
f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['genre'])))
f.write("\n")

print("Completed!\nCheck output.txt for results")
