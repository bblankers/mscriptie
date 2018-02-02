from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn import svm
from sklearn import datasets, linear_model
from sklearn.pipeline import FeatureUnion
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from math import  sqrt
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
import nltk
import numpy as np
import random
import pickle
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVR
from sklearn.metrics import explained_variance_score
import sys

from features import *

# This section reads the provided data and splits the file in two lists,
# one with the document data and one with the corresponding labels, which are both returned.
def read_corpus(corpus_file):
    documents = []
    labels = []
    with open(corpus_file, encoding='ISO-8859-1') as f:
        for line in f:
            tokens = line.strip().split("\t")
            documents.append(tokens[1])
            labels.append( float(tokens[0]) )
    return documents, labels

# a dummy function that just returns its input
def identity(x):
    return x

"""
with open('freq100.txt', 'r') as f:
    list = []
    for line in f:
        line2 = line.split()
        list.append(line2[1])
    print(list)
"""
# This part reads and splits the corpus in a train(60%), dev(20%) and test(20%) set.
# When ready to use the test set change 0.6 to 0.8 and 0.8 to 1.
# After the split the training and test values for X(data) and Y(labels) are initialized

X, Y = read_corpus(sys.argv[1])
"""
c = list(zip(X, Y))

random.shuffle(c)

X, Y = zip(*c)


trainLength = int(0.8 * len(X))
devLength = int(1 * len(X))

Xtrain = X[:trainLength]
Ytrain = Y[:trainLength]
Xtest = X[trainLength:devLength]
Ytest = Y[trainLength:devLength]


with open('Xtest', 'wb') as fp:
    pickle.dump(Xtest, fp)
with open('Ytest', 'wb') as fp:
    pickle.dump(Ytest, fp)
with open('Xtrain', 'wb') as fp:
    pickle.dump(Xtrain, fp)
with open('Ytrain', 'wb') as fp:
    pickle.dump(Ytrain, fp)
Ytest2 = []


with open ('Xtrain', 'rb') as fp:
    Xtrain = pickle.load(fp)
with open ('Ytrain', 'rb') as fp:
    Ytrain = pickle.load(fp)
with open ('Xtest', 'rb') as fp:
    Xtest = pickle.load(fp)
with open ('Ytest', 'rb') as fp:
    Ytest = pickle.load(fp)
"""
n_gram_char_vec = TfidfVectorizer(ngram_range=(1, 9), min_df = 2, analyzer = 'char', binary=True, preprocessor = identity)

n_gram_word_vec = TfidfVectorizer( ngram_range=(2,7),analyzer="word" , binary=True,preprocessor = identity,
                          tokenizer = identity)
vec = FeatureUnion([
                    ("char", n_gram_char_vec),
                    ("word",n_gram_word_vec),
                    ('average_word_length', AverageWordLength()),
                    ("average_sentence_length_words", AverageSentenceLengthWords()),
                    #("average_sentence_lenght_char", AverageSentenceLength()),
                    #("freq50", Freq()),
                    ("freq100", Freq100())
                    ])


# Create linear regression object
regr = Pipeline( [('vec', vec),
                        ('regr', linear_model.Ridge())] )



# Train the model using the training sets
regr.fit(X, Y)

# Make predictions using the testing set
Yguess = regr.predict(X)


def classify(classifier, data, labeltype):
    return cross_val_predict(classifier, data, labeltype, cv = 8)

Yguess = classify(regr, X, Y)


print("Mean squared error: %.9f"
      % mean_squared_error(Y, Yguess))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y, Yguess))
print('explaine variance: ',explained_variance_score(Y, Yguess))
#plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
