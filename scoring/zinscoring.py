from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn import datasets, linear_model
from sklearn.pipeline import FeatureUnion
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from math import  sqrt
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import numpy as np
import random
import pickle

class AverageWordLength(BaseEstimator, TransformerMixin):
    def average_word_length(self, x):
        return np.mean([len(word) for word in x])

    def transform(self, X, y=None):
        return [[self.average_word_length(x)] for x in X]

    def fit(self, X, y=None):
        return self



class POSVectorizer(TfidfVectorizer):
    """ adds postags, learns weights """

    def postag(self, X):
        new_X = POStest
        print(new_X)
        return new_X

    def transform(self, X, y=None):
        X = self.postag(X)
        return super(POSVectorizer, self).transform(X, y)

    def fit(self, X, y=None):
        X = self.postag(X)
        return super(POSVectorizer, self).fit(X, y)

class AverageSentenceLength(BaseEstimator, TransformerMixin):
    def average_sent_length(self, x):
        return np.mean([len(x)])

    def transform(self, X, y=None):
        return [[self.average_sent_length(x)] for x in X]

    def fit(self, X, y=None):
        return self

class Freq(BaseEstimator, TransformerMixin):
    def freq_50(self, x):
        freq50 = ['die', 'van', 'het', 'en', 'in', "'n", 'is', 'nie', 'te', 'wat', 'om', 'op', 'se', 'vir', 'sy', 'met',
             'dit', 'hy', 'word', 'as', 'dat', 'aan', 'was', 'by', 'of', 'sal', 'hulle', 'the', 'kan', 'ek', 'diÈ',
             'ons', 'ook', 'oor', 'maar', 'gesÍ', 'deur', 'na', 'daar', 'tot', 'date', 'moet', 'hul', 'gaan', 'jaar',
             'toe', 'haar', 'teen', 'sÍ', 'meer']
        n = 0
        for i in x:
            if i in freq50:
                n = n+ 2
            else:
                continue
        return n

    def transform(self, X, y=None):
        return [[self.freq_50(x)] for x in X]

    def fit(self, X, y=None):
        return self


class Freq100(BaseEstimator, TransformerMixin):
    def freq_100(self, x):
        freq100 = ['die', 'van', 'het', 'en', 'in', "n", 'is', 'nie', 'te', 'wat', 'om', 'op', 'se', 'vir', 'sy', 'met',
                   'dit', 'hy', 'word', 'as', 'dat', 'aan', 'was', 'by', 'of', 'sal', 'hulle', 'the', 'kan', 'ek', 'diÈ',
                   'ons', 'ook', 'oor', 'maar', 'gesÍ', 'deur', 'na', 'daar', 'tot', 'date', 'moet', 'hul', 'gaan', 'jaar',
                   'toe', 'haar', 'teen', 'sÍ', 'meer', 'uit', 'jy', 'my', 'net', 'p', 'een', 'mnr', 'nog', 'wees', 'twee'
            , 'al', 'soos', 'and', 'mense', 'baie', 'hom', 'jou', 'ander', 'to', 'nou', 'a', 'volgens', 'eerste', 'maak',
                   'wil', 'nuwe', 'shy', 'so', 'groot', 'onder', 'gister', 'begin', 'n·', 'suid-afrika', 'kry', 'waar',
                   'kom', 'burger', 'laat', 'weer', 'voor', 'verlede', 'hoe', 'nadat', 'doen', 'drie', 'omdat', 'datum', 'tussen', 'de']

        n = 0
        for i in x:
            if i in freq100:
                n = n+ 2
            else:
                continue
        return n/len(x)

    def transform(self, X, y=None):
        return [[self.freq_100(x)] for x in X]

    def fit(self, X, y=None):
        return self


# This section reads the provided data and splits the file in two lists,
# one with the document data and one with the corresponding labels, which are both returned.
def read_corpus(corpus_file):
    documents = []
    posdocuments = []
    labels = []
    with open(corpus_file, encoding='ISO-8859-1') as f:
        for line in f:
            tokens = line.strip().split("\t")
            posdocuments.append(tokens[2])
            documents.append(tokens[1])
            labels.append(tokens[0])
    return documents, labels, posdocuments

# a dummy function that just returns its input
def identity(x):
    return x

def identitystr(x):
    for m in x:
        return m
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

X, Y, XPOS = read_corpus('tekstmetscore.txt')

c = list(zip(X, Y, XPOS))

random.shuffle(c)

X, Y, XPOS = zip(*c)

trainLength = int(0.8 * len(X))
devLength = int(1 * len(X))

Xtrain = X[:trainLength]
Ytrain = Y[:trainLength]
Xtest = X[trainLength:devLength]
Ytest = Y[trainLength:devLength]
POStrain = XPOS[:trainLength]
POStest = XPOS[trainLength:devLength]


with open('Xtest', 'wb') as fp:
    pickle.dump(Xtest, fp)
with open('Ytest', 'wb') as fp:
    pickle.dump(Ytest, fp)
with open('Xtrain', 'wb') as fp:
    pickle.dump(Xtrain, fp)
with open('Ytrain', 'wb') as fp:
    pickle.dump(Ytrain, fp)
with open('POStrain', 'wb') as fp:
    pickle.dump(POStrain, fp)
with open('Ytrain', 'wb') as fp:
    pickle.dump(POStest, fp)

"""
with open ('Xtrain', 'rb') as fp:
    Xtrain = pickle.load(fp)
with open ('Ytrain', 'rb') as fp:
    Ytrain = pickle.load(fp)
with open ('Xtest', 'rb') as fp:
    Xtest = pickle.load(fp)
with open ('Ytest', 'rb') as fp:
    Ytest = pickle.load(fp)
"""
n_gram_char_vec = TfidfVectorizer(ngram_range=(1, 3), min_df = 2, analyzer = 'char', binary=True, preprocessor = identitystr)

n_gram_word_vec = TfidfVectorizer( ngram_range=(1,9),analyzer="word" , preprocessor = identity,
                          tokenizer = identity)

vec = FeatureUnion([
                    #("char", n_gram_char_vec),
                    ("word",n_gram_word_vec),
                    #('pos_ngrams', POSVectorizer(ngram_range=(1, 4), analyzer="word")),
                    ('average_word_length', AverageWordLength()),
                    ("average_sentence_lenght", AverageSentenceLength()),
                    #("freq50", Freq()),
                    ("freq100", Freq100()),
                    #("POS", POS(POStrain, POStest))
                    ])


# Create linear regression object
regr = Pipeline( [('vec', vec),
                        ('regr', linear_model.LinearRegression())] )



# Train the model using the training sets
regr.fit(Xtrain, Ytrain)

# Make predictions using the testing set
Yguess = regr.predict(POStest)

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
Ytest2 = []
for item in Ytest:
    Ytest2.append(float(item))

print("Mean squared error: %.2f"
      % mean_squared_error(Ytest2, Yguess))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Ytest2, Yguess))
print("test ",Ytest2,  "guess ",Yguess)

