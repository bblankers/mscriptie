"""
Model that got the highest score in the development phase, it is unstable and should not be used 


Call python3 tekstmodel_max.py traindocument testdocument
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.metrics import *
from sklearn import svm
from sklearn import datasets, linear_model
from sklearn.pipeline import FeatureUnion
from nltk.corpus import stopwords
from sklearn.pipeline import make_pipeline
from nltk import word_tokenize
from math import  sqrt
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
import nltk
import numpy as np
import random
import pickle
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVR
from sklearn.metrics import explained_variance_score
import sys
import pandas as pd
import pylab as pl
from sklearn import datasets
from sklearn.decomposition import PCA
from features import *

# This section reads the provided data and splits the file in two lists,
# one with the document data and one with the corresponding labels, which are both returned.
def read_corpus(corpus_file):
    documents = []
    labels = []
    with open(corpus_file, encoding='ISO-8859-1') as f:
        for line in f:
            tokens = line.strip().split("\t")
            documents.append(tokens[1:])
            labels.append( float(tokens[0]) )
    #print(documents)
    return documents, labels

# a dummy function that just returns its input

def identity(x):
    return x

def docs(x):
    return x[0]


# this pos function returns only the first letter of the tag = the type of word
def pos(x):
    letters = []
    for i in x[1].split():
        letters.append(i[0])
    return ' '.join(letters)

#this pos function return the entire POS tag, which is a very specific tag
def pos_specific(x):
    return(x[1])

def pc(x):
    return(x[2])


#this pos function returns only sentence ending, indicating the number of sentences in a doc
def pos_ze(x):
    ze = []
    for i in x[1].split():
        if i == 'ZE':
            ze.append(i)
    return ze

def type_token_ratio(x):
    uniquewords=dict()
    words=0
    for word in x[0].split():
        words+=1
        if word in uniquewords:
            uniquewords[word]+=1
        else:
            uniquewords[word]=1
    TTR= len(uniquewords)/words
    print(TTR)
    return x[0]



def syllables(word):
    count = 0
    vowels = 'aeiouy'
    word = word.lower().strip(".:;?!")
    if word and word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if count == 0:
        count += 1
    return count

def syl(x):
    x = x[0].split()
    return [str(syllables(word)) for word in x]

def tokenize(x):
    return nltk.word_tokenize(x)


def freq(x):
    freq100 = ['die', 'van', 'het', 'en', 'in', "n", 'is', 'nie', 'te', 'wat', 'om', 'op', 'se', 'vir', 'sy', 'met',
                   'dit', 'hy', 'word', 'as', 'dat', 'aan', 'was', 'by', 'of', 'sal', 'hulle', 'the', 'kan', 'ek', 'diÈ',
                   'ons', 'ook', 'oor', 'maar', 'gesÍ', 'deur', 'na', 'daar', 'tot', 'date', 'moet', 'hul', 'gaan', 'jaar',
                   'toe', 'haar', 'teen', 'sÍ', 'meer', 'uit', 'jy', 'my', 'net', 'p', 'een', 'mnr', 'nog', 'wees', 'twee'
            , 'al', 'soos', 'and', 'mense', 'baie', 'hom', 'jou', 'ander', 'to', 'nou', 'a', 'volgens', 'eerste', 'maak',
                   'wil', 'nuwe', 'shy', 'so', 'groot', 'onder', 'gister', 'begin', 'n·', 'suid-afrika', 'kry', 'waar',
                   'kom', 'burger', 'laat', 'weer', 'voor', 'verlede', 'hoe', 'nadat', 'doen', 'drie', 'omdat', 'datum', 'tussen', 'de']
    tokens = x[0].split()
    #lens = len(list(set(freq100)&set(tokens)))
    return set(freq100)&set(tokens)


def freq_yes(tokens):
    #print([tokens for tokens in freq(tokens)])
    return [tokens for tokens in freq(tokens)]


# This part reads and splits the corpus in a train(60%), dev(20%) and test(20%) set.
# When ready to use the test set change 0.6 to 0.8 and 0.8 to 1.
# After the split the training and test values for X(data) and Y(labels) are initialized

Xtrain, Ytrain = read_corpus(sys.argv[1])
"""
Few lines to create 80% of train for final comparison

dataGood = list(zip(Xtrain,Ytrain))

random.seed()
random.shuffle(dataGood)

Xtrain,Ytrain = zip(*dataGood)
trainLength = int(0.8*len(Xtrain))
Xtrain, Ytrain = Xtrain[:trainLength], Ytrain[:trainLength]
print(len(Xtrain))
"""
Xtest, Ytest = read_corpus(sys.argv[2])


# Character ngrams
n_gram_char_vec = CountVectorizer(ngram_range=(1, 4), analyzer = 'char', min_df=2, binary=True, preprocessor = docs, tokenizer= tokenize)

# POS ngrams
n_gram_pos_vec = CountVectorizer( ngram_range=(1,2), analyzer= 'word', min_df=1 ,binary=False,preprocessor = pos_specific, tokenizer = identity)

# POS_letter ngrams
n_gram_pos_letter_vec = TfidfVectorizer( ngram_range=(1,2),analyzer="word",binary=False,preprocessor = pos, tokenizer = identity)

# Word ngrams
n_gram_word_vec = CountVectorizer( ngram_range=(1,2), analyzer="word",binary=False,preprocessor = docs, tokenizer = tokenize)

# syllable ngrams
n_gram_syl_vec = CountVectorizer( ngram_range=(1,1),analyzer="word" ,preprocessor = syl,
                          tokenizer = identity)
# PC ngrams
n_gram_pc_vec = CountVectorizer( ngram_range=(1,2),analyzer="word",binary=True,preprocessor = pc, tokenizer = tokenize)


# Only returns words if they are in the most frequent 50 words, not in use
#n_gram_freq_vec = TfidfVectorizer(ngram_range=(1,1),analyzer="word" ,preprocessor = identity, binary=True, min_df=2,
 #                         tokenizer = freq_yes)
vec = FeatureUnion([
                    #("ttr", type_token()),
                    #("char", n_gram_char_vec),
                    ("pos",n_gram_pos_vec),
                    #("pos_letter",n_gram_pos_letter_vec),
                    #("word", n_gram_word_vec),
                    ("syll_word", n_gram_syl_vec),
                    #("PC_ngrams",n_gram_pc_vec),
                    ('AVG word length', AverageWordLength()),
                    ("average_sentence_length_words", AverageSentenceLengthWords()),
                    #("abs_sentence_length_words", SentenceLength()),
                    #("average_sentence_lenght_char", AverageSentenceLength()),
                    ("freq_feature", Freq502()),
                    ])

# Create linear regression object
#anova_filter = SelectKBest(f_regression, k='all')
#clf = SVR(kernel='linear')
clf = linear_model.LinearRegression()
#clf = linear_model.SGDRegressor()
#regr = Pipeline( [('vec', vec),
 #                           ('regr',SVR(kernel='linear'))] )



regr=make_pipeline(vec,clf)
# Train the model using the training sets
regr.fit(Xtrain, Ytrain)


# Make predictions using the testing set
Yguess = regr.predict(Xtest)

# CV currently set to 8, which is too much.
#def classify(classifier, data, labeltype):
 #   return cross_val_predict(classifier, data, labeltype, cv = 8)

#Yguess = classify(regr, Xtrain, Ytrain)


print("Mean squared error: %.9f"
      % mean_squared_error(Ytest, Yguess))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Ytest, Yguess))
print('explaine variance: ',explained_variance_score(Ytest, Yguess))

print(np.corrcoef(Ytest, Yguess))
#print(stats.pearson3(Yguess,Yguess))

print("Y: ", Ytest)
for i in Ytest:
    print(i)
print("Yguess: ", Yguess)
for x in Yguess:
    print(x)