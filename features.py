from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class SentenceLength(BaseEstimator, TransformerMixin):
    def abs_sentence_length(self, x):
        return len(x[0].split())

    def transform(self, X, y=None):
        return [[self.abs_sentence_length(x)] for x in X]

    def fit(self, X, y=None):
        return self


class AverageWordLength(BaseEstimator, TransformerMixin):
    def average_word_length(self, x):
        #print(np.mean([syllables(word) for word in x[0].split()]))
        return np.mean([len(word) for word in x[0]])

    def transform(self, X, y=None):
        return [[self.average_word_length(x)] for x in X]

    def fit(self, X, y=None):
        return self

class AverageSentenceLength(BaseEstimator, TransformerMixin):
    def average_sent_length(self, x):
        #print(np.mean([len(x[0])]))
        return np.mean([len(x[0].split())])

    def transform(self, X, y=None):
        return [[self.average_sent_length(x)] for x in X]

    def fit(self, X, y=None):
        return self

class AverageSentenceLengthWords(BaseEstimator, TransformerMixin):
    def average_sent_length_words(self, x):
        return np.mean([len(x[0].split())])

    def transform(self, X, y=None):
        return [[self.average_sent_length_words(x)] for x in X]

    def fit(self, X, y=None):
        return self


class Freq502(BaseEstimator, TransformerMixin):
    def freq2(self,x):
        freq50 = ['die', 'van', 'het', 'en', 'in', "'n", 'is', 'nie', 'te', 'wat', 'om', 'op', 'se', 'vir', 'sy', 'met',
                  'dit', 'hy', 'word', 'as', 'dat', 'aan', 'was', 'by', 'of', 'sal', 'hulle', 'the', 'kan', 'ek', 'diÈ',
                  'ons', 'ook', 'oor', 'maar', 'gesÍ', 'deur', 'na', 'daar', 'tot', 'date', 'moet', 'hul', 'gaan',
                  'jaar',
                  'toe', 'haar', 'teen', 'sÍ', 'meer']
        tokens = x[0].split()
        #print(len(tokens))
        #print((len(list(set(freq50) & set(tokens)))+0.1)/len(tokens))
        return (len(list(set(freq50) & set(tokens)))+1)/len(tokens)

    def transform(self, X, y=None):
        return [[self.freq2(x)] for x in X]

    def fit(self, X, y=None):
        return self


class type_token(BaseEstimator, TransformerMixin):
    def type_token_ratio(self,x):
        uniquewords = dict()
        words = 0
        for word in x[0].split():
            words += 1
            if word in uniquewords:
                uniquewords[word] += 1
            else:
                uniquewords[word] = 1
        TTR = len(uniquewords) / words
        return TTR

    def transform(self, X, y=None):
        return [[self.type_token_ratio(x)] for x in X]

    def fit(self, X, y=None):
        return self


