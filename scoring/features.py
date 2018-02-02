from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class AverageWordLength(BaseEstimator, TransformerMixin):
    def average_word_length(self, x):
        return np.mean([len(word) for word in x])

    def transform(self, X, y=None):
        return [[self.average_word_length(x)] for x in X]

    def fit(self, X, y=None):
        return self


class AverageSentenceLength(BaseEstimator, TransformerMixin):
    def average_sent_length(self, x):
        return np.mean([len(x)])


    def transform(self, X, y=None):
        return [[self.average_sent_length(x)] for x in X]

    def fit(self, X, y=None):
        return self

class AverageSentenceLengthWords(BaseEstimator, TransformerMixin):
    def average_sent_length_words(self, x):
        return np.mean([len(x.split())])


    def transform(self, X, y=None):
        return [[self.average_sent_length_words(x)] for x in X]

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
