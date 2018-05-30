"""
Bo Blankers
Master thesis 2018 

This program reads the import from several repositories:
NER - containing files which have identiefied NER in a file. The first column contains the words and the second their tags (Type of NER or OUT)
PC - Phrase Chunker repo. Contains files with words in the words column and their phrase tag in the second column
POS - POS tagged files. First column the words, second column their corresponding POS tag
TOKEN - tokenized versions of text file. One token per row.
SENTENCE - Text files separated per sentence. One sentence per line. 

Output of this program is a matrix filled with all characteristics count per file. 
"""
import sys
from collections import Counter
import os
from pandas import *
import numpy as np
from numpy import array
from numpy import sum
import csv

def get_tags_and_features(file, tag_and_feature_list):
    with open(file) as files:
        for line in files:
              try:
                   word_tag = line.split("\t")
                   tag = word_tag[1].split()
                   if tag[0] not in tag_and_feature_list:
                        tag_and_feature_list.append(tag[0])
                   else:
                        continue
              except:
                   continue
    return(tag_and_feature_list)

def get_counts(file):
    taglist, poslist, countdict = [], [], {}
    with open(file) as files:
        for line in files:
            try:
                word_tag = line.split("\t")
                tag = word_tag[1].split()
                taglist.append(tag[0])
            except:
                continue
        for i in taglist:
            y = taglist.count(i)
            countdict[i] = y
    return(countdict)



def frequency( d, title,n):
    frequent_set, not_frequent_set  = set([]), set([])
    frequent_list = ["freq"+str(title)]
    type_list_count = ["freq"+str(title)+"types"]
    # n is the number by which the total list (44566) is divided to create a subset of the list
    half_list_len = int(len(d)/n)

    for item in d[:half_list_len]:
        frequent_set.add(item[1])
    for item in d[half_list_len:]:
        not_frequent_set.add(item[1])

    for file in os.listdir("TOKEN"):
        file = "TOKEN/" + file
        type_list = []
        with open(file) as tokenized_file:
            freq_count = 0
            not_freq_count = 0
            for line in tokenized_file:
                if line.rstrip().lower() in frequent_set:
                    freq_count = freq_count + 1
                    if line.rstrip().lower() not in type_list:
                        type_list.append(line.rstrip().lower())
                else:
                    not_freq_count = not_freq_count + 1
            type_list_count.append(len(type_list))

            frequent_list.append(freq_count)
    return(array(frequent_list), array(type_list_count))

def wordcount():
    wordcounter_list = ["# of words"]
    haak_count = ["# of brackets in pair"]
    syllable_count = ["# of syllables"]
    for file in os.listdir("TEXT"):
        haak  = 0
        syl_count = 0
        syl_3 = 0
        file = "TEXT/" + file
        with open(file, "r", encoding='latin-1') as original_text:
            wordcounter = original_text.read().split()
            wordcounter_list.append(len(wordcounter))
            for item in wordcounter:
                #print(item)
                syl_count = syl_count + syllables(item)
                if syllables(item) > 3:
                    syl_3 = syl_3 + 1
                else:
                    syl_3 = syl_3
                syl_count_word = syl_count/len(wordcounter)
                char = item[0].split()
                if char[0] == "(":
                    haak = haak + 1
            #print(syl_3)
            print(syl_count)
            haak_count.append(haak)syl

    return(array(wordcounter_list), array(haak_count), array(syllable_count))

def sentence_count():
    sentence_counter_list = ["# of sentences"]
    for file in os.listdir("SENTENCE"):
        n = 0
        file = "SENTENCE/" + file
        with open(file, "r", encoding='latin-1') as original_text:
            for line in original_text:
                n = n +1
            sentence_counter_list.append(n)
    return(array(sentence_counter_list))

def syllables(word):
    count = 0
    vowels = 'aeiouy'
    word = word.lower().strip(".:;?!")
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if count == 0:
        count +=1
    return count


def character_count():
    char_count_list = ["number of characters"]
    for file in os.listdir("TOKEN"):
        file = "TOKEN/" + file
        with open(file) as tokenized_file:
            char_count = 0
            for line in tokenized_file:
                char_count = char_count + len(line)
        char_count_list.append(char_count)
    return(np.array(char_count_list))


def wps_count(x, sent, n):
    c = x/sent
    c = c.tolist()
    return([n]+c)


def sumColumn(m,column):
    return [sum(col) for col in zip(*m)]


def main():
    #first part is to initialize a matrix with all possible tags and features given by the NCHLT tagger

    tag_and_feature_list = []
    repositories = ["NER", "POS", "PC", "TOKEN", "SENTENCE"]
    for repo in repositories:
        for file in os.listdir(repo):
            file = repo + "/" + file
            tags_and_features = get_tags_and_features(file,tag_and_feature_list)

    matrix = [[0 for x in range(len(tags_and_features))] for x in range(26)]
    place = 0
    # fill in the first row with all possible tags
    for item in tags_and_features:
        matrix[0][place] = item
        place = place + 1

    """
    word and sentence counters
    """
    word_count, haak_counter, syllable_count  = wordcount()

    sentence_counter, character_counter =  sentence_count(), character_count()
    word_per_sentence = wps_count(np.array(word_count[1:], dtype = np.float), np.array(sentence_counter[1:], dtype = np.float), "words per sentence ")
    char_per_sentence = wps_count(np.array(character_counter[1:], dtype = np.float), np.array(sentence_counter[1:], dtype = np.float), "characters per sentence")
    char_per_word = wps_count(np.array(character_counter[1:], dtype = np.float), np.array(word_count[1:], dtype = np.float), "characters per word")

    for repo in repositories:
        text = 1
        text2 = 1
        for file in os.listdir(repo):
            file = repo+"/" + file
            count_dict_text = get_counts(file)
            for key in count_dict_text:
                try:
                    tag_look_up = tags_and_features.index(key)
                except:
                    continue
                matrix[text][tag_look_up] = float(count_dict_text[key])/float(word_count[text2])*100
            text = text + 1
            text2 = text2+1

    # second part is to extend the matrix will any other feature
    # first transform matrix to numpy matrix
    matrix = np.matrix(matrix, dtype=None)

    """
    call frequency function where n = number the original frequency list should be divided by (total = 44566 #
    so n = 44.566 gives a cut off at the 1000 most frequent words. First argument is the feature title (FreqX), second number the division. 
    """
    with open("Gerhard.Frekwensielys.txt", "r", encoding='latin-1') as file:
        reader = csv.reader(file, delimiter = "\t")
        d = list(reader)
        frequency_array50, frequency_array_type50 = frequency(d,50,891.3)
        frequency_array1000, frequency_array_type1000 = frequency(d,1000, 45.566)
        frequency_array2000, frequency_array_type2000  = frequency(d,2000,22.283)
        frequency_array3000, frequency_array_type3000  = frequency(d,3000,14.855)
        frequency_array5000, frequency_array_type5000  = frequency(d,5000,8.913)
        frequency_array10000, frequency_array_type10000  = frequency(d,10000,4.5566)
        frequency_array20000, frequency_array_type20000  = frequency(d,20000,2.2283)

    matrix = np.column_stack(
        [matrix,  word_count, haak_counter, character_counter, word_per_sentence, char_per_sentence, char_per_word, frequency_array1000, frequency_array2000, frequency_array3000, frequency_array5000,
         frequency_array10000, frequency_array20000, frequency_array50, frequency_array_type50, frequency_array_type1000, frequency_array_type2000,
         frequency_array_type3000, frequency_array_type5000, frequency_array_type10000, frequency_array_type20000])


    pandas.set_option('display.max_columns', None), pandas.set_option('display.max_rows', None), pandas.set_option('expand_frame_repr', False)
    pdmatrix = pandas.DataFrame(matrix)
    #print(pdmatrix)
   # print(pdmatrix.shape)

""" 
Uncomment if you want the final matrix to be printed 
    for row in matrix:
        print(row)
"""

main()
