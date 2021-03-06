"""
This program scores the digitized cloze tests automatically
"""

import sys
import numpy as np


def main():
    with open("testtexts.txt", "r", encoding='latin-1') as file:
        number_of_words_in_sent = 0
        zin_score = 0
        sentence = []
        #number_of_words_in_version = 0
        #number_of_correct_words = 0
        avg = []
        for line in file:

            line = line.rstrip()
            if line == "":
                try:
                    zin_score_totaal = zin_score / number_of_words_in_sent
                    number_of_words_in_sent = 0
                    print(zin_score_totaal,"\t"," ".join(sentence))
                    zin_score=0
                    sentence = []
                    avg = []
                except:
                    continue

            else:
                number_of_words_in_version = 0
                number_of_correct_words = 0
                answer_list = line.split("\t")
                number_of_words_in_sent = number_of_words_in_sent + 1
                sentence.append(answer_list[0])
                for item in answer_list[1:]:
                    if item.lower() == answer_list[0].lower():
                        number_of_correct_words = number_of_correct_words + 1
                        number_of_words_in_version = number_of_words_in_version  + 1
                    elif item.lower() == "missing":
                        continue
                    else:
                        number_of_words_in_version = number_of_words_in_version + 1
                #print("verson", number_of_words_in_version, "correct", number_of_correct_words)
                #print("correct words = ", number_of_correct_words, "total number of words in version =  ", number_of_words_in_version)

                try:
                    word_score = number_of_correct_words/number_of_words_in_version
                except:
                    word_score = 0
                #print("word score = ", word_score)

                zin_score = zin_score + word_score
                #print("zin_score = ", zin_score)
main()