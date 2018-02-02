import sys


with open('zinscores4.txt', encoding='ISO-8859-1') as f:

    documents = []
    labels = []
    for line in f:
        tokens = line.strip().split()
        documents.append(tokens[1:])
        labels.append(tokens[0])


with open('POS.AF.text1.txt', encoding='ISO-8859-1') as file:

    POSX = []
    POS = []
    for lines in file:
        if lines.rstrip() == "":
            continue
        else:
            all = lines.strip().split()
            POSX.append(all[0])
            POS.append(all[1])
    #print(POS)
for line in POS:
    print(line, end=" ")

#print(documents[0])



