import pandas as pd
import pickle
import spacy
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import scipy as sp
import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
import re
import os

#Unpickle model
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),'wikipredict/voterS.pkl'), "rb") as f:
    voter = pickle.load(f)

nlp = spacy.load('en_core_web_sm')

#Unpickle lists for use in vectorize
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),'wikipredict/lexicon.pkl'),"rb") as f:
    lexicon = pickle.load(f)
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),'wikipredict/cols.pkl'),"rb") as f:
    column = pickle.load(f)

# lexicon = bowS.columns.tolist()
cols = column.tolist()

#Relative frequency of each POS tag
def counts(doc):
    count = Counter(([token.pos_ for token in doc]))
    countSum = sum(count.values())
    d = dict()
    d["ADJ"],d["ADP"],d["ADV"],d["AUX"],d["CONJ"],d["DET"],d["INTJ"],d["NOUN"],d["NUM"],d["PART"],d["PRON"],d["PROPN"],d["PUNCT"],d["SCONJ"],d["SYM"],d["VERB"],d["X"] = [0 for i in range(17)]
    for part, c, in count.items():
        d[part]=(c/countSum)
    return d

colsS = set(cols)

cV = CountVectorizer(vocabulary=lexicon) #We only want to count about 1200 words or so. Same 1200 words model was trained with.
def vectorize(sentence):
    doc = nlp(sentence)
    split = sentence.split()
    numWords = len(split)
    wordLength = sum([len(i) for i in split]) / max(1,numWords)
    d = counts(doc)
    
    vdf = pd.DataFrame(columns = cols)

    bow = cV.fit_transform([sentence]).toarray()[0]

    for i in d.items():
        vdf[i[0]] = [i[1] * 250]



    vdf["numWords"] = [numWords]
    vdf["wordLength"] = [wordLength]
    vdf = vdf.drop(columns = ["SPACE"])

    for i in range(len(lexicon)):
        vdf[lexicon[i]] = bow[i]

    vdf.fillna(0, inplace=True)
    return vdf.to_numpy().astype(np.uint8)[0]

def splitter(string): #Split into sentences
    if(len(string) > 601): #"Max 600 characters" (601 okay)
        string = string[0:600]
    string = string.replace(".",".~")
    string = string.replace("?","?~")
    string = string.replace("!","!~")
    strings = string.split("~")

    split = []
    for s in strings:
        if (s != " ") and (s != "") and (len(s.split()) > 0):
            split.append(s)

    return split

def predictor(sentence):
    predict = voter.predict_proba([vectorize(sentence)])[0][0]
    if(predict > .92): #RED
        return 0
    elif(predict > .5): #ORANGE
        return 1
    else: #GREEN: GOOD TONE
        return 2
