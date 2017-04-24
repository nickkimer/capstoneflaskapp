import pandas as pd
from gensim import corpora
import numpy as np
import gensim.matutils
from gensim import corpora, models, similarities
import re

# Load in Models
dictionary = corpora.Dictionary.load('./models/FINAL.dict')
corpus = corpora.MmCorpus('./models/FINAL.mm')
lda = gensim.models.LdaModel.load('./models/FINAL30.model')
index = np.load('./models/FINALH30.npy')

topics = lda.show_topics(num_topics=-1, num_words=5, formatted=True)
reg = re.compile('\"(\\S+)\"')
words = [None] * 50
whole = []
for (i, j) in topics:
    print(j)
    temp = [None] * 6
    words[i] = j.split('+')
    temp[0] = i + 1
    temp[1] = re.findall(reg, words[i][0])
    temp[2] = re.findall(reg, words[i][1])
    temp[3] = re.findall(reg, words[i][2])
    temp[4] = re.findall(reg, words[i][3])
    temp[5] = re.findall(reg, words[i][4])
    print(temp)

print(whole)
