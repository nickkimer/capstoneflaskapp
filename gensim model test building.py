# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:29:26 2017

@author: Nick
"""

import pandas as pd
from gensim import corpora

data = pd.read_csv("C:/Users/Nick/Downloads/reddit_doc_2017-02-16.dump.csv")

documents = data.doc_str.tolist()

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]
# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]
from pprint import pprint  # pretty-printer
#pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/capstone.dict')  # store the dictionary, for future reference
print(dictionary)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('capstone.mm', corpus)  

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/capstone.dict')
corpus = corpora.MmCorpus('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/capstone.mm') # comes from the first tutorial, "From strings to vectors"
#print(corpus)

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
lda.save('lda_reddit.model')
#print(lda)

#index = similarities.MatrixSimilarity(lda[corpus]) # transform corpus to LDA space and index it
#index.save('capstone.index')
index = similarities.MatrixSimilarity.load('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/capstone.index')


#Here's the query
doc = """we do have experts here but i can add a bit about 
medieval piracy in the north sea navigation was more or 
less done by orienting on land features landmarks etc and
 thus the ships were in turn seen from the land pirates would 
 do short raids from less populated areas or hand around the 
 entrances to ports routes through sea gates at islands etc """
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lda = lda[vec_bow] # convert the query to LDA space
sims = index[vec_lda]

sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims[1:10])
print(data.doc_str[751])

