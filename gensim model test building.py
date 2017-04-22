# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:29:26 2017

@author: Nick
"""

#Updated 3/29/2017 Right before submission of the paper

import pandas as pd
from gensim import corpora
from nltk.corpus import stopwords
import re

data = pd.read_csv("C:/Users/Nick/Downloads/reddit_doc_2017-02-16.dump.csv")
documents = data.doc_str.tolist()

data = pd.read_csv("C:/Users/Nick/Desktop/training_corpus_final.csv",encoding = 'iso-8859-1')

#documents = data.doc_str.tolist()
data['body'] = data['body'].astype(str)
documents = data.body.tolist()

# remove common words and tokenize
#stoplist = set('for a of the and to in '.split())
stoplist = set(stopwords.words('english'))
stoplist.update(['reddit','www','com','askhistorians','subreddit','imgur','for','a','of','the','and','to','in','would','*','http','org','en','comment','comments','could','would','also','really','-','/','u','r',',','?','--','!','|','<','>','=','+','[',']','{','}','d','s','m','its','dont','wont','cant','im'])

for document in documents:
	for word in document.lower().split():
		word = re.sub(r'[^\w\s]','',word)

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
dictionary.save('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINAL.dict')  # store the dictionary, for future reference
print(dictionary)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINAL.mm', corpus)  

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINAL.dict')
corpus = corpora.MmCorpus('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINAL.mm') # comes from the first tutorial, "From strings to vectors"
#print(corpus)

lda100 = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
lda100.save('FINAL100.model')
#print(lda)
lda40 = models.LdaModel(corpus, id2word=dictionary, num_topics=40)
lda40.save('FINAL40.model')

lda50 = models.LdaModel(corpus, id2word=dictionary, num_topics=50)
lda50.save('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINAL50.model')



index100 = similarities.MatrixSimilarity(lda100[corpus]) # transform corpus to LDA space and index it
index100.save('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINAL100.index')
index100 = similarities.MatrixSimilarity.load('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINAL100.index')

index40 = similarities.MatrixSimilarity(lda40[corpus]) # transform corpus to LDA space and index it
index40.save('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINAL40.index')
index40 = similarities.MatrixSimilarity.load('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINAL40.index')

index50 = similarities.MatrixSimilarity(lda50[corpus]) # transform corpus to LDA space and index it
index50.save('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINAL50.index')
index50 = similarities.MatrixSimilarity.load('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINAL50.index')


#Here's the query
doc = """we do have experts here but i can add a bit about 
medieval piracy in the north sea navigation was more or 
less done by orienting on land features landmarks etc and
 thus the ships were in turn seen from the land pirates would 
 do short raids from less populated areas or hand around the 
 entrances to ports routes through sea gates at islands etc """
doc = "Japanese warfare asia 19th century world war2 pacific ocean front"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lda = lda50[vec_bow] # convert the query to LDA space
sims = index50[vec_lda]

sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims[1:10])
print(data.doc_str[3956])


#Testing Hellinger Distance Measure
#index = numpy.sqrt(gensim.matutils.corpus2dense(lda[corpus], lda.num_topics).T)
doc = """we do have experts here but i can add a bit about 
medieval piracy in the north sea navigation was more or 
less done by orienting on land features landmarks etc and
 thus the ships were in turn seen from the land pirates would 
 do short raids from less populated areas or hand around the 
 entrances to ports routes through sea gates at islands etc """

# Query Document into bag of words
import numpy as np
#from gensim.matutils import kullback_leibler, jaccard, hellinger
import gensim.matutils

vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lda = lda50[vec_bow] # convert the query to LDA space

# query index
q = np.sqrt(gensim.matutils.sparse2full(vec_lda, lda50.num_topics))
indexH50 = np.sqrt(gensim.matutils.corpus2dense(lda50[corpus], lda50.num_topics).T)
#np.save('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINALH50.npy',indexH50)
#indexH50 = np.load('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/FINALH50.npy')
# calculate similarity score from script implementation
sims = np.sqrt(0.5 * np.sum((q - indexH50)**2, axis=1))

#HOW MANY RESUlTS FOR SIMS? 
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(list(reversed(sims[-10:len(sims)])))
result_doc = list(reversed(sims[-10:len(sims)])) #Save the smallest distance results
print(data.doc_str[751])
#print(data.body[571])

# Coherence Model Evaluation

#Tutorial http://nbviewer.jupyter.org/github/dsquareindia/gensim/blob/a4b2629c0fdb0a7932db24dfcf06699c928d112f/docs/notebooks/topic_coherence_tutorial.ipynb
# Info http://www.kdnuggets.com/2016/07/americas-next-topic-model.html
# Visualizing topic distances http://nbviewer.jupyter.org/github/alexperrier/datatalks/blob/master/twitter/LDAvis.ipynb#topic=0&lambda=1&term=
# another example http://alexperrier.github.io/jekyll/update/2015/09/04/topic-modeling-of-twitter-followers.html
# The paper that shows c_v is the best

from gensim.models.coherencemodel import CoherenceModel
cm50 = CoherenceModel(model=lda50,texts=texts,dictionary=dictionary,coherence='c_v')
cm40 = CoherenceModel(model=lda40,texts=texts,dictionary=dictionary,coherence='c_v')
cm100 = CoherenceModel(model=lda100,texts=texts,dictionary=dictionary,coherence='c_v')
print(cm50.get_coherence())
print(cm40.get_coherence())
print(cm100.get_coherence())



#ABOVE THIS LINE IS UPDATED VERSION OF THIS FILE
#########################################################################################33
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
type(texts)
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

