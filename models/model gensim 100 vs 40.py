# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:15:02 2017

@author: Nick
"""
from gensim import corpora, models, similarities

import pandas as pd

data = pd.read_csv("C:/Users/Nick/Desktop/data_edit.csv",header=None, encoding = "ISO-8859-1")
data.columns = ['doc_str']
data.head(1)

import csv

with open('C:/Users/Nick/Desktop/data_edit.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)
    
    
    
final_list = []
for row in data.doc_str:
    row = row.split()
    final_list.append(row)
    
texts = final_list



dictionary = corpora.Dictionary(texts)
dictionary.save('C:/Users/Nick/Desktop/capstone3-26.dict')  # store the dictionary, for future reference
print(dictionary)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('capstone3-26.mm', corpus)  

dictionary = corpora.Dictionary.load('C:/Users/Nick/Desktop/capstone3-26.dict')
corpus = corpora.MmCorpus('C:/Users/Nick/Desktop/capstone3-26.mm') # comes from the first tutorial, "From strings to vectors"
#print(corpus)

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
lda.save('C:/Users/Nick/Desktop/lda_reddit3-26.model')
#print(lda)

#Create the 40 topic version of the model
lda40 = models.LdaModel(corpus, id2word=dictionary, num_topics=40)
lda40.save("C:/Users/Nick/Desktop/lda40_reddit3-26.model")

#index = similarities.MatrixSimilarity(lda[corpus]) # transform corpus to LDA space and index it
#index.save('C:/Users/Nick/Desktop/capstone3-26.index')
index = similarities.MatrixSimilarity.load('C:/Users/Nick/Desktop/capstone3-26.index')

#index40 = similarities.MatrixSimilarity(lda40[corpus]) # transform corpus to LDA space and index it
#index40.save('C:/Users/Nick/Desktop/capstone3-26_40.index')

#Here's the query
doc = """we do have experts here but i can add a bit about 
medieval piracy in the north sea navigation was more or 
less done by orienting on land features landmarks etc and
 thus the ships were in turn seen from the land pirates would 
 do short raids from less populated areas or hand around the 
 entrances to ports routes through sea gates at islands etc """
 
doc = 'largest european navy'
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lda = lda[vec_bow] # convert the query to LDA space
sims = index[vec_lda]

sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims[1:10])
print(data.doc_str[751])

list40 = [6943,6086,4817]
list100 = [3192,1388,6737]
    
for doc in list40:
    print(data.doc_str[doc],"\n")

for doc in list100:
    print(data.doc_str[doc],"\n")