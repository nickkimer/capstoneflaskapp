# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:40:00 2017

@author: Nick
"""

from gensim import corpora, models, similarities
import pyLDAvis

dictionary = corpora.Dictionary.load('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/capstoneflaskapp/models/FINAL416.dict')
corpus = corpora.MmCorpus('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/capstoneflaskapp/models/FINAL416.mm')
lda = models.ldamodel.LdaModel.load('C:/Users/Nick/Desktop/DSI Spring 2017/capstone/capstoneflaskapp/models/FINAL50_416.model')

import pyLDAvis.gensim

p = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(p, 'C:/Users/Nick/Desktop/DSI Spring 2017/capstone/capstoneflaskapp/pyldavis_416.html')
