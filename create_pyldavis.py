# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:40:00 2017

@author: Nick
"""

from gensim import corpora, models, similarities
import pyLDAvis

dictionary = corpora.Dictionary.load('./models/FINAL416.dict')
corpus = corpora.MmCorpus('./models/FINAL416.mm')
lda = models.ldamodel.LdaModel.load('./models/FINAL50_416.model')

import pyLDAvis.gensim

p = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(p, 'pyldavis_test.html')
