# -*- coding: utf-8 -*-
'''
Read already constructed db into gensim corups element
*** Currently using sample database ***

@author: Nick, Matt
'''

import gensim
from gensim import corpora, similarities
import sqlite3
import nltk
import pandas as pd
from nltk.corpus import stopwords

### DB comments to list
conn = sqlite3.connect('mitre_2_full.db')
conn.row_factory = lambda cursor, row: row[0]
cur = conn.cursor()
documents = cur.execute('SELECT body FROM documents_copy').fetchall()


# documents = docs.doc_str.tolist()

# remove common words and tokenize
stoplist = set(stopwords.words('english'))
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
# pprint(texts)

# Word2Vec model
#nltk.download()
#texts = unicode(texts, 'utf-8')
model = gensim.models.Word2Vec(texts, min_count = 1)
model.save('word2vec_reddit.model')

# Create dictionary
dictionary = corpora.Dictionary(texts)
dictionary.save('reddit.dict')  # store the dictionary, for future reference
print(dictionary)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('reddit.mm', corpus)

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('reddit.dict')
corpus = corpora.MmCorpus('reddit.mm') # comes from the first tutorial, "From strings to vectors"
#print(corpus)

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
lda.save('lda_reddit.model')
#print(lda)

index = similarities.MatrixSimilarity(lda[corpus]) # transform corpus to LDA space and index it
index.save('reddit.index')
index = similarities.MatrixSimilarity.load('reddit.index')


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
# print(sims[1:10])
# print(documents[751])
