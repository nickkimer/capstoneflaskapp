import gensim
import sqlite3 as sql
from gensim import models
import numpy as np
import os, re
import gensim.matutils




# sentences = [['first', 'sentence'], ['second', 'sentence'],['this','is','the','third','sentence'],['this','is','the','fourth','sentence']]
# train word2vec on the two sentences
# model = gensim.models.Word2Vec(sentences, min_count=1)

#This is the document similarity setup
from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('./static/FINAL.dict')
corpus = corpora.MmCorpus('./static/FINAL.mm')
model = models.Word2Vec.load('./static/word2vec_reddit.model')


#lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
lda = gensim.models.LdaModel.load('./static/FINAL50.model')
#index = similarities.MatrixSimilarity.load('./static/reddit.index')
index = np.load('./static/FINALH50.npy')

def get_bodies(result_doc):
    con = sql.connect("./static/mitre_2_full.db")
    cur = con.cursor()
    bodies=[0]*10
    indices = [0]*10

    for i in range(0,10):
        indices[i] = result_doc[i][0] + 1
    for i in range(0,10):
        bodies[i] = cur.execute('''SELECT body FROM documents_copy WHERE rowid=?''',(indices[i],))
        bodies[i] = bodies[i].fetchall()
        #Showing the first 100 characters of a string
        #bodies[i] = bodies[i][0][0][0:100] + "..." - moved to above inside of the function
    return bodies

def doc_topic_retriev(doc):
    vec_bow = dictionary.doc2bow(doc.lower().split())
    topics = []
    topics = lda.get_document_topics(vec_bow)
    return topics

text_sim = 'Largest Naval ships in british history during world war ii'
doc = text_sim
#vec_bow = dictionary.doc2bow(doc.lower().split())
#vec_lda = lda[vec_bow] # convert the query to LDA space
#sims = index[vec_lda]
#sims = sorted(enumerate(sims), key=lambda item: -item[1])
#result_doc = sims[0:10]
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lda = lda[vec_bow]
# query index
q = np.sqrt(gensim.matutils.sparse2full(vec_lda, lda.num_topics))
sims = np.sqrt(0.5 * np.sum((q - index)**2, axis=1))

#HOW MANY RESUlTS FOR SIMS?
sims = sorted(enumerate(sims), key=lambda item: -item[1])
result_doc = list(reversed(sims[-10:len(sims)]))

final = get_bodies(result_doc)
for i in range(0,10):
    result_doc[i] = result_doc[i] + (final[i][0][0],)
    result_doc[i] = result_doc[i] + (final[i][0][0][0:100] + "...",)

print(result_doc)

doc_topics = []
for i in range(0,10):
    doc_topics[i] = doc_topic_retriev(result_doc[0][2])

for i,tup in enumerate(result_doc):
    doc_topics[i] = doc_topic_retriev(result_doc[i][2])