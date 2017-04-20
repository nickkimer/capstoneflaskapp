import gensim, logging
import sqlite3 as sql
import numpy as np
import os, re
import gensim.matutils
from gensim import corpora, models, similarities

# Load in Models
dictionary = corpora.Dictionary.load('./models/FINAL416.dict')
corpus = corpora.MmCorpus('./models/FINAL416.mm')
model = models.Word2Vec.load('./models/word2vec.model')
lda = gensim.models.LdaModel.load('./models/FINAL50_416.model')
index = np.load('./models/FINALH50.npy')


#Get topics of docs in sim docsim
def get_top_docs(result_doc):
    with sql.connect('static/mitre_2_full.db') as conn:
        cur = conn.cursor()
        doc_topics_temp = []

        for i in range(0, 10):
            doc_id = result_doc[i][0]
            result = conn.execute('''SELECT topic_id, percent FROM doc_topic WHERE doc_id == (?) ORDER BY
                                    percent DESC LIMIT 5''', (int(doc_id),))
            temp = result.fetchall()
            doc_topics_temp.append(temp)
            if len(doc_topics_temp[i]) < 5:
                short = 5 - len(doc_topics_temp[i])
                empty_tup = (0, 0)
                for element in range(short):
                    doc_topics_temp[i].append(empty_tup)

    return doc_topics_temp

#Database writing and queries
def get_bodies(result_doc):
    con = sql.connect("./static/mitre_2_full.db")
    cur = con.cursor()
    bodies = [0]*10
    indices = [0]*10

    for i in range(0, 10):
        indices[i] = result_doc[i][0] + 1
    for i in range(0, 10):
        bodies[i] = cur.execute('''SELECT body FROM documents_copy WHERE rowid=?''', (indices[i],))
        bodies[i] = bodies[i].fetchall()
        #Showing the first 100 characters of a string
        #bodies[i] = bodies[i][0][0][0:100] + "..." - moved to above inside of the function
    return bodies

# Doc to Doc Similarity Routes
text_sim = 'navy marine ww-ii armed forces'
doc = text_sim
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lda = lda[vec_bow]
# query index
q = np.sqrt(gensim.matutils.sparse2full(vec_lda, lda.num_topics))
sims = np.sqrt(0.5 * np.sum((q - index)**2, axis=1))
#HOW MANY RESUlTS FOR SIMS?
sims = sorted(enumerate(sims), key=lambda item: -item[1])
result_doc = list(reversed(sims[-10:len(sims)]))

doc_topics = get_top_docs(result_doc)

final = get_bodies(result_doc)
for i in range(0, 10):
    result_doc[i] = result_doc[i] + (final[i][0][0],)
    result_doc[i] = result_doc[i] + (final[i][0][0][0:100] + "...",)

print(doc_topics)