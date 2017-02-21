'''
Read already constructed db into gensim corups element
*** Currently using sample database ***

'''

import gensim
from gensim import corpora
import sqlite3
import nltk

### DB comments to list
conn = sqlite3.connect('mitre_2.db')
conn.row_factory = lambda cursor, row: row[0]
cur = conn.cursor()
docs = cur.execute('SELECT comment FROM comments').fetchall()

### Clean text
### add more stopwords or find other list? nltk possibly
stoplist = set('reddit askhistorians a the of to in if so'.split())
texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in docs]

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(texts)
dictionary.save('reddit.dict')

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('reddit.mm', corpus)
