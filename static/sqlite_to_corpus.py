'''
Read already constructed db into gensim corups element
*** Currently using sample database ***

'''

import sqlite3
import nltk
import re
from nltk.corpus import stopwords
nltk.download("stopwords")

### DB comments to list
conn = sqlite3.connect('mitre_2.db')
conn.row_factory = lambda cursor, row: row[0]
cur = conn.cursor()
text = cur.execute('SELECT comment FROM comments').fetchall()

### Clean list
# NEED TEXT CLEANING

print(text[0:5])
