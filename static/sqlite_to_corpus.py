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
import numpy as np
from nltk.stem import PorterStemmer
import re
import csv
from bs4 import BeautifulSoup
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

stoplist = set(stopwords.words('english'))
stoplist.update(['reddit','www','com','askhistorians','subreddit','imgur'])
#print(stoplist)

# from cleaner.py import cleaner

### DB comments to list
conn = sqlite3.connect('mitre_2_full.db')
#conn.row_factory = lambda cursor, row: row[0]
cur = conn.cursor()
documents = pd.read_sql("SELECT * from documents_copy", conn)
#sql = 'SELECT body FROM documents_copy'
#documents = cur.execute('SELECT body FROM documents_copy').fetchall()
#documents = pd.read_sql('SELECT body FROM documents_copy', conn)
#documents = psql.read_frame(sql, conn)

# documents = docs.doc_str.tolist()
### Text Cleaning

def cleaner(inputdata):

    lines = []

    for row in inputdata:

        line = BeautifulSoup(str(inputdata), 'html.parser')

        line = re.sub(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', '_EM', str(line))
        line = re.sub(r'\w+:\/\/\S+', r'_U', line)

        line = line.replace('"', ' ')
        line = line.replace('\'', ' ')
        line = line.replace('\\xa0', ' ')
        line = line.replace('_', ' ')
        line = line.replace('-', ' ')
        line = line.replace('\n', ' ')
        line = line.replace('\\n', ' ')
        line = re.sub(' +',' ', line)
        line = line.replace('\'', ' ')

        line = re.sub(r'([^!\?])(\?{2,})(\Z|[^!\?])', r'\1 _BQ\n\3', line)
        line = re.sub(r'([^\.])(\.{2,})', r'\1 _SS\n', line)
        line = re.sub(r'([^!\?])(\?|!){2,}(\Z|[^!\?])', r'\1 _BQ\n\3', line)
        line = re.sub(r'([^!\?])\?(\Z|[^!\?])', r'\1 _Q\n\2', line)
        line = re.sub(r'([^!\?])!(\Z|[^!\?])', r'\1 _X\n\2', line)
        line = re.sub(r'([a-zA-Z])\1\1+(\w*)', r'\1_EL', line)
        line = re.sub(r'(\w+)\.(\w+)', r'\1\2', line)

        line = re.sub(r'([#%&\*\$]{2,})(\w*)', r'\1\2 _SW', line)

        line = re.sub(r' [8x;:=]-?(?:\)|\}|\]|>){2,}', r' _BS', line)
        line = re.sub(r' (?:[;:=]-?[\)\}\]d>])|(?:<3)', r' _S', line)
        line = re.sub(r' [x:=]-?(?:\(|\[|\||\\|/|\{|<){2,}', r' _BF', line)
        line = re.sub(r' [x:=]-?[\(\[\|\\/\{<]', r' _F', line)
        line = re.sub('[%]','', line)

        lines.append(line)

        phrases = re.split(r'[;:\.()\n]', line)
        phrases = [re.findall(r'[\w%\*&#]+', ph) for ph in phrases]
        phrases = [ph for ph in phrases if ph]

        words = []

        for ph in phrases:
            words.extend(ph)

        tmp = words
        words = []
        new_word = ''
        for word in tmp:
            if len(word) == 1:
                new_word = new_word + word
            else:
                if new_word:
                    words.append(new_word)
                    new_word = ''
                words.append(word)

        words = [w for w in words if not w in stoplist]

        lemmantizer = WordNetLemmatizer()

        tagged = []
        for t in words:
            t = t.lower()
            treebank_tag = pos_tag([t])
            tagged.append(treebank_tag)

        def get_wordnet_pos(tagged):
            if treebank_tag[0][1].startswith('J'):
                return wordnet.ADJ
            elif treebank_tag[0][1].startswith('V'):
                return wordnet.VERB
            elif treebank_tag[0][1].startswith('N'):
                return wordnet.NOUN
            elif treebank_tag[0][1].startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN

        postagged = []
        for t in tagged:
            newtag = t[0][0],get_wordnet_pos(t)
            postagged.append(newtag)

        lemmatized = []
        for t in postagged:
            lemmatized.append(lemmatizer.lemmatize(t[0],t[1]))

        for t in lemmatized:
            t = np.asarray(t)

        return(lemmatized)


texts = documents.body.apply(cleaner)

texts.to_csv('data.csv')

#print(texts.head())

'''
# remove common words and tokenize
stoplist = set(stopwords.words('english'))
#extra_stop = ['reddit','www','com','askhistorians','subreddit','imgur','you','i',"i'm",'would',"i've",'know','anyone','could','please','like','ok']
#stoplist = stoplist.update(extra_stop)
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
# quotes, hyphens, underscores,slashes
texts = [[word.replace('"', ' ').replace("\'", '').replace('_', ' ').replace('-', ' ') for word in text] for text in texts]

stemmer = PorterStemmer()

for text in texts:
    for word in text:
        word = (stemmer.stem(word))

### Other Option using other cleaning code
# for text in texts:
#    apply.cleaner()


from pprint import pprint  # pretty-printer
# pprint(texts)
'''

with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)
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

lda_100 = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
lda_100.save('lda_reddit_100.model')
#print(lda)
index = similarities.MatrixSimilarity(lda_100[corpus]) # transform corpus to LDA space and index it
index.save('reddit_100.index')
# index = similarities.MatrixSimilarity.load('reddit_100.index')

lda_40 = models.LdaModel(corpus, id2word=dictionary, num_topics=40)
lda_40.save('lda_reddit_40.model')
#print(lda)

index = similarities.MatrixSimilarity(lda_40[corpus]) # transform corpus to LDA space and index it
index.save('reddit_40.index')
# index = similarities.MatrixSimilarity.load('reddit_40.index')

'''

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
# print(documents[131])
'''
