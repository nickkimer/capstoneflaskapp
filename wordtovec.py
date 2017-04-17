'''
word2vec model creation
'''
import pandas as pd
from gensim import corpora
from nltk.corpus import stopwords

data = pd.read_csv("C:/Users/Nick/Desktop/training_corpus_final.csv",encoding = 'iso-8859-1')

data['body'] = data['body'].astype(str)
documents = data.body.tolist()

stoplist = set(stopwords.words('english'))
stoplist.update(['reddit','www','com','askhistorians','subreddit','imgur','for','a','of','the','and','to','in','would','*','http','org','en','comment','comments','could','would','also','really'])

texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

wordtovec = gensim.models.Word2Vec(texts, min_count = 1)
model.save('word2vec.model')
