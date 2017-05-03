#from app import app
import gensim, logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#from flask import Flask, jsonify, request, render_template, redirect, json, url_for
#import sqlite3 as sql
import numpy as np
import os, re
import gensim.matutils
from gensim import corpora, models, similarities
from pprint import pprint 

# Load in Models
dictionary = corpora.Dictionary.load('./models/FINAL.dict')
#corpus = corpora.MmCorpus('./models/FINAL.mm')
#model = models.Word2Vec.load('./models/word2vec.model')
#lda = gensim.models.LdaModel.load('./models/FINAL30.model')
#index = np.load('./models/FINALH30.npy')

# Dict Works
#for x in dictionary:
#    print (str(x) + '---' + dictionary[x])

pprint(dictionary.token2id)

#pprint(corpus)
#np.savetxt("corpus.csv", index[0:10,0:10], delimiter=",")
