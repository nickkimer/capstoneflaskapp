from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('reddit.dict')
corpus = corpora.MmCorpus('reddit.mm')

print(corpus)
print(dictionary)
