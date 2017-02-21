# capstoneflaskapp

The gensim files (LDA model, corpus, indexing, dict, etc.) were trained locally using reddit_doc_2017-02-16.dump.csv and added to the repo as temporary representation of gensim objects. The python script used to train the model and build the objects is "gensim model test building.py"

#Updates 2-20-2017: 
  (1) Jinja2 template using bootstrap and sidebar panel have been implemented.
  (2) Document similarity function utilizing scraped data has been put in place under /docsim
  
#TO-DO:
(1) Matt is working on hooking up the DB from sqlite3 into gensim and returning those objects for usage in the application.
(2) Fully flush out the word2vec model using doc2vec functions from our corpus
(3) Next steps are to take that and use queries for updates in realtime using update model function in gensim.
