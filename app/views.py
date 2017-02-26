from app import app
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from flask import Flask, jsonify, request, render_template, redirect,json, url_for
import sqlite3 as sql


sentences = [['first', 'sentence'], ['second', 'sentence'],['this','is','the','third','sentence'],['this','is','the','fourth','sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)

#This is the document similarity setup
from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('./static/reddit.dict')
corpus = corpora.MmCorpus('./static/reddit.mm')

#lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
lda = gensim.models.LdaModel.load('./static/lda_reddit.model')
index = similarities.MatrixSimilarity.load('./static/reddit.index')



@app.route('/')
def home_page():
    return render_template("template.html")

# route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'guest':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect('/')
    return render_template('login_test.html', error=error)

@app.route('/word2vec')
def blankword2vec():
    return render_template("my-form.html")

@app.route('/word2vec', methods=['GET','POST'])
def my_form_post():
    if request.method=='POST':
    	text = request.form['text']
    	#processed_text = text.upper()
    	processed_text = text
    	templateData = {#'title':'- We will display the most similar words from Word2Vec',
    	#'result':json.dumps(model.most_similar([processed_text]),indent=4,separators=(',', ': ')),
    	'result':model.most_similar([processed_text]),
        'text':text}
    return render_template("my-form.html",**templateData)

@app.route('/docsim')
def my_form2():
    return render_template("my-form2.html")

@app.route('/docsim', methods=['GET','POST'])
def my_form_post2():
    if request.method=='POST':
        text_sim = request.form['text_sim']
        #processed_text = text.upper()
        #processed_text = text_sim
        doc = text_sim
        vec_bow = dictionary.doc2bow(doc.lower().split())
        vec_lda = lda[vec_bow] # convert the query to LDA space
        sims = index[vec_lda]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        result_doc = sims[0:10]
        final = get_titles(result_doc)
        for i in range(0,10):
            result_doc[i] = result_doc[i] + (final[i],)
        templateData2 = {
        'result2':result_doc,
        'text_sim':text_sim
        }
    return render_template("my-form2.html",**templateData2)

#Database writing and queries
def get_titles(result_doc):
    con = sql.connect("./static/mitre_2_full.db")
    cur = con.cursor()
    titles=[0]*10
    indices = [0]*10

    for i in range(0,10):
        indices[i] = result_doc[i][0]
    for i in range(0,10):
        titles[i] = cur.execute('''SELECT title FROM documents WHERE rowid=?''',(indices[i],))
        titles[i] = titles[i].fetchall()
    return titles



@app.route('/similarity/<word1>/<word2>')
def similarity(word1, word2):
    # show the user profile for that user
    return jsonify({"similarity": model.similarity(word1, word2), "word1": word1, "word2": word2})

@app.route('/doesnt_match/<words>')
def doesnt_match(words):
    # show the user profile for that user
    word_list = words.split("+")
    return jsonify({"doesnt_match": model.doesnt_match(word_list), "word_list": word_list})

@app.route('/view')
def view_document():
    post_id = request.args.get('id')
    with sql.connect('static/mitre_2_full.db') as conn:
        cur = conn.cursor()
        result = conn.execute('''SELECT date, title, body, post_author FROM posts WHERE post_id=?''', (post_id,))
        result = result.fetchone()
        print(result)
        date = result[0]
        title = result[1]
        body = result[2]
        author = result[3]

        templateData = {'title':title, 'body':body, 'date':date, 'author':author} 
    return render_template("view_document.html", **templateData)

@app.route('/savedoc/<docid>/<qid>')
def savedoc(docid,qid):
    pass
