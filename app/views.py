from app import app
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from flask import Flask, jsonify, request, render_template, redirect,json, url_for
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

dictionary = corpora.Dictionary.load('./models/FINAL416.dict')
corpus = corpora.MmCorpus('./models/FINAL416.mm')
model = models.Word2Vec.load('./models/word2vec.model')


#lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
lda = gensim.models.LdaModel.load('./models/FINAL50_416.model')
#index = similarities.MatrixSimilarity.load('./static/reddit.index')
index = np.load('./models/FINALH50.npy')

@app.route('/home')
def home_page():
    return render_template("pyldavis_final50.html")


# route for handling the login page logic
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'guest':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect('/home')
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

@app.route('/savedoc', methods=['GET','POST'])
def addRegion():
    return render_template("save-doc.html")

@app.route('/docsim')
def my_form2():
    return render_template("my-form2.html")

@app.route('/docsim', methods=['GET','POST'])
def my_form_post2():
    if request.method=='POST':
        text_sim = request.form['text_sim']
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
        '''
        doc_topics = []
        for (i,j) in result_doc:
            doc_topics[i] = doc_topic_retriev(result_doc[0][0])
        '''
        templateData2 = {
        'result2':result_doc,
        'text_sim':text_sim,
        #'doc_topics2':doc_topics,
        }
    return render_template("my-form2.html",**templateData2)

#Get topics of docs in sim docsim
def doc_topic_retriev(doc):
    vec_bow = dictionary.doc2bow(doc.lower().split())
    topics = []
    topics = lda.get_document_topics(vec_bow)
    return topics


#Database writing and queries
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



@app.route('/similarity/<word1>/<word2>')
def similarity(word1, word2):
    # show the user profile for that user
    return jsonify({"similarity": model.similarity(word1, word2), "word1": word1, "word2": word2})

@app.route('/doesnt_match/<words>')
def doesnt_match(words):
    # show the user profile for that user
    word_list = words.split("+")
    return jsonify({"doesnt_match": model.doesnt_match(word_list), "word_list": word_list})

@app.route('/view/<id>')
def view_document(id):
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


@app.route('/add', methods=['GET','POST'])
def add_entry():

    debug = ""

    if request.method=='POST':
        debug = str(request.form['Question_ID']) + str(request.form['Question']) + str(request.form['Date']) + str(request.form['Analyst_Name']) + str(request.form['Response'])
        new = [None] * 7
        new[0] = 'New'
        new[1] = request.form['Question_ID']
        new[2] = request.form['Date']
        new[3] = request.form['Question']
        new[4] = request.form['Analyst_Name']
        new[5] = request.form['Response']
        new[6] = 'NA'

        with sql.connect('static/mitre_2_full.db') as conn:
            cur = conn.cursor()
            cur.execute('INSERT INTO documents_copy VALUES(?,?,?,?,?,?,?)',(new[0],new[1],new[2],new[3],new[4],new[5],new[6]))
            conn.commit()

    templateData={"debug":debug}
    return render_template("data_entry.html", **templateData)

@app.route('/topics')
def list_topics():
    topics = lda.show_topics(num_topics=-1, num_words=5, formatted =True)
    reg = re.compile('.+\\"(\\w+)\\"')

    words = [None] * 50
    whole = []
    for (i,j) in topics:
        temp = [None] * 6
        words[i] = j.split('+')
        temp[0] = i + 1
        temp[1] = re.findall(reg, words[i][0])
        temp[2] = re.findall(reg, words[i][1])
        temp[3] = re.findall(reg, words[i][2])
        temp[4] = re.findall(reg, words[i][3])
        temp[5] = re.findall(reg, words[i][4])
        whole.append(temp)

    topicsData = {'topicsData': whole}
    return render_template("topics.html", **topicsData)

@app.route('/visuals')
def show_visuals():
    # associated = find_associated()
    filename = 'static/js/nodes.json'
    # generate_network_file(associated, filename)
    # terms = get_topic_terms(0, lda, dictionary)


    # with sql.connect('static/mitre_2_full.db') as conn:
    #     cur = conn.cursor()
    #     result = cur.execute('SELECT * FROM doc_topic')
    # debug = gensim.matutils.sparse2full(result.fetchall(),lda.num_topics)

    dense_doc_topic = create_dense_doc_topic()
    associations = find_associated(dense_doc_topic)
    filename = 'static/js/nodes.json'
    generate_network_file(associations, filename)
    templateData = {'debug':associations}
    return render_template("visuals.html", **templateData)

def build_doc_topics():
     with sql.connect('static/mitre_2_full.db') as conn:
        cur = conn.cursor()
        for each_doc in enumerate(corpus):
            doc_id = each_doc[0]
            topics = lda[each_doc[1]]
            for each_topic in topics:
                topic_id = each_topic[0]
                percent = each_topic[1]
                cur.execute('INSERT INTO doc_topic VALUES(?,?,?)',(doc_id, topic_id, percent))

def get_top_docs(topic_id):
    with sql.connect('static/mitre_2_full.db') as conn:
        cur = conn.cursor()
        result = conn.execute('''SELECT doc_id FROM doc_topic WHERE topic_id == (?) ORDER BY percent DESC LIMIT 5''', (str(topic_id),))

    return result.fetchall()

def create_dense_doc_topic():
    with sql.connect('static/mitre_2_full.db') as conn:
        cur = conn.cursor()
        result = cur.execute('SELECT * FROM doc_topic')
        threshold = .2
        dense_doc_topic = np.zeros([corpus.num_docs, lda.num_topics])
        for each in result.fetchall():
            doc_id = each[0]
            topic_id = each[1]
            percent = each[2]
            if percent > threshold:
                dense_doc_topic[doc_id,topic_id] = 1

        return dense_doc_topic

def find_associated(doc_topic):
    # calculate marginal probabilities
    marginal_prob = doc_topic.sum(axis=0)/corpus.num_docs

    topic_topic = np.zeros([lda.num_topics,lda.num_topics])
    # calculate joint probabilities
    for i in range(0,corpus.num_docs):
        for j in range(0,lda.num_topics):
            for k in range(0,lda.num_topics):
                if doc_topic[i,j] == 1 and doc_topic[i,k] ==1:
                    # if j != k:
                    topic_topic[j,k] = topic_topic[j,k] + 1

    topic_topic = topic_topic / corpus.num_docs

    # find associated topics
    associated = [[]] * lda.num_topics
    threshold = 1
    for i in range(0, lda.num_topics):
        for j in range(0, lda.num_topics):
            topic_topic[i,j] = topic_topic[i,j] / (marginal_prob[i] * marginal_prob[j])
            if topic_topic[i,j] > threshold:
                if associated[i] == []:
                    associated[i] = [j]
                else:
                    associated[i].append(j)

    return associated

def generate_network_file(associations, filename):

    with open(os.path.join(app.root_path, filename), 'w') as f:
        out = """{\n"nodes":[\n"""

        num_topics = len(associations)

        first = True
        for node_id in range(0 ,num_topics):
            if associations[node_id] != []:
                terms = get_topic_terms(node_id, lda, dictionary)
                if first:
                    out += """\t{\"id\":""" + str(node_id) + """,\"label\":\"""" + terms[0][0] + " " + terms[1][0] +"\"}"
                    first = False
                else:
                    out += """,\n\t{\"id\":""" + str(node_id) + """,\"label\":\"""" + terms[0][0] + " " + terms[1][0] +"\"}"

        out += """],\n"edges":["""
        first = True
        edge_id = 0
        for node_id in range(0,num_topics):
            for association in associations[node_id]:
                if node_id != association and node_id < association:
                    if first:
                        out += """\n\t{\"from\":""" + str(node_id) + """,\"to\":""" + str(association) + "}"
                        first = False
                    else:
                        out += """,\n\t{\"from\":""" + str(node_id) + """,\"to\":""" + str(association) + "}"
                    edge_id = edge_id +1
        out += """]\n}\n"""
        f.write(out)

def get_topic_terms(topicId, model, dictionary):
    topic_terms = []
    topic_terms = lda.get_topic_terms(topicId)
    topic_terms = list(map(lambda each: (dictionary.get(each[0]), each[1]), topic_terms))
    return topic_terms
