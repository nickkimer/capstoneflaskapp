# -*- coding: utf-8 -*-
'''
Cleaner

@author: Matt
'''

import re
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

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
        line = re.sub('[1|2|3|4|5|6|7|8|9|0]', '', line)

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

        words = [w for w in words if not w in stopwords.words("english")]

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
