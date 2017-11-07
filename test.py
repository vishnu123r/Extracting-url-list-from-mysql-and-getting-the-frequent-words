import nltk
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenizer
from textblob import TextBlob
import pickle

import gensim

import html
import re

###############################################################################

with open('text_lst.pkl', 'rb') as f:
    text_lst = pickle.load(f)

gen_docs = []
    
for t in text_lst:
    
    if type(t) != str:
     t = t.decode("UTF-8").encode('ascii','ignore')
     
    t = html.unescape(t)# get rid of the html tags
    t = re.sub(r'[^a-zA-Z0-9 ]',r' ',t)
    t = re.sub(r'[0-9+]',r' ',t)
    
    del_words = ['thi', 'ymy']#list to be ommited from analysis
    stop_words = set(stopwords.words("english"))
    stop_words.update(del_words)
    
    text = TextBlob(t)
    text = text.words.singularize()
    text = (t.lower() for t in text)
    text = [t.strip() for t in text if t not in stop_words and len(t) != 1]
    
    gen_docs.append(text)
    
dictionary = gensim.corpora.Dictionary(gen_docs)# Creates a dictionary of words
dictionary.save('webscrap.dict')

corpus = [dictionary.doc2bow(text) for text in gen_docs]#Bag of words for texts
gensim.corpora.MmCorpus.serialize('webscrap.mm', corpus)

tfidf = gensim.models.TfidfModel(corpus)
index = gensim.similarities.SparseMatrixSimilarity(tfidf[corpus], num_features= len(dictionary))

sims = index[tfidf[corpus[1]]]
print(list(enumerate(sims)))