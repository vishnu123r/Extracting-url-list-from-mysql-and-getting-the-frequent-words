import nltk
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenizer
from textblob import TextBlob
import pickle

import gensim

import html
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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
    
    text = " ".join(text)
    gen_docs.append(text)
    
#dictionary = gensim.corpora.Dictionary(gen_docs)# Creates a dictionary of words
#dictionary.save('webscrap.dict')
#
#corpus = [dictionary.doc2bow(text) for text in gen_docs]#Bag of words for texts
#gensim.corpora.MmCorpus.serialize('webscrap.mm', corpus)
#
#tfidf = gensim.models.TfidfModel(corpus)
#index = gensim.similarities.Similarity('',tfidf[corpus], num_features= len(dictionary))
#
#sim_mat = [list(index[tfidf[c]]*100) for c in corpus]
    

vectorizer = TfidfVectorizer(max_df = 0.5, min_df = 2, stop_words = 'english')
X = vectorizer.fit_transform(gen_docs)


km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 100, n_init = 1, verbose = True)
km.fit(X)
print(km.labels_)

################################################################################