
import requests
from bs4 import BeautifulSoup
import pandas as pd

import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

import mysql.connector

import html
import re

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


import matplotlib.pyplot as plt
import scipy.io

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import gensim

###############################################################################

def extractUrlMysql():
    
    """Extracts the url history from the mysql database and returns a set of url names"""
    
    print('Extracting Url History')
    conn = mysql.connector.connect(user = 'root',password='danekane',host='localhost',database = 'url_list')
    cursor = conn.cursor()
    cursor.execute("SELECT urls FROM url_names")
#    
#    url_list = cursor.fetchall()

    url_list = []    
    for i in range(500):
        url = cursor.fetchone()
        url_list.append(url)
    
    ret_lst = [i[0] for i in url_list]
    ret_lst = list(set(ret_lst))
    
    return ret_lst

################################################################################

def getUrlText(url_lst):
    """
    url_lst - List of URLs to be processed
    Will return list of strings(text from the URLs)
    """
    
    print("Retriving text")
    #Initialising array for return
    ret_text = []
    
    rem_url = []
    for url in url_lst:
        #Get text from Url
        try: 
            r = requests.get(url)
            html_code = r.text
        
        except:
            print("URL not reached: " + url)
            rem_url.append(url)
            ret_text.append("")
            continue 
        
        #Format the text for extraction
        soup = BeautifulSoup(html_code, "lxml")
        
        for script in soup(["script", "style"]):
            script.extract()
        
        ret_text.append(soup.get_text())
    
    ret_text = list(filter(None, ret_text))
    
    for url in rem_url:
        url_lst.remove(url)
    
    assert len(url_lst) == len(ret_text)
    
    return ret_text, url_lst

###############################################################################

def getFrequentWords(text_lst):
    """
    text_lst - List of strings to be processed
    exclude - List of words to be pervented
    quart - Quartile for words
    Will return a list of frequent words(tuples)
    """
    print("Obtaining list of frequent words")
    
    #Concat all the texts
    concat_text = " ".join(text_lst)

    if type(concat_text) != str:
     t = concat_text.decode("UTF-8").encode('ascii','ignore')
     
    t = html.unescape(concat_text)# get rid of the html tags
    t = re.sub(r'[^a-zA-Z0-9 ]',r' ',t)
    t = re.sub(r'[0-9+]',r' ',t)
    
    del_words = ['thi', 'ymy','www','http','com','searchsearch','view','play','nextplay','duration','ago']#list to be ommited from analysis
    stop_words = set(stopwords.words("english"))
    stop_words.update(del_words)
    
    text = TextBlob(t)
    text = text.words.singularize()
    text = (t.lower() for t in text)
    text = [t.strip() for t in text if t not in stop_words and len(t) != 1]
        
    word_freq = nltk.FreqDist(text)
    ret_freq_words = word_freq.most_common(500)

    return ret_freq_words

###############################################################################

def cleanText(text_lst):
    
    print('Cleaning text')
    ret_text_lst = []
    
    for t in text_lst:
        
        if type(t) != str:
         t = t.decode("UTF-8").encode('ascii','ignore')
         
        t = html.unescape(t)# get rid of the html tags
        t = re.sub(r'[^a-zA-Z0-9 ]',r' ',t)
        t = re.sub(r'[0-9+]',r' ',t)
        
        del_words = ['thi', 'ymy','www','http','com','searchsearch','view','play','nextplay','duration','ago']#list to be ommited from analysis
        stop_words = set(stopwords.words("english"))
        stop_words.update(del_words)
        
        text = TextBlob(t)
        text = text.words.singularize()
        text = (t.lower() for t in text)
        text = [t.strip() for t in text if t not in stop_words and len(t) != 1]
        
        text = " ".join(text)
        ret_text_lst.append(text)
        
    return ret_text_lst

###############################################################################

def clusterText(text_lst, url_lst, n = 5):
    """This function will cluster the given text files"""
    
    print("Clustering URLs")
    
    #vectorizer = TfidfVectorizer(max_df = 0.5, min_df = 2, stop_words = 'english')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_lst)
    
    km = KMeans(n_clusters = n, init = 'k-means++', max_iter = 100, n_init = 1, verbose = True, random_state = 42)
    km.fit(X)

    y = list(km.labels_)
    clusters = list(zip(y, url_lst, text_lst))
    ret_clusters = pd.DataFrame(clusters, columns = ['cluster_no','url_name','text_lst'])
    
    return ret_clusters, X

###############################################################################

def plotIneClus(X,n):
    
    ret_ine = []
    
    for i in range(2,n):
        km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 100, n_init = 1, verbose = True, random_state = 42)
        km.fit(X)
        ret_ine.append(km.inertia_)
    
    plt.scatter(range(2,n), ret_ine)
    plt.show()

###############################################################################
    
def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111)
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
#    ax.invert_yaxis()
#    ax.set_aspect('equal')
#    ax.set_xticks([])
#    ax.set_yticks([])
    
    print(type(ax))
    
    return ax

###############################################################################
    
def plot_sparse(X):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spy(X,precision=0.01, markersize=0.5)
    ax.set_xlim(0, X.shape[1])
    ax.set_ylim(0, X.shape[0])
    ax.set_aspect('auto')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
###############################################################################

def clusterFreWords(df, n):
    
    ret_lst = []
    
    for i in range(n):
        df = df[df['cluster_no'] == i]
        tex_lst = list(df['text_lst'].values)
        fre = getFrequentWords(tex_lst)
        ret_lst.append(fre[0:5])
        
    return ret_lst

###############################################################################
    
def getClusteredText(df, n):
    ret_dict = {}
    
    for i in range(n):
        df1 = df[df['cluster_no'] == i]
        tex_lst = list(df1['text_lst'].values)
        ret_dict[i] = tex_lst
    
    return ret_dict

###############################################################################
    
def labelLda(dic, n):
    """
    This function will return a dictionary with top 5 words for each cluster
    dic - Clustered dictionary
    n - Number of topics
    
    """
    
    vectorizer = CountVectorizer()

    fin_dic = {}
    for k,v in dic.items():
        X = vectorizer.fit_transform(v)
        vocab = vectorizer.get_feature_names()
        model = LatentDirichletAllocation(n_components=n, random_state=100)
        id_topic = model.fit_transform(X)
        
        
        topic_words = {}
        for topic, comp in enumerate(model.components_):    
            word_idx = np.argsort(comp)[::-1][:5]
            topic_words[topic] = [vocab[i] for i in word_idx]
            fin_dic[k] =  topic_words
    
    return fin_dic

###############################################################################
    
def gensimLda(dic, n):
    
    fin_dic = {}
    
    for k,v in dic.items():
        texts = [[token for token in text.split()] for text in v]
        dictionary = gensim.corpora.Dictionary(texts)# Creates a dictionary of words
        #texts = [[t.split(" ")] for t in v]
        corpus = [dictionary.doc2bow(text) for text in texts]#Bag of words for texts
        
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=n, id2word = dictionary, passes=20)
        fin_dic[k] = ldamodel.print_topics(num_topics=n, num_words=3)
        
    return fin_dic

        