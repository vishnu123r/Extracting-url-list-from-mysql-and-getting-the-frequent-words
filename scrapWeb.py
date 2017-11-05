import requests
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

import pandas as pd
import mysql.connector

import html
import re
###############################################################################

def extractUrlMysql():
    
    """Extracts the url history from the mysql database"""
    
    conn = mysql.connector.connect(user = 'root',password='danekane',host='localhost',database = 'url_list')
    cursor = conn.cursor()
    cursor.execute("SELECT urls FROM url_names")
    
#    url_list = cursor.fetchall()

    url_list = []    
    for i in range(10):
        url = cursor.fetchone()
        url_list.append(url)
    
    ret_lst = [i[0] for i in url_list]
    
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
    
    for url in url_lst:
        #Get text from Url
        try: 
            r = requests.get(url)
            html_code = r.text
        
        except:
            print("URL not reached: " + url)
            ret_text.append(" ")
            continue 
        
        #Format the text for extraction
        soup = BeautifulSoup(html_code, "lxml")
        
        for script in soup(["script", "style"]):
            script.extract()
        
        ret_text.append(soup.get_text())
    
    assert len(url_lst) == len(ret_text)
    
    return ret_text

###############################################################################

def getFrequentWords(text_lst, exclude = [], quart = 0.9):
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
    
    del_words = ['thi', 'ymy']#list to be ommited from analysis
    stop_words = set(stopwords.words("english"))
    stop_words.update(del_words)
    
    text = TextBlob(t)
    text = text.words.singularize()
    text = (t.lower() for t in text)
    text = (t for t in text if t not in exclude)
    text = [t.strip() for t in text if t not in stop_words and len(t) != 1]
        
    word_freq = nltk.FreqDist(text)
    freq_words = word_freq.most_common(500)

    return freq_words

###############################################################################

def getSentiment(txt_lst):
    con_str = "".join(txt_lst)
    sent = TextBlob(con_str)
    print("The Polarity is : " + str(sent.sentiment.polarity) +" and the subjectivity is "
          + str(sent.sentiment.subjectivity))

###############################################################################

url_lst = extractUrlMysql()
text_lst = getUrlText(url_lst)
freq_words = getFrequentWords(text_lst)
getSentiment(text_lst)

