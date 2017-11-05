import requests
from bs4 import BeautifulSoup

import nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenizer
from textblob import TextBlob
import html.parser 
import re

#import pandas as pd
#import mysql.connector
#
#from scrapWeb import extractUrlMysql, getUrlText

###############################################################################
r = requests.get("https://docs.python.org/2/library/re.html")
html_code = r.text
soup = BeautifulSoup(html_code, "lxml")

for script in soup(["script", "style"]):
    script.extract()

sample = soup.get_text()
sample = html.unescape(sample) #html_tags =  re.findall(r'&[a-z]+',sample)

#Converting the text to utf -8 
if type(sample) != str:
    sample = sample.decode("UTF-8").encode('ascii','ignore')


sample1 = re.sub(r'[^a-zA-Z0-9 ]',r' ',sample)
sample2 = re.sub(r'[0-9+]',r' ',sample1)


text = TextBlob(sample2)

#Extracting phrases
text0 = text.noun_phrases
tg = [t.lower() for t in text0]

freq_words = nltk.FreqDist(tg)


print(freq_words.most_common(15)) 

#
#y = set(tg)
#
#for w in tg:
#    if w in y:
#        print("NOT FOUND : " +w)

###############################################################################

#url_lst = extractUrlMysql()
#txt_lst = getUrlText(url_lst)
#
#letter = []
#for t in txt_lst:
#    #getting rid of html tags
#    sample = html.unescape(t) #html_tags =  re.findall(r'&[a-z]+',sample)
#
#    #Converting the text to utf -8 
#    if type(sample) != str:
#        sample = sample.decode("UTF-8").encode('ascii','ignore')
#    
#    text = TextBlob(sample)
#    
#    #Extracting phrases
#    text0 = text.

###############################################################################
#from nltk.corpus import wordnet
#
#syns = wordnet.synsets("mysql")
#print(syns)
