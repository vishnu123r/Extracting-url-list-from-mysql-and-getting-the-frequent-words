import requests
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenizer
from textblob import TextBlob
import html.parser 
import re

import pandas as pd
import mysql.connector

from scrapWeb import extractUrlMysql, getUrlText

###############################################################################
#r = requests.get("https://docs.python.org/2/library/re.html")
#html_code = r.text
#soup = BeautifulSoup(html_code, "lxml")
#
#for script in soup(["script", "style"]):
#    script.extract()
#
#sample = soup.get_text()

###############################################################################

url_lst = extractUrlMysql()
txt_lst = getUrlText(url_lst)

letter = []
for t in txt_lst:
    #getting rid of html tags
    charNo1 = len(t)
    sample = html.unescape(t) #html_tags =  re.findall(r'&[a-z]+',sample)
    charNo2 = len(sample)
    
    charNo = charNo1 - charNo2
    letter.append(charNo)
    
    
    #Converting the text to utf -8 
#    if type(sample) != str:
#        sample = sample.decode("UTF-8").encode('ascii','ignore')
    
    #text = TextBlob(sample1)
    
    #Extracting phrases
    #text0 = text.noun_phrases

