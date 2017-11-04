import requests
from bs4 import BeautifulSoup

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
        
        #print(len(url_lst)-len(ret_text))
        #print(len(ret_text))
        
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
    Will return list of strings(text from the URLs)
    """
    print("Obtaining list of frequent words")
    
    #Intialising list for dataframes
    df_con = []
    
    #Extracting words and getting count of the words and adding to a dataframe
    
    del_words = ['thi', 'ymy']#list to be ommited from analysis
    stop_words = set(stopwords.words("english"))
    stop_words.update(del_words)
    
    for t in text_lst:
        
        if type(t) != str:
         t = t.decode("UTF-8").encode('ascii','ignore')
         
        t = html.unescape(t)# get rid of the html tags
        t = re.sub(r'[^a-zA-Z0-9 ]',r' ',t)
        t = re.sub(r'[0-9+]',r' ',t)
        
        text = TextBlob(t)
        text = text.words.singularize()
        text = (t.lower() for t in text)
        text = (t for t in text if t not in exclude)
        text = [t for t in text if t not in stop_words]
        
        #Getting word count
        wordsFiltered = {}
        for w in text:
            #gets rid of single alphabets
            if len(w) == 1:
                continue
            if w in wordsFiltered:
                wordsFiltered[w] = wordsFiltered[w] +1
            else:
                wordsFiltered[w] = 1
                
            
        df = pd.DataFrame(list(wordsFiltered.items()), columns= ['word', 
                          'counts'])
        df_con.append(df)
    
    assert len(df_con) == len(text_lst)
        
    #Combine all the URLs' frequently used words above the given quantile
    df_unfinal = pd.concat(df_con, axis =0)
    assert pd.notnull(df_unfinal).all().all()
    
    #Combine all the common words
    if df_unfinal.word.value_counts()[0] != 1:
        print("*****There are common words****")
        df_unfinal['counts'] = df_unfinal.groupby('word')['counts'].transform('sum')
        df_unfinal = df_unfinal.drop_duplicates(subset = 'word')
    
    assert df_unfinal.word.value_counts()[0] == 1
    
    quant= df_unfinal.quantile(quart)
    ret_df = df_unfinal[df_unfinal['counts']>quant.loc['counts']]
    
    assert ret_df.word.dtype == object
    assert ret_df.counts.dtype == 'int64'
    
    return ret_df

###############################################################################

def getSentiment(txt_lst):
    con_str = "".join(txt_lst)
    sent = TextBlob(con_str)
    print("The Polarity is : " + str(sent.sentiment.polarity) +" and the subjectivity is "
          + str(sent.sentiment.subjectivity))

###############################################################################

url_lst = extractUrlMysql()
text_lst = getUrlText(url_lst)
df = getFrequentWords(text_lst)
getSentiment(text_lst)
