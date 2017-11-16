from scrapWeb import *

with open('text_lst.pickle', 'rb') as handle:
    text_lst = pickle.load(handle)

url_lst = extractUrlMysql()
#text_lst = getUrlText(url_lst)
#text_lst = cleanText(text_lst)
#df = clusterText(text_lst, url_lst, n=5)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_lst)

plotIneClus(X,9)
