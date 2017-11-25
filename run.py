from scrapWeb import *

url_list = extractUrlMysql()
##text_lst = getUrlText(url_list)
##
##with open('other/text_lst_fin.pickle', 'wb') as handle:
##    pickle.dump(text_lst, handle)

with open('other/text_lst.pickle', 'rb') as handle:
    text_lst = pickle.load(handle)
#text_lst = cleanText(text_lst)
df, X = clusterText(text_lst, url_list, n=7)

dic = getClusteredText(df, 7)




#plot_sparse(X)
#vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(text_lst)
#
#plotIneClus(X,100)


