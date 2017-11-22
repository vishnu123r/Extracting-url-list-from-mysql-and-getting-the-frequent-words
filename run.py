from scrapWeb import *

with open('text_lst.pickle', 'rb') as handle:
    text_lst = pickle.load(handle)

url_list = extractUrlMysql()
##text_lst = getUrlText(url_lst)
#text_lst = cleanText(text_lst)
df, X = clusterText(text_lst, url_list, n=5)
plot_sparse(X)



#ax = plot_coo_matrix(X)
#plt.pause(1000)
#ax.figure.show()



#jkj = []
#
#for i in range(5):
#    df1 = df[df['cluster_no'] == i]
#    tex_lst = list(df1['text_lst'].values)
#    fre = getFrequentWords(tex_lst)
#    jkj.append(fre[0:5])
    
    
#vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(text_lst)
#
#plotIneClus(X,100)

