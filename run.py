from scrapWeb import *
#
url_list = extractUrlMysql()
#
##for i in range(0, len(url_list), 500):
##    start = i
##    if (i + 500) < len(url_list):
##        stop = i + 499
##    else:
##        stop = len(url_list)
##
##    url = url_list[i]
##    text_lst = getUrlText(url_list)
##    text_lst = cleanText(text_lst)
##    with open('other/text_lst{0}.pickle'.format(str(i)), 'wb') as handle:
##        pickle.dump(text_lst, handle)
#
with open('other/text_lst.pickle', 'rb') as handle:
    text_lst = pickle.load(handle)
#text_lst = cleanText(text_lst)
df, X = clusterText(text_lst, url_list, n=7)
dic = getClusteredText(df, 7)
fin_dic = labelLda(dic,2)




