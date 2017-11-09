from scrapWeb import *

#url_lst = extractUrlMysql()
#text_lst = getUrlText(url_lst)
df = clusterText(text_lst, url_lst, n=5)