from gensim.models import word2vec
import chardet
import pandas as pd
from datetime import datetime
import time
import sys
# reload(sys)
sys.setdefaultencoding( "utf-8" )
pas_col = ['id', 'date', 'time', 'station','transport', 'money', 'discount']

# time format HH:MM:SS change into seconds
def trans_seconds(t):
    t_date = t.encode("utf-8")
    lists = t_date.split(':', 2)
    seconds = int(lists[0]) * 3600 + int(lists[1]) * 60 + int(lists[2])
    return seconds

# remove the line identifier in every station name
def trans_station(st):
    stemp = st.rstrip().encode("utf-8")
    st_list = stemp.split("线", 1)
    return st_list[1]


def daily_process(x):
    if x < 10:
        pas_data = pd.read_csv('/data/SPTCC-2015040' + str(x) + '.csv', names=pas_col, encoding='GB2312')
    else:
        pas_data = pd.read_csv('/data/SPTCC-201504' + str(x) + '.csv', names=pas_col, encoding='GB2312')
    print("transport is not subway:", len(pas_data.loc[pas_data.transport != u"地铁", :]))
    #filter records whose transport is not subway
    indexs = list(pas_data.loc[pas_data.transport != u"地铁", :].index)
    pas_data = pas_data.drop(indexs).reset_index()
    pas_data = pas_data.drop(['discount', 'transport'], 1)
    #change time format
    pas_data['time'] = pas_data['time'].apply(trans_seconds)
    pas_data = pas_data.drop(['index'], 1)
    #----drop records with abnormal user behavior---
    pas_data = pas_data.sort_values(by=['id', 'time']).reset_index()
    pas_data = pas_data.drop(['index'], 1)
    cid = pas_data.loc[0, 'id']
    pas_data_size = pas_data.groupby(['id']).size().reset_index()
    pas_data_size.columns = ['id', 'counts']
    # total times of entering and exiting the stations is an odd number
    wrongid = list(pas_data_size.loc[pas_data_size.counts % 2 != 0, 'id'])
    # first entrance the transaction amount is not 0
    if pas_data.loc[0, 'money'] != 0:
        wrongid.append(pas_data.loc[0,'id'])
    for i in range(1,len(pas_data)):
        if pas_data.loc[i, 'id'] != cid:
            cid = pas_data.loc[i, 'id']
            if pas_data.loc[i, 'money'] != 0:
               wrongid.append(cid)
    # print(len(wrongid))
    # print(wrongid[0:5])
    wrong_indexs=[]
    for j in range(len(wrongid)):
        w=list(pas_data.loc[pas_data.id == wrongid[j], :].index)
        wrong_indexs.extend(w)
    pas_data = pas_data.drop(wrong_indexs).reset_index()
    return pas_data

#Corpus extraction
def extract(df):
    station_corpus = []
    tindex = 0
    while True:
        if tindex >= len(df):
            break
        sentence = []
        pasid = df.loc[tindex, 'id']
        sentence.append(df.loc[tindex, 'station'])
        #print type(temp[0])
        tindex += 1
        if tindex >= len(df):
            break
        while True:
            if tindex >= len(df) or df.loc[tindex,'id'] != pasid:
                break
            sentence.append(df.loc[tindex, 'station'])
            tindex += 1
        station_corpus.append(sentence)
        return station_corpus

pas = pd.DataFrame()
# integrate the 30-day metro records
for i in range(1,31):
     #discount=[4,5,6,11,12]
     #if i not in discount:
    p = daily_process(i)
    p = p.drop(['index'],1)
    pas = pas.append(p)

pas=pas.reset_index()
pas['station'] = pas['station'].apply(trans_station)
print(pas.head(20))
#sort each passenger by time
pas = pas.sort_values(by=['id','date','time'])
pas = pas.reset_index()
pas = pas.drop(['level_0','index'],1)
station_corpus=extract(pas)
print(station_corpus[0:5])

#station embedding(word2vec)
model4_500 = word2vec.Word2Vec(station_corpus,workers=8,window=4,size=500,min_count=4,iter=30)

# station name in the corpus
st=pas.groupby(['station']).size().reset_index()
st.columns=['station','counts']
print(st.head(5))
print(len(st))
station_list=[]
for i in range(len(st)):
     station_list.append(st.loc[i,'station'])

# show top 10 similar stations and similarity
def show_similar(sim_st):
    for j in range(len(sim_st)):
        print(sim_st[j][0], sim_st[j][1])

zjgk = model4_500.most_similar('张江高科')
show_similar(zjgk)
model4_500.save('/data/word2vec/model30days/modelwin4')

#station & represent vector
station_vec=[]
for i in range(len(station_list)):
    station_vec.append(list(model4_500.wv[station_list[i]]))
data={'station':station_list,'vector':station_vec}
st_vec=pd.DataFrame(data)
for i in range(len(station_vec[0])):
    st_vec['vec'+str(i)]=st_vec['vector'].apply(lambda x: x[i])
st_vec=st_vec.drop(['vector'],1)
st_vec.to_csv('/data/word2vec/model30days/stavec.csv', header=False, index=False)

#save to pickle
st_pic=[]
for i in range(len(station_list)):
    temp = model4_500.wv[station_list[i]]
    st_pic.append(temp)
data3 = {'station':station_list,'vector':st_pic}
st_str=pd.DataFrame(data3)
st_str.to_pickle('/data/word2vec/model30days/stavec.pkl')

#客流潮汐性
# res=[0 for i in range(108)]
# for i in range(108):
#     t=18000+i*600
#     res[i]=len(xjhin.loc[xjhin.time>=t,:].loc[xjhin.time<t+600])