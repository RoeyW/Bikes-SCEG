import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from  sklearn import preprocessing
import numpy as np
from analyze.reorgdata import callAllfile
from sklearn.preprocessing import OneHotEncoder
import json
import scipy.stats as sci

# feature:[lat,lon,weekday_checkin,weekday_checkout,weekend_checkin,weekend_checkout]
# k={20，30，40]
# output: cluster{station_id} to csv

# average check-in / check-out demand for stations in workday or holiday
def diffTwolist(l1,l2):
    return list(set(l1)-set(l2))

def avgDemand(demand_type,path,start_date, end_date):
    # demand_type: Start / End
    # path: 'D:\my\dataset\citibike\data\day\demand\' /// 'D:\my\dataset\captitalbike\data\day\demand\'
    # output: (work_day demand and holiday_demand) for demand_type
    demand_f=path+'day/demand/'
    if demand_type=='Start':
        demand_f = demand_f+'demand_eaStart.csv'
    if demand_type=='End':
        demand_f = demand_f+'demand_eaEnd.csv'
    demand_df = pd.read_csv(demand_f)
    calendar = pd.read_csv('D:\my\dataset\calendar\calendar.csv')
    # find workday
    if end_date=='':
        work_day = \
        calendar[(calendar['workday'] == 1) & (calendar['Date'] > start_date)][
            'Date'].values.tolist()
        holi_day = \
        calendar[(calendar['workday'] == 0) & (calendar['Date'] > start_date)][
            'Date'].values.tolist()
    else:
        work_day = calendar[(calendar['workday']==1)&(calendar['Date']>start_date)&(calendar['Date']<end_date)]['Date'].values.tolist()
        holi_day = calendar[(calendar['workday']==0)&(calendar['Date']>start_date)&(calendar['Date']<end_date)]['Date'].values.tolist()
    num_wd = len(work_day)
    num_holi = len(holi_day)
    # print(work_day)
    # no demands in some days
    diff_set4work = diffTwolist(work_day,demand_df.columns)
    for d in diff_set4work:
        work_day.remove(d)

    diff_set4holi = diffTwolist(holi_day,demand_df.columns)
    for d in diff_set4holi:
        holi_day.remove(d)

    work_demand = demand_df[work_day]
    holi_demand=demand_df[holi_day]
    demand4starions_workday = work_demand.sum(axis=1)/num_wd
    demand4station_holiday = holi_demand.sum(axis=1)/num_holi
    demand_df = pd.concat([demand4starions_workday,demand4station_holiday],axis=1)
    col1 = demand_type+'_'+'workday'
    col2 = demand_type+'_'+'holiday'
    # change col name
    demand_df.rename(columns = {0:col1,1:col2},inplace=True)

    return demand_df

def NORMARLIZATION(df):
    feat = df.values
    mm_scaler  = preprocessing.MinMaxScaler()
    feat_sacler = mm_scaler.fit_transform(feat)
    return feat_sacler

def cluster(cluster_n,path,start_date,end_date):
    # start_date : citibike:(2013-06-01),  capitalbike()
    # end_date: citibike(2017-11-30),  captitalbike()
    # return station sets
    kwds='index_num'
    # if 'capi' in path:
    #     kwds = 'short_name'
    # else:
    #     kwds = 'station_id'
    station_infor_path = path+'station_info.csv'
    station_info_df = pd.read_csv(station_infor_path)
    Start_demand = avgDemand('Start',path,start_date,end_date)
    End_demand = avgDemand('End',path,start_date,end_date)
    feature_df = pd.concat([Start_demand,End_demand],axis=1)
    feature_df = pd.concat([station_info_df[[kwds,'lat','lon']],feature_df],axis=1)

    # drop the rows without demand
    # real_bike = feature_df[(feature_df['Start_holiday']!=0)&(feature_df['Start_workday']!=0)&(feature_df['End_workday']!=0)&(feature_df['End_holiday']!=0)]
    real_bike = feature_df
    real_bike_feature = real_bike[['lat','lon','Start_holiday','Start_workday','End_holiday','End_workday']]
    nm_feat = NORMARLIZATION(real_bike_feature)
    C_MODEL = KMeans(cluster_n)
    C_MODEL.fit(nm_feat)
    label = C_MODEL.labels_
    center = C_MODEL.cluster_centers_
    # print(len(real_bike_feature['lat'].values),len(real_bike_feature['lon'].values))
    # real_bike_feature.plot.scatter(x='lat',y='lon',color='b')
    plt.scatter(nm_feat[:,0],nm_feat[:,1],c=label)
    plt.scatter(center[:,0],center[:,1],c='r',marker='s')
    plt.show()
    cluster_r = np.concatenate((np.reshape(real_bike[kwds].values,newshape=[-1,1]),np.reshape(label,newshape=[-1,1])),axis=1)
    # print(cluster_r)
    stationToonehotcluster(path,cluster_r,cluster_n)
    return cluster_r

def stationToonehotcluster(path, cluster_list,cluster_n):
    dic = {}
    # {label:[station_id]}
    for station_id, label in cluster_list:
        # print(station_id,label)
        if station_id not in dic.keys():
            dic[int(station_id)] = int(label)
    station_f = path + 'station_info.csv'
    station_df = pd.read_csv(station_f)
    station_c = []
    stations = station_df['index_num'].values
    onehot = OneHotEncoder(sparse=False)
    for i in stations:
        c_i = dic[i]
        station_c.append(c_i)

    # save station_id + cluster_id into a  file
    j = json.dumps(dic)
    f_n =path+str(cluster_n)+ "scid.json"
    w = open(f_n,'w')
    w.write(j)
    w.close()
    station_c = np.expand_dims(station_c, -1)
    station_c_one = onehot.fit_transform(station_c)
    sc_f = path + str(cluster_n)+'station_cluster.csv'
    np.savetxt(sc_f, station_c_one, delimiter=',', newline='\n')

def calProb4eacluster(path,cluster_n,dur_stop):
    # probability of one cluster to others
    # cluster_list [station_id,label]
    # print(cluster_list)
    dic = json.load(open('D:\my\dataset\citibike\data\\20scid.json','r'))
    c_metric = np.zeros(shape=[cluster_n,cluster_n])

    # find edges and fill the c_metric (2010-9-20,2011-9-20)
    dir = path+'day\graph/'
    file_list = callAllfile(dir)
    i=0
    for file in file_list:
        # Daily cluster distribution
        c_daily_m = np.zeros(shape=[cluster_n, cluster_n])
        if i==dur_stop:
            break
        df = pd.read_csv(file)
        for index,row in df.iterrows():
            # print(index)
            if str(row['source'].astype(int)) not in dic.keys() or str(row['target'].astype(int)) not in dic.keys(): continue
            c_s = dic[str(row['source'].astype(int))]
            c_t = dic[str(row['target'].astype(int))]
            c_metric[c_s,c_t]+=1
            c_metric[c_t,c_s]+=1
            c_daily_m[c_t,c_s]+=1
            c_daily_m[c_s,c_t]+=1
        i+=1

        c_daily_sum = np.sum(c_daily_m,axis=1)
        c_daily_m = c_daily_m/c_daily_sum
        c_daily_m[np.isnan(c_daily_m)]=0.0
        f_d = file.split('\\')[-1]
        c_daily_f = path + 'day\graph\cluster_graph\\'+f_d
        np.savetxt(c_daily_f,c_daily_m,delimiter=',',newline='\n')
    c_sum = np.sum(c_metric,axis=1)
    c_prob = np.zeros_like(c_metric)
    c_prob = c_metric/c_sum
    c_prob[np.isnan(c_prob)]=0.0
    c_metric_f = path+'cluster.csv'
    # print(c_prob)
    np.savetxt(c_metric_f,c_prob,delimiter=',',newline='\n')
    # station_c_pro = np.matmul(station_c_one,c_prob)
    # station_c_pro_f = path + 'station_c_pro.csv'
    # np.savetxt(station_c_pro_f,X=station_c_pro,delimiter=',')





def corrGraph(path,start_date,end_date):
    kwds = 'index_num'
    station_infor_path = path+'station_info.csv'
    station_info_df = pd.read_csv(station_infor_path)
    Start_demand = avgDemand('Start',path,start_date,end_date)
    End_demand = avgDemand('End',path,start_date,end_date)
    feature_df = pd.concat([Start_demand,End_demand],axis=1)
    feature_df = pd.concat([station_info_df[[kwds,'lat','lon']],feature_df],axis=1)
    real_bike_feature = feature_df[['Start_holiday','Start_workday','End_holiday','End_workday']]
    nm_feat = NORMARLIZATION(real_bike_feature)
    T_feat= pd.DataFrame(data=nm_feat,columns=['Start_holiday','Start_workday','End_holiday','End_workday']).T
    sToSpearson = T_feat.corr(method='pearson')
    st_corr_f = path +'sTos_corr.csv'
    sToSpearson.to_csv(st_corr_f)


#
path = 'D:\my\dataset\citibike\data/'
start_date = '2016-10-01'
end_date = '2017-10-28'
cluster_n = 20
# cluster_res = cluster(cluster_n,path,start_date,end_date)
# calProb4eacluster(path,cluster_n,389)
# corrGraph(path,start_date,end_date)

