import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # a_hat = D^(-0.5)*a^*D^(-0.5)
    # adj = sp.coo_matrix(adj)

    rowsum = np.array(adj.sum(1))  # D
    # D = diag(row_sum(a^))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5


def transforTosparsematrix(edge, hasvalue, shape):
    # transfer sparse matrix to normal matrix
    # input: source, target nodes index, edge_value(=1 if hasvalue=0)
    # output: normal matrix

    # e.g.
    # # source index
    # i = [0, 0, 0, 1, 1, 1]
    # # target index
    # j = [0, 1, 2, 0, 3, 4]
    # value on the edge
    edge = edge.astype(int)
    if hasvalue:
        value = edge[:, 2]
    else:
        value = np.ones_like(edge[:, 0], dtype=np.float)
    # print(value.shape)
    # print(edge.shape)
    A = sp.coo_matrix((value, (edge[:, 0], edge[:, 1])), shape)


def loadinput(city, seq_len, start_NORM, end_NORM,loop_num):
    # city: capital/nyc
    graph_path = ''
    if city == 'nyc':
        graph_path = 'D:\my\dataset\citibike\data\day\graph/new_station/'
        start_demand = pd.read_csv('D:\my\dataset\citibike\data\day\demand\demand_eaStart.csv')
        end_demand = pd.read_csv('D:\my\dataset\citibike\data\day\demand\demand_eaEnd.csv')
        # weather_type temp wind_speed
        wea_df = pd.read_csv('D:\my\dataset\citibike\data\day/weather_enc.csv')
    else:
        graph_path = 'D:\my\dataset\capitalbike\data\day\graph/new_station'
        start_demand = pd.read_csv('D:\my\dataset\capitalbike\data\day\demand\demand_eaStart.csv')
        end_demand = pd.read_csv('D:\my\dataset\capitalbike\data\day\demand\demand_eaEnd.csv')
        wea_df = pd.read_csv('D:\my\dataset\weather\washington/Washington-day.csv')

    calendar = pd.read_csv('D:\my\dataset\calendar/calendar_enc.csv')  # weekday[1-6] workday[0/1]

    start_demand = start_NORM.fit_transform(start_demand)
    end_demand = end_NORM.fit_transform(end_demand)
    os.chdir(graph_path)
    file_list = os.listdir(graph_path)
    # print(start_demand.columns)
    adj_list = []
    feat_list = []
    # end_list=[]
    label_list = []
    timefeat_list = []
    label_date = ''
    station_num = start_demand['Station_id'].values.shape[0]
    # batch
    loop=0
    i=0
    while i < (len(file_list) - seq_len):
        # print(i)
        if i + seq_len == len(file_list): break
        label_date = file_list[i + seq_len].split('.')[0]
        # label
        label_value = np.reshape(end_demand[label_date].values, [-1, 1])
        label_list.append(label_value)
        for j in range(seq_len):
            filename = file_list[i + j]
            date = filename.split('.')[0]
            start_value = np.reshape(start_demand[date].values, [-1, 1])
            end_value = np.reshape(end_demand[date].values, [-1, 1])
            feat_value = np.concatenate((start_value, end_value), axis=1)
            weather_v = wea_df[wea_df['date'] == date].values[:, 2:]
            calendar_v = calendar[calendar['Date'] == date].values[:, 2:]
            other_feat = np.concatenate([weather_v, calendar_v], axis=1).astype(float)

            # end_value = end_demand[date]

            graph_df = pd.read_csv(filename)[['source', 'target']]
            edge = graph_df[(graph_df['source'] > 0) & (graph_df['target'] > 0)].values
            adj = transforTosparsematrix(edge, hasvalue=False, shape=[station_num, station_num])
            # adj normalization
            # adj_list.append(normalize_adj(adj))
            adj_list.append(adj.toarray())
            feat_list.append(feat_value)
            timefeat_list.append(other_feat[0])
            # end_list.append((end_value))
        yield adj_list, feat_list, timefeat_list, label_date, label_list
        if loop<loop_num-1 and i+1>0.95*(len(file_list) - seq_len):
            loop+=1
            i=0
        adj_list = []
        feat_list = []
        label_list = []
        timefeat_list = []
        i+=1

class MinMaxNormalization(object):
    '''MinMax Normalization --> [0, 1]
       x = (x - min) / (max - min).
    '''

    def __init__(self):
        pass

    def fit(self, X):
        x_v = X.values[:, 1:]
        self._min = x_v.min()
        self._max = x_v.max()
        # print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        # X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        # X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


def station_file_num(path):
    df = pd.read_csv(path + '/demand/demand_eaEnd.csv')
    file_n = path + 'graph/new_station/'
    return len(df['Station_id'].unique()), len(os.listdir(file_n))


