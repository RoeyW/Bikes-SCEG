
#input X,C, f_x,f_c,adjlist_s,adjlist_c
#output X_t
# (x,c)->z_x ///  ()
# input_dim, gcn_output_dim, output_dim, activation, rnn_units,featureless


import tensorflow as tf
from BikeS.dataprocess import loadinput, MinMaxNormalization,station_file_num
import BikeS.Egcn as Egcn
from tensorflow.keras import optimizers
from BikeS.Metrics import pred_MSE, news_MSE
import csv
import pandas as pd
import datetime
import numpy as np


city = 'wash'

def mask(metric):
    m_one = tf.ones_like(metric)
    m_zero = tf.zeros_like(metric)
    metric_onehot = tf.where(metric > 0, x=m_one, y=m_zero)

    return metric_onehot

# *********** hyp-para*************
path,cluster_adj='',''
if city=='nyc':
    path = 'D:/my/dataset/citibike/data/day/'
    cluster_adj = pd.read_csv('D:/my/dataset/citibike/data/cluster.csv',header=None).values
    cluster_feat = pd.read_csv('D:/my/dataset/citibike/data/station_cluster.csv',header=None).values
    new_station  = pd.read_csv('D:/my/dataset/citibike/data/new_stations.csv',header=None).values
    learning_rate = 5e-4
    seq_len = 6
    c_gcn_output_dim = 64
    s_gcn_output_dim = 128
    activation = tf.nn.tanh
    rnn_units = 128
    loop_num = 5
    lamda_reg = 0.001
else:
    path = 'D:/my/dataset/capitalbike/data/day/'
    cluster_adj =pd.read_csv('D:/my/dataset\capitalbike/data/cluster.csv',header=None).values
    cluster_feat = pd.read_csv('D:/my/dataset/capitalbike/data/station_cluster.csv', header=None).values
    new_station = pd.read_csv('D:/my/dataset/capitalbike/data/new_stations.csv',header=None).values
    learning_rate = 5e-4
    seq_len = 6
    c_gcn_output_dim = 32
    s_gcn_output_dim = 64
    activation = tf.nn.tanh
    rnn_units = 128
    loop_num = 5
    lamda_reg = 0.001
station_num, timestamp =station_file_num(path)
cluster_num = cluster_adj.shape[1]
# print(station_num)
output_dim = station_num

#  path for result
w_filename = 'D:\\my\\mypaper\\BIKEsharing\\results\\'+city+'model_res_end_egvae.csv'
ns_w_filename = 'D:\\my\\mypaper\\BIKEsharing\\results\\'+city+'WEIGHT_res_end_egvae.csv'



# ----Model initial-------
c_hid_layer_list = [cluster_num, c_gcn_output_dim]
s_hid_layer_list = [station_num, s_gcn_output_dim]


# ------inputs load ----------

def selectModel(m):
    if m == 'GCNGRU':
        MODEL = Egcn.GCNGRU(station_num, s_gcn_output_dim, output_dim, activation, rnn_units,  seq_len)
    elif m=='GRU':
        MODEL = Egcn.GRU(output_dim,rnn_units,seq_len)
    elif m == 'CGCNGRU':
        MODEL = Egcn.C_GCNGRU(c_hid_layer_list, s_hid_layer_list, output_dim, activation, rnn_units, seq_len)
    elif m == 'GCNVAE':
        MODEL = Egcn.GCNVAE(c_hid_layer_list, s_hid_layer_list, output_dim, rnn_units, seq_len)
    elif m == 'EGCN_VAE':
        MODEL = Egcn.EGCNVAE(c_hid_layer_list, s_hid_layer_list, rnn_units, seq_len, output_dim)
    elif m=="MG_VAE":
        MODEL = Egcn.MG_VAE(c_hid_layer_list,s_hid_layer_list,cluster_num,rnn_units,seq_len,output_dim)
    elif m == 'EGCN':
        MODEL = Egcn.EGCN(c_hid_layer_list, s_hid_layer_list, output_dim, rnn_units, seq_len)
    return MODEL



whole_dataset_len = timestamp-seq_len
print('whole_dataset_len :{}-----------learning rate:{}'.format(whole_dataset_len,learning_rate))

cluster_adj = tf.expand_dims(tf.convert_to_tensor(cluster_adj,dtype=tf.float32),axis=0)
cluster_feat = tf.convert_to_tensor(cluster_feat,dtype=tf.float32)


# Model initialization
# models=['GRU','GCNGRU','CGCNGRU','EGCN','GCNVAE','EGCN_VAE']
models=['EGCN_VAE']
optimizer = optimizers.Adam(lr=learning_rate)

for m in models:
    MODEL = selectModel(m) # initialization
    NS_RES_writer = csv.writer(open(ns_w_filename, 'w', newline=''))

    # result writers
    GCNCRU_RES_writer = csv.writer(open(w_filename, 'a', newline=''))
    cur_time = datetime.datetime.now()
    GCNCRU_RES_writer.writerow([learning_rate,lamda_reg, seq_len, s_gcn_output_dim, rnn_units, m,'0.01w',cur_time])
    GCNCRU_RES_writer.writerow(['date', 'MAE', 'RMSE', 'N_MAE', 'N_RMSE','O_MAE','O_RMSE'])
    # NS_RES_writer.writerow([m, cur_time])
    print('MODEL_NAME:',m)

    # record losses for each epoch
    EPOCH = []
    start_NORM = MinMaxNormalization()
    end_NORM = MinMaxNormalization()
    inputs = loadinput(city, seq_len, start_NORM, end_NORM, loop_num)
    for i in range(loop_num):
        EPOCH.append(0)
    loop = 0
    epoch = 0
    loss=0
    while epoch < whole_dataset_len - 1:
        inputs_list = inputs.__next__()

        adj_list = tf.expand_dims(tf.convert_to_tensor(inputs_list[0], dtype=tf.float32), axis=0)
        x_list = tf.expand_dims(tf.convert_to_tensor(inputs_list[1], dtype=tf.float32), axis=0)
        label = tf.convert_to_tensor(inputs_list[-1], dtype=tf.float32)
        time_feat = tf.expand_dims(tf.convert_to_tensor(inputs_list[2], dtype=tf.float32), axis=0)
        label_mask = tf.expand_dims(mask(x_list[0, 0, :, 0]), axis=-1)
        # label_mask = tf.squeeze(mask(label),axis=0)
        cluster_feat=label_mask*cluster_feat #[s_num,1]*[s_num,c_num]


        if m == 'GCNGRU':
            INPUTS = (adj_list, x_list, time_feat, label)
        elif m=='GRU':
            INPUTS = (x_list,time_feat,label)
        else:
            INPUTS = (adj_list, x_list, cluster_adj, cluster_feat, time_feat, label)

        if epoch + 1 >= whole_dataset_len * 0.85 and loop < loop_num:
            if loop < loop_num - 1:
                epoch = 0
            loop += 1
        # train
        if epoch < whole_dataset_len * 0.85 and loop < loop_num:

            # GCNGRU_
            print('--Train-epoch:{}-sample:{}---:'.format(loop, epoch), '----date:', inputs_list[-2])
            with tf.GradientTape() as tape:
                l2_reg = tf.reduce_mean([tf.nn.l2_loss(v) for v in MODEL.trainable_variables])
                if 'VAE' in m:
                    outputs,_= MODEL(INPUTS)
                    loss= Egcn.vae_lossfunction(label, outputs[0], outputs[1:], label_mask, loop) + lamda_reg * l2_reg
                    # loss+=loss_i
                    MAE, RMSE = pred_MSE(label.numpy(), outputs[0].numpy(), start_NORM, label_mask.numpy())
                else:  # GCNGRU C_GCNGRU EGCN
                    outputs = MODEL(INPUTS)
                    loss= Egcn.lossfunction(label, outputs, label_mask) + lamda_reg * l2_reg
                    # loss+=loss_i
                    MAE, RMSE = pred_MSE(label.numpy(), outputs.numpy(), start_NORM, label_mask.numpy())
                print('---------loss:', float(loss), '-------RMSE:', float(RMSE))
                EPOCH[loop] += loss
                # if (epoch + 1) % 5  == 0:
                #     loss = loss / 5

                grads = tape.gradient(loss, MODEL.trainable_variables)
                optimizer.apply_gradients(zip(grads, MODEL.trainable_variables))
                # print('----Train output----')
                # loss = 0
        # validation
        elif epoch < whole_dataset_len*0.9:
            output = MODEL.predict(INPUTS)
            # output = GCNGRU.predict(X_inputs)
            MAE,RMSE = pred_MSE(label.numpy(),output[0],start_NORM,label_mask.numpy())
            print('--------Validation--sample:{}---date:{}---MAE:{}--RMSE:{}'.format(epoch,inputs_list[-2],MAE,RMSE))
        # test
        else:
            # new station for each time
            output= MODEL.predict(INPUTS)
            # output = GCNGRU.predict(X_inputs)
            MAE, RMSE = pred_MSE(label.numpy(), output[0], start_NORM, label_mask.numpy())
            rate,n_mae,n_rmse,o_mae,o_rmse = news_MSE(label.numpy(), output[0], start_NORM, new_station, label_mask.numpy())
            print('----------TEST----date:{}---MAE:{}----RMSE:{}'.format(inputs_list[-2], MAE, RMSE))
            GCNCRU_RES_writer.writerow([inputs_list[-2], MAE, RMSE, n_mae, n_rmse,o_mae,o_rmse])
            sc_weight =np.expand_dims(x_list[0, 0, :, 0],axis=-1)* output[-1][0]
            NS_RES_writer.writerow(inputs_list[-2])
            NS_RES_writer.writerows(sc_weight)

        epoch += 1

    print(EPOCH)

    # test

