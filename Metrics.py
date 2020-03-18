import tensorflow as tf
import  numpy as np
from BikeS.dataprocess import MinMaxNormalization

# MAPE=sum[abs(real_label-real_pre)/real_label]
# RMSPE
def pred_MSE(label,pred,norm,label_mask):
    # stations in current timestamp
    real_label = norm.inverse_transform(label_mask*label)
    real_pre =  norm.inverse_transform(label_mask*pred)
    # error_rate = abs(y_label-y_pre)/y_label
    rate = calRate(real_label,real_pre)

    mae = np.sum(rate)/np.sum(label_mask)
    rmse = np.sqrt(np.sum(np.square(rate))/np.sum(label_mask))
    # print(real_pre)
    return mae,rmse

def news_MSE(label,pred,norm,ns_mask,time_mask):
    # metrics for new stations
    mask = ns_mask*time_mask
    real_label = norm.inverse_transform( mask*label)
    real_pre = norm.inverse_transform( mask*pred)
    n_rate = calRate(real_label,real_pre)
    n_mae = np.sum(n_rate)/np.sum(mask)
    n_rmse = np.sqrt(np.sum(np.square(n_rate))/np.sum(mask))

    # metrics for existing stations
    o_mask = time_mask-mask
    o_label = norm.inverse_transform(o_mask*label)
    o_pre = norm.inverse_transform(o_mask*pred)
    o_rate = calRate(o_label,o_pre)
    o_mae = np.sum(o_rate)/np.sum(o_mask)
    o_rmse = np.sqrt(np.sum(np.square(o_rate))/np.sum(o_mask))

    rate = np.squeeze(n_rate,axis=0)
    return rate,n_mae,n_rmse,o_mae,o_rmse

def calRate(l,p):
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.abs((l - p) / l)
    rate[np.isnan(rate)] = 0.0
    rate[np.isinf(rate)] = 0.0
    return rate