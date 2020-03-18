import tensorflow as tf
from BikeS.GCN_layer import GCN_layer,GRCU
from BikeS.vaeTL import vaeTL

class EGCNVAE(tf.keras.Model):
    def __init__(self, c_hid_layer_list,s_hid_layer_list,rnn_units, seq_len,output_dim):
        super(EGCNVAE,self).__init__()
        # E-GCN for time-evolving embedding
        self.GRCU_layers = []
        for i in range(1):
            # s_hid_layer_list: feat_dim,gcn_out_dim
            grcu_i = GRCU(s_hid_layer_list,activation=tf.nn.leaky_relu,featureless=True)
            # print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i)
        self.RNN_layer = tf.keras.layers.GRU(rnn_units)
        # GCN for community
        self.c_gcn_layer = GCN_layer(c_hid_layer_list, featureless=True, layer_num=1, activation=tf.nn.leaky_relu)

        # demand and extra-feature embedding
        self.demd_emb_layer = tf.keras.layers.Dense(4)
        self.feat_emb_layer = tf.keras.layers.Dense(8)

        # fuse output of EGCN and embedding of demands
        # nyc:128
        self.fusion_fc = tf.keras.layers.Dense(64,activation=tf.nn.leaky_relu)

        # fc layer after B-GCN
        self.sc_fc = tf.keras.layers.Dense(rnn_units,activation=tf.nn.leaky_relu)

        # E^{intra}: occupation rates of stations in a community
        self.SC_weight = self.add_weight('SC',[s_hid_layer_list[0],c_hid_layer_list[0]],initializer='random_normal')
        # E^{intre} weights between station and other communities
        self.SC_weight_invers = self.add_weight('SC_invers',[s_hid_layer_list[0],c_hid_layer_list[0]],initializer='random_normal')

        self.seq_len = seq_len
        self.station_num = s_hid_layer_list[0]

        # vae for two latent representation[sc, Rount_x]
        # nyc:128
        self.VAE_layer = vaeTL(latent_dim=64,out_dim=output_dim)

    def __call__(self,inputs, **kwargs):
        adjlist, x, c_adj, c_feat, time_feature, _ = inputs

        x_emb = self.demd_emb_layer(x)
        feat_emb = self.feat_emb_layer(time_feature) #[1,seq, dim]


        for unit in self.GRCU_layers:
            Nodes_list = unit(adjlist) #[seq,nodes,feat]
        Nodes_timeemb = tf.stack(Nodes_list,axis=0)
        Nodes_timeemb = tf.expand_dims(Nodes_timeemb,axis=0)
        Nodes_timeemb = tf.concat([Nodes_timeemb,x_emb],axis=-1)
        Nodes_timeemb = self.fusion_fc(Nodes_timeemb)

        Nodes_timeemb = tf.nn.l2_normalize(Nodes_timeemb,axis=1)
        times_emb = tf.reshape(Nodes_timeemb,[1,self.seq_len,-1])
        # feat_emb_node = self.feat_emb4node(feat_emb_node)
        times_emb = tf.concat([times_emb,feat_emb],axis=-1)

        Rout_x = self.RNN_layer(times_emb) #[nodes,dim]
        # E^c: embedding for community nodes
        c_gcn_out = self.c_gcn_layer(c_adj) #[c,dim]

        #  intraweights for each community
        sc_bel = tf.multiply(c_feat,self.SC_weight)
        sc_bel = tf.nn.l2_normalize(sc_bel,axis=0)
        sc_bel = tf.nn.softmax(sc_bel,axis=0)

        # interweights for each station
        sc_invers = tf.multiply((1-c_feat),self.SC_weight_invers)
        sc_invers  =tf.nn.l2_normalize(sc_invers,axis=-1)
        sc_invers  =tf.nn.softmax(sc_invers,axis=-1)

        sc_metric = sc_invers+sc_bel
        sc_metric = tf.expand_dims(sc_metric, axis=0)
        sc = tf.matmul(sc_metric,c_gcn_out) #[s,dim]

        sc = tf.reshape(sc,[1,-1])
        sc = self.sc_fc(sc)
        outputs = self.VAE_layer((Rout_x,sc))
        return outputs,sc_metric



def vae_lossfunction(y, y_pre, miu_var_list,mask,loop):
    mask = tf.expand_dims(mask, 0)
    if len(miu_var_list)==4:
        miu_zx, miu_zc, log_var_zx, log_var_zc = miu_var_list
        mask_y = mask * y
        mask_pre = mask * y_pre
        MSE_loss = tf.reduce_sum(tf.square(mask_y - mask_pre))
        # THE SAME AS REDUCE_MEAN
        KLD_zc = -0.5 * tf.math.reduce_mean(1 + log_var_zc - tf.math.square(miu_zc) - tf.math.exp(log_var_zc))
        KLD_zX = -0.5 * tf.math.reduce_mean(1 + log_var_zx - tf.math.square(miu_zx) - tf.math.exp(log_var_zx))
        # warm up
        TOTAL_loss = MSE_loss + 0.001 * loop * KLD_zX + 0.001 * loop * KLD_zc
    else:
        miu_zc,  log_var_zc = miu_var_list
        mask_y = mask * y
        mask_pre = mask * y_pre
        MSE_loss = tf.reduce_sum(tf.square(mask_y - mask_pre))
        # THE SAME AS REDUCE_MEAN
        KLD_zc = -0.5 * tf.math.reduce_mean(1 + log_var_zc - tf.math.square(miu_zc) - tf.math.exp(log_var_zc))
        TOTAL_loss = MSE_loss  + 0.001 * loop * KLD_zc
    return TOTAL_loss

def lossfunction(y, y_pre, mask):
    mask = tf.expand_dims(mask,0)
    msk_label = mask*y
    msk_pre = mask* y_pre
    loss = tf.reduce_sum(tf.square(msk_label- msk_pre))
    return loss





