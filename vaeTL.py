import tensorflow as tf
from numpy import pi
# z_x (u_zx,sig_zx)
# z_c (u_zc,sig_zc)

# zc
class vaeTL(tf.keras.Model):
    def __init__(self, latent_dim,out_dim):
        super(vaeTL,self).__init__()
        # encoder
        # z_c
        self.miuEncoder4c = tf.keras.layers.Dense(latent_dim)
        self.varEncoder4c = tf.keras.layers.Dense(latent_dim)
        # z_x = f(z_c,x)
        self.miuEncoder4x = tf.keras.layers.Dense(latent_dim)
        self.varEncoder4x = tf.keras.layers.Dense(latent_dim)


        # decoder
        # (z_x,z_c)->x'
        self.fusion_fc= tf.keras.layers.Dense(latent_dim)

        # NYC:256
        self.fc1 = tf.keras.layers.Dense(64,activation=tf.nn.leaky_relu)
        self.fc2 = tf.keras.layers.Dense(out_dim)


    def reparamZ(self,z_u,z_log_var):
        # z = miu+var*epsilon
        epsilon = tf.random.normal(shape=z_u.shape)
        return z_u+ tf.exp(z_log_var/2)*epsilon

    def calZ(self,z_u,z_log_var):
        # z = miu+var*epsilon
        return z_u+ tf.exp(z_log_var/2)


    def guassian_decoder(self,z,miu,log_var):
        # y = -1/ 2 * tf.math.log(2 * pi) - 1 / 2 * log_var - 1 / 2 * (z - miu)**2 / tf.exp(log_var)
        y = tf.exp(-0.5*(z-miu)**2/tf.exp(log_var))/tf.sqrt(2*pi*tf.exp(log_var))
        return y


    def decoder(self,z):
        y = self.fc2(self.fc1(z))
        return y

    def call(self,inputs,**kwargs):
        Rout_x,Rout_c = inputs
        # encoder
        miu_c = self.miuEncoder4c(Rout_c)
        log_var2_c = self.varEncoder4c(Rout_c)
        z_c = self.reparamZ(miu_c,log_var2_c)
        miu_x = self.miuEncoder4x(Rout_x)
        log_var2_x = self.varEncoder4x(Rout_x)
        z_x = self.reparamZ(miu_x,log_var2_x)
        decode_hid = tf.concat([z_x,z_c],axis=-1)
        decode_hid = tf.nn.l2_normalize(decode_hid)
        decode_hid = self.fusion_fc(decode_hid)
        # decoder
        y_pre = self.decoder(decode_hid)
        y_pre = tf.expand_dims(y_pre,axis=-1)

        return y_pre,miu_x,miu_c,log_var2_x,log_var2_c
