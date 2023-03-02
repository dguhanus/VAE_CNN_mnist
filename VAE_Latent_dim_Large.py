# -*- coding: utf-8 -*-
"""
Spyder Editor: dguha
Email: dj.gssst@gmail.com

This is for VAE with large dimensional Latent dimension.

Below is for tensorflow 1.14
import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np

Below is for tensorflow 2.x.x
"""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

#encoding_dim = 12

sig_shape = (3600,)
batch_size = 128
latent_dim = 12
input_sig = keras.Input(shape=sig_shape)

x = layers.Dense(128, activation='relu')(input_sig)
x = layers.Dense(64, activation='relu')(x)
shape_before_flattening = K.int_shape(x)

x = layers.Dense(32, activation='relu')(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

encoder=Model(input_sig,[z_mean,z_log_var])



def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
    mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

decoder_input = layers.Input(K.int_shape(z)[1:])
x = layers.Dense(np.prod(shape_before_flattening[1:]),activation='relu')(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(3600, activation='linear')(x)

decoder = Model(decoder_input, x)
z_decoded = decoder(z)

class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.mae(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

y = CustomVariationalLayer()([input_sig, z_decoded])

vae = Model(input_sig, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()
vae.fit(x=X_train, y=None,shuffle=True,epochs=100,batch_size=batch_size,validation_data=(X_test, None))