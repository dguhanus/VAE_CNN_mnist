# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 23:25:26 2023
This for LSTM with VAE for High dimensional Latent space.
 trying to implement LSTM based VAE. Input shape is (samples/batch_size, timestep, feature) 
 And I want to have a model's output shape as (24)
@author: dguha
"""

# encoder
latent_dim = 24
inter_dim = 32
timesteps, features = 96, 24

def sampling(args):
    z_mean, z_log_sigma = args
    batch_size = tf.shape(z_mean)[0] # <================
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return z_mean + z_log_sigma * epsilon

# timesteps, features
input_x = Input(shape= (timesteps, features)) 

#intermediate dimension 
h = LSTM(inter_dim)(input_x)


#z_layer
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)
z = Lambda(sampling)([z_mean, z_log_sigma])

# Reconstruction decoder
decoder1 = RepeatVector(timesteps)(z)
decoder1 = LSTM(inter_dim, return_sequences=True)(decoder1)
decoder1 = Dense(features)(decoder1)

'''
If you are training a text VAE, you can maybe try using a TimeDistributed layer like this:
The TimeDistributed layer simply applies a Dense layer with a softmax activation 
function to each time step in the sequence coming from the LSTM layer.

decoder1 = tf.keras.layers.RepeatVector(timesteps)(z)
decoder1 = tf.keras.layers.LSTM(inter_dim, return_sequences=True)(decoder1)
output =  tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24, activation='linear'))(decoder1)

'''

output = (Dense(24, activation='softmax'))(decoder1)


def vae_loss2(input_x, decoder1, z_log_sigma, z_mean):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    #recon = K.sum(K.binary_crossentropy(input_x, decoder1))
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    recon = cross_entropy(input_x, output)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma)

    return recon + kl

m = Model(input_x, output)
m.compile(loss=vae_loss2, optimizer='adam', metrics=['accuracy'])