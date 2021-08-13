import tensorflow as tf 
import keras
from keras import backend as K


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.output_dim = output_dim

    def call(self, x):
        x = K.cast(K.cast(x / 3600, dtype=tf.int32), dtype=tf.float32) 

        evens = K.reshape(K.stack([K.constant([0.0, 1.0])] * int(self.output_dim / 2), axis=0), shape=(-1, ))
        odds = K.reshape(K.stack([K.constant([1.0, 0.0])] * int(self.output_dim / 2), axis=0), shape=(-1, ))
        pos = K.reshape(K.repeat(K.reshape(K.pow(10000.0, K.cast((K.arange(self.output_dim) / 2) * 2, dtype=tf.float32) / self.output_dim), shape=(1, -1)), K.shape(x)[0] * K.shape(x)[1]), shape=(K.shape(x)[0], K.shape(x)[1], -1))
        evenEmb = K.sin(K.permute_dimensions(K.repeat(x, self.output_dim), pattern=(0, 2, 1)) / pos) * evens
        oddEmb = K.cos(K.permute_dimensions(K.repeat(x, self.output_dim), pattern=(0, 2, 1)) / pos) * odds
        posEmbedding = evenEmb + oddEmb

        return posEmbedding

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
