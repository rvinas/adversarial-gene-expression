from keras.engine.topology import Layer
import keras.backend as K
from keras.layers import LSTM
import tensorflow as tf

from keras.constraints import MinMaxNorm


class fNRI(Layer):
    def __init__(self, **kwargs):
        self.embedding_size = 4
        super(fNRI, self).__init__(**kwargs)

    def build(self, input_shape):
        # f_emb
        self.f_emb = tf.keras.layers.Dense(self.embedding_size)
        self.f_emb.build(input_shape)
        self._trainable_weights = self.f_emb.trainable_weights

        # f1_e
        self.f1_e = tf.keras.layers.Dense(3)
        self.f1_e.build(input_shape)
        self._trainable_weights.append(self.f1_e.trainable_weights)

        # f_v
        self.f_v = tf.keras.layers.Dense(self.embedding_size)
        self.f_v.build(input_shape)
        self._trainable_weights.append(self.f_v.trainable_weights)

        super(fNRI, self).build(input_shape)  # Be sure to call this at the end

    @staticmethod
    def cross_concat(h):
        batch_size, nb_genes, dim = h.shape
        h_ = tf.tile(h[..., None, :],
                      [1, 1, nb_genes, 1])  # Shape=(batch_size, nb_genes, nb_genes, dim)
        h_t = tf.transpose(h_, perm=(0, 2, 1, 3))  # Shape=(batch_size, nb_genes, nb_genes, dim)
        c = tf.stack((h_, h_t), axis=-1)  # Shape=(batch_size, nb_genes, nb_genes, dim, 2)
        c_ = tf.reshape(c, shape=(-1, dim * 2))  # Shape=(batch_size, nb_genes, nb_genes, dim * 2)
        return c_

    def call(self, x, **kwargs):
        x_ = x[..., None]  # Shape=(batch_size, nb_genes, 1)
        h1 = self.f_emb(x_)  # Shape=(batch_size, nb_genes, embedding_size)
        c = self.cross_concat(h1)  # Shape=(batch_size, nb_genes, nb_genes, embedding_size * 2)
        # TODO: Set diagonal to 0
        h1_edges = self.f1_e(c)  # Shape=(batch_size, nb_genes, nb_genes, 3)
        h2 = self.f_v(tf.reduce_sum(h1_edges, axis=2))  # Shape=(batch_size, nb_genes, 3)
        c = self.cross_concat(h2)  # Shape=(batch_size, nb_genes, nb_genes, 3)
        # TODO: Reduce batch size?
        edge_probs = tf.nn.softmax(c, axis=-1)
        
        pass

    def compute_output_shape(self, input_shape):
        return input_shape

class MinibatchDiscrimination(Layer):
    def __init__(self, units=5, units_out=10, **kwargs):
        self._w = None
        self._units = units
        self._units_out = units_out
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        self._w = self.add_weight(name='w',
                                  shape=(input_shape[1], self._units * self._units_out),
                                  initializer='uniform',
                                  trainable=True
                                  )
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, **kwargs):
        h = K.dot(x, self._w)  # Shape=(batch_size, units * units_out)
        h = K.reshape(h, (-1, self._units, self._units_out))  # Shape=(batch_size, units, units_out)
        h_t = K.permute_dimensions(h, [1, 2, 0])  # Shape=(units, units_out, batch_size)
        diffs = h[..., None] - h_t[None, ...]  # Shape=(batch_size, units, units_out, batch_size)
        abs_diffs = K.sum(K.abs(diffs), axis=1)  # Shape=(batch_size, units_out, batch_size)
        features = K.sum(K.exp(-abs_diffs), axis=-1)  # Shape=(batch_size, units_out)
        return features

    def compute_output_shape(self, input_shape):
        return input_shape[0], self._units_out


class CorrDiscr(Layer):
    def __init__(self, **kwargs):
        self._w = None
        self._projection_dim = 1
        super(CorrDiscr, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CorrDiscr, self).build(input_shape)

    def call(self, x, **kwargs):
        x_mean = K.mean(x, axis=-1)  # Shape=(batch_size,)
        x_std = K.std(x, axis=-1)  # Shape=(batch_size,)
        x = (x - x_mean[:, None]) / x_std[:, None]  # Shape=(batch_size, units)
        x_t = K.transpose(x)  # Shape=(units, batch_size)
        out = K.dot(x, x_t) / K.cast(x.shape[1], dtype=tf.float32)  # Shape=(batch_size, batch_size)
        return K.std(out, axis=-1)[:, None] * K.ones((self._projection_dim,),
                                                     dtype=tf.float32)  # Shape=(batch_size, projection_dim)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self._projection_dim
