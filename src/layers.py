from keras.engine.topology import Layer
import keras.backend as K
from keras.layers import LSTM, Lambda, Dense, TimeDistributed
import tensorflow as tf
from keras.constraints import MinMaxNorm


class fNRI(Layer):
    def __init__(self, **kwargs):
        self.embedding_size = 4
        self.K = 3

        self.f1_e_dim = 2
        self.f1_v_dim = 3
        self.message_size = 2
        super(fNRI, self).__init__(**kwargs)

    def build(self, input_shape):
        bs, nb_genes = input_shape

        # f1_emb
        self.f1_emb = Dense(self.embedding_size)
        self.f1_emb.build((bs, nb_genes, 1))
        self._trainable_weights = self.f1_emb.trainable_weights

        # f2_emb
        self.f2_emb = Dense(self.embedding_size)
        self.f2_emb.build((bs, nb_genes, 1))
        self._trainable_weights.extend(self.f2_emb.trainable_weights)

        # f1_e
        self.f1_e = Dense(self.f1_e_dim)
        self.f1_e.build((bs, nb_genes, nb_genes, self.embedding_size * 2))
        self._trainable_weights.extend(self.f1_e.trainable_weights)

        # f2_e
        self.f2_e = Dense(self.K)
        self.f2_e.build((bs, nb_genes, nb_genes, self.f1_v_dim * 2))
        self._trainable_weights.extend(self.f2_e.trainable_weights)

        # fk_e
        self.fk_e = []
        # TODO: Set first as null channel (e.g. no interaction) -> function returns zeros
        # fk_e_K = lambda x: K.zeros(shape=(bs, nb_genes, nb_genes, self.message_size), dtype=tf.float32)
        # self.fk_e.append(fk_e_K)
        for k in range(self.K):
            fk_e_K = Dense(self.message_size)
            fk_e_K.build((bs, nb_genes, nb_genes, self.embedding_size * 2))
            self._trainable_weights.extend(fk_e_K.trainable_weights)
            self.fk_e.append(fk_e_K)

        # f1_v
        self.f1_v = Dense(self.f1_v_dim)
        self.f1_v.build((bs, nb_genes, self.f1_e_dim))
        self._trainable_weights.extend(self.f1_v.trainable_weights)

        # f2_v
        self.f2_v = Dense(1)
        self.f2_v.build((bs, nb_genes, self.K))
        self._trainable_weights.extend(self.f2_v.trainable_weights)

        super(fNRI, self).build(input_shape)  # Be sure to call this at the end

    @staticmethod
    def cross_concat(h):
        batch_size, nb_genes, dim = h.shape
        h_ = tf.tile(h[..., None, :],
                      [1, 1, nb_genes, 1])  # Shape=(batch_size, nb_genes, nb_genes, dim)
        h_t = tf.transpose(h_, perm=(0, 2, 1, 3))  # Shape=(batch_size, nb_genes, nb_genes, dim)
        c = tf.stack((h_, h_t), axis=-1)  # Shape=(batch_size, nb_genes, nb_genes, dim, 2)
        c_ = tf.reshape(c, shape=(-1, nb_genes, nb_genes, dim * 2))  # Shape=(batch_size, nb_genes, nb_genes, dim * 2)
        return c_

    def call(self, x, **kwargs):
        # Estimate edge probabilities
        # x_ = Lambda(lambda x: x[..., None])(x)  # Shape=(batch_size, nb_genes, 1)
        x_ = x[..., None]
        h1 = self.f1_emb(x_)  # Shape=(batch_size, nb_genes, embedding_size)
        c = self.cross_concat(h1)  # Shape=(batch_size, nb_genes, nb_genes, embedding_size * 2)

        # TODO: Set diagonal to 0
        h1_edges = self.f1_e(c)  # Shape=(batch_size, nb_genes, nb_genes, f1_e_dim)
        h = tf.reduce_sum(h1_edges, axis=2)  # Shape=(batch_size, nb_genes, f1_e_dim)
        h2 = self.f1_v(h)  # Shape=(batch_size, nb_genes, f1_v_dim)
        c = self.cross_concat(h2)  # Shape=(batch_size, nb_genes, nb_genes, f1_v_dim * 2)
        ## TODO: Reduce batch size?
        h = tf.reduce_sum(c, axis=0)  # Shape=(nb_genes, nb_genes, f1_v_dim * 2)
        edge_logits = self.f2_e(h)
        edge_probs = tf.nn.softmax(edge_logits, axis=-1)   # Shape=(nb_genes, nb_genes, K)

        # Sample data from inferred graph
        # TODO: Add noise???
        h1_ = self.f2_emb(x_)  # Shape=(batch_size, nb_genes, embedding_size)
        c = self.cross_concat(h1_)  # Shape=(batch_size, nb_genes, nb_genes, embedding_size * 2)
        messages = []
        for f in self.fk_e:
            message = f(c)  # Shape=(batch_size, nb_genes, nb_genes, message_size, 1)
            messages.append(message)
        messages = tf.stack(messages, axis=-1)  # Shape=(batch_size, nb_genes, nb_genes, message_size, K)
        v_e_messages = tf.reduce_sum(edge_probs[None, :, :, None, :] * messages,
                                     axis=(2, 4))  # Shape=(batch_size, nb_genes, message_size)
        h2_ = x_ + self.f2_v(tf.concat((v_e_messages, x_), axis=-1))  # Shape=(batch_size, nb_genes, 1)
        out = tf.squeeze(h2_, axis=-1)  # Shape=(batch_size, nb_genes)
        return out

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
