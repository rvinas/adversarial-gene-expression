from keras.engine.topology import Layer
import keras.backend as K
from keras.initializers import RandomUniform


class GRI(Layer):
    def __init__(self, root_node, nodes, edges, **kwargs):
        """
        :param root_node: node on top of the hierarchy
        :param nodes: list of nodes (gene symbols) according to which the output nodes will be sorted
        :param edges: dictionary of edges (keys: *from* node, values: dictionary with key *to* node and
               value regulation type)
        """
        self._output_dim = len(nodes)
        self._root_node = root_node
        self._nodes = nodes
        self._edges = edges
        self._r_edges = self._reverse_edges(edges)
        self._structure = None
        super(GRI, self).__init__(**kwargs)

    @staticmethod
    def _reverse_edges(edges):
        r_edges = {}
        for tf, tgs in edges.items():
            for tg, reg_type in tgs.items():
                if tg in r_edges:
                    if tf not in r_edges[tg]:
                        r_edges[tg][tf] = reg_type
                else:
                    r_edges[tg] = {tf: reg_type}
        return r_edges

    def _create_params(self, node, latent_dim, nb_incoming):
        bias = self.add_weight(name='{}_bias'.format(self._root_node),
                               shape=(1, 1),
                               initializer='zeros',
                               trainable=True)
        weights = self.add_weight(name='{}_weights'.format(node),
                                  shape=(latent_dim + nb_incoming, 1),
                                  initializer='glorot_uniform',
                                  trainable=True)
        params = K.concatenate([bias, weights], axis=0)  # Shape=(1 + latent_dim + nb_incoming, 1)
        return params

    def build(self, input_shape):
        batch_size, latent_dim = input_shape

        # Structure is a list of tuples with format: (NODE, INCOMING_NODES, WEIGHTS)
        # The list is sorted so that node i does not depend on node j if j>i
        w = self._create_params(self._root_node, latent_dim, 0)
        self._structure = [(self._root_node, [], w)]

        nodes = set(self._nodes) - {self._root_node}
        remaining = len(nodes)
        while remaining > 0:
            for node in nodes:
                regulated_by = self._r_edges[node].keys()
                if all([parent not in nodes for parent in regulated_by]):
                    w = self._create_params(node, latent_dim, len(regulated_by))
                    t = (node, regulated_by, w)
                    self._structure.append(t)
                    nodes = nodes - {node}
            assert len(nodes) < remaining
            remaining = len(nodes)

        super(GRI, self).build(input_shape)  # Be sure to call this at the end

    def call(self, z, **kwargs):
        """
        :param z: Noise tensor. Shape=(batch_size, LATENT_DIM)
        :return: synthetic gene expressions tensor
        """
        # Dictionary holding the current value of each node in the GRI.
        # Key: gene symbol. Value: gene Keras tensor
        units = {}

        # Compute feedforward pass for GRI layer
        x_bias = K.ones_like(z)  # Shape=(batch_size, latent_dim)
        x_bias = K.mean(x_bias, axis=-1)[:, None]  # Shape=(batch_size, 1)
        for t in self._structure:
            node, incoming, weights = t
            x_units = [units[in_node] for in_node in incoming]
            x_in = K.concatenate([x_bias, z] + x_units, axis=-1)  # Shape=(batch_size, 1 + latent_dim + nb_incoming)
            x_out = K.dot(x_in, weights)  # Shape=(batch_size, 1)
            units[node] = K.tanh(x_out)

        return K.concatenate([units[node] for node in self._nodes], axis=-1)  # Shape=(batch_size, nb_nodes)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self._output_dim


class ClipWeights():
    def __init__(self, min_value, max_value):
        self._min_value = min_value
        self._max_value = max_value

    def __call__(self, w):
        return K.clip(w, self._min_value, self._max_value)


class NormWeights():
    def __init__(self):
        pass

    def __call__(self, w):
        return K.softmax(w)


class ExperimentalNoise(Layer):
    def __init__(self, gene_stds, noise_strength=1, ampl_factor=1, **kwargs):
        self._gene_stds = K.constant(gene_stds)
        self._noise_strength = noise_strength
        self._ampl_factor = ampl_factor
        self._w = None
        super(ExperimentalNoise, self).__init__(**kwargs)

    @staticmethod
    def _noise_regularizer(alpha=0.013):
        def reg(weights):
            return -alpha * K.sum(K.square(weights))
        return reg

    def build(self, input_shape):
        self._w = self.add_weight(name='w',
                                  shape=(input_shape[1],),
                                  initializer='uniform',
                                  trainable=True,
                                  # constraint=NormWeights(),
                                  regularizer=self._noise_regularizer())
        super(ExperimentalNoise, self).build(input_shape)

    def call(self, x, **kwargs):
        noise = K.random_normal(K.shape(x), mean=0, stddev=1)
        # signal = (1-self._noise_strength) * x
        signal = x
        additive_noise = self._noise_strength * noise * self._w  # self._gene_stds
        out = signal + additive_noise
        out = K.clip(out, -2, 2)
        return self._ampl_factor * out

    def compute_output_shape(self, input_shape):
        return input_shape
