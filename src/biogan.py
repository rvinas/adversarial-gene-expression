from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from data_pipeline import load_data, save_synthetic, reg_network, split_train_test
from utils import *
from keras import backend as K
from keras.engine.topology import Layer
import warnings

warnings.filterwarnings('ignore', message='Discrepancy between')

LATENT_DIM = 10


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


class BioGAN():
    def __init__(self, data, root_node, nodes, edges, latent_dim=LATENT_DIM):
        self._latent_dim = latent_dim
        self._data = data
        self._nb_samples, self._nb_genes = data.shape
        self._root_node = root_node
        self._nodes = nodes
        self._edges = edges

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self._latent_dim,))
        gen_out = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(gen_out)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        # Build generator
        noise = Input(shape=(self._latent_dim,))
        h = Dense(400, activation='relu')(noise)
        h = Dense(self._nb_genes, activation='tanh')(h)
        # h = GRI(self._root_node, self._nodes, self._edges)(h)
        model = Model(inputs=noise, outputs=h)
        model.summary()
        return model

    def build_discriminator(self):
        # Build discriminator
        expressions_input = Input(shape=(self._nb_genes,))
        h = Dense(20, activation='relu')(expressions_input)
        h = Dropout(0.2)(h)
        h = Dense(1, activation='sigmoid')(h)
        model = Model(inputs=expressions_input, outputs=h)
        model.summary()
        return model

    def train(self, epochs, batch_size=32, save_interval=50):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, self._nb_samples, batch_size)
            x_real = self._data[idx, :]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self._latent_dim))
            x_fake = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(x_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(x_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                pass

    def generate_batch(self, batch_size=32):
        noise = np.random.normal(0, 1, (batch_size, self._latent_dim))
        return self.generator.predict(noise)


if __name__ == '__main__':
    # Load data
    root_gene = 'CRP'
    minimum_evidence = 'weak'
    max_depth = np.inf
    expr, gene_symbols, sample_names = load_data(root_gene=root_gene,
                                                 minimum_evidence=minimum_evidence,
                                                 max_depth=max_depth)
    # Split data into train and test sets
    train_idxs, test_idxs = split_train_test(sample_names)
    expr_train = expr[train_idxs, :]
    expr_test = expr[test_idxs, :]

    # Get regulatory network
    nodes, edges = reg_network(root_gene=root_gene,
                               gene_symbols=gene_symbols,
                               minimum_evidence=minimum_evidence,
                               max_depth=max_depth,
                               break_loops=True)
    root_node = root_gene.lower()

    # Standardize data
    mean = np.mean(expr, axis=0)
    std = np.std(expr, axis=0)
    expr_train = (expr_train - mean) / std

    # Train GAN
    biogan = BioGAN(expr_train, root_node, gene_symbols, edges)
    biogan.train(epochs=10000, batch_size=32, save_interval=50)
    s_expr = biogan.generate_batch(expr_test.shape[0])
    s_expr = std * s_expr + mean
    file_name = 'EColi_n{}_r{}_e{}_d{}'.format(len(gene_symbols), root_gene, minimum_evidence, max_depth)
    save_synthetic(file_name, s_expr, gene_symbols)
