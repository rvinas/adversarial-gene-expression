from keras.layers import Input, Dense, Dropout, LeakyReLU, Activation, Lambda, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import TensorBoard
import numpy as np
from data_pipeline import load_data, save_synthetic, reg_network, split_train_test, clip_outliers
from utils import *
import datetime
import tensorflow as tf
import warnings
from layers import GeneWiseNoise, ClipWeights

warnings.filterwarnings('ignore', message='Discrepancy between')

LATENT_DIM = 20
DEFAULT_NOISE_RATE = 0.25
CHECKPOINTS_DIR = '../checkpoints'


# TODO: Pass function of evaluation scores in train instead of gene symbols

class gGAN:
    def __init__(self, data, gene_symbols, latent_dim=LATENT_DIM, noise_rate=DEFAULT_NOISE_RATE, discriminate_batch=False,
                 max_replay_len=None):
        """
        Initialize GAN
        :param data: expression matrix. Shape=(nb_samples, nb_genes)
        :param gene_symbols: list of gene symbols. Shape=(nb_genes,)
        :param latent_dim: number input noise units for the generator
        :param noise_rate: rate of noise being added in the gene-wise layer of the generator (psi/nb_genes).
        :param discriminate_batch: whether to discriminate a whole batch of samples rather than discriminating samples
               individually. NOTE: Not implemented
        :param max_replay_len: size of the replay buffer
        """
        self._latent_dim = latent_dim
        self._noise_rate = noise_rate
        self._data = data
        self._gene_symbols = gene_symbols
        self._nb_samples, self._nb_genes = data.shape
        self._discriminate_batch = discriminate_batch
        self._max_replay_len = max_replay_len

        # Build and compile the discriminator
        self.discriminator = self._build_discriminator()
        optimizer = Adam(0.0002, 0.5)
        loss = 'binary_crossentropy'
        if self._discriminate_batch:
            # loss = self._minibatch_binary_crossentropy
            raise NotImplementedError

        self.discriminator.compile(loss=loss,
                                   optimizer=optimizer)
        self._gradients_discr = self._gradients_norm(self.discriminator)

        # Build the generator
        self.generator = self._build_generator()
        z = Input(shape=(self._latent_dim,))
        gen_out = self.generator(z)

        # Build the combined model
        optimizer = Adam(0.0002, 0.5)
        self.discriminator.trainable = False
        valid = self.discriminator(gen_out)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        self._gradients_gen = self._gradients_norm(self.combined)

        # Prepare TensorBoard callbacks
        time_now = str(datetime.datetime.now())
        log_path = '../logs/discr/{}'.format(time_now)
        self._callback_discr = TensorBoard(log_path)
        self._callback_discr.set_model(self.discriminator)
        log_path = '../logs/gen/{}'.format(time_now)
        self._callback_gen = TensorBoard(log_path)
        self._callback_gen.set_model(self.generator)

        # Initialize replay buffer
        noise = np.random.normal(0, 1, (32, self._latent_dim))
        x_fake = self.generator.predict(noise)
        self._replay_buffer = np.array(x_fake)

    def _build_generator(self):
        """
        Build the generator
        """
        noise = Input(shape=(self._latent_dim,))
        h = noise
        # h = Dropout(0.5)(h)
        h = Dense(1000)(h)
        h = LeakyReLU(0.3)(h)
        h = Dropout(0.5)(h)
        h = Dense(self._nb_genes)(h)
        # h = Activation('tanh')(h)
        # h = LeakyReLU(0.3)(h)
        self._noise_layer = GeneWiseNoise(self._noise_rate)
        h = self._noise_layer(h)
        # h = Activation('tanh')(h)
        model = Model(inputs=noise, outputs=h)
        model.summary()
        return model

    def _build_discriminator(self, clipvalue=0.5):
        """
        Build the discriminator
        """
        expressions_input = Input(shape=(self._nb_genes,))
        h = Dense(1000, kernel_constraint=ClipWeights(-clipvalue, clipvalue))(expressions_input)
        h = LeakyReLU(0.3)(h)
        h = Dropout(0.5)(h)
        h = Dense(1, activation='sigmoid', kernel_constraint=ClipWeights(-clipvalue, clipvalue))(h)
        model = Model(inputs=expressions_input, outputs=h)
        model.summary()
        return model

    @staticmethod
    def _write_log(callback, names, logs, epoch):
        """
        Write log to TensorBoard callback
        :param callback: TensorBoard callback
        :param names: list of names for each log. Shape=(nb_logs,)
        :param logs: list of scalars. Shape=(nb_logs,)
        :param epoch: epoch number
        """
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, epoch)
            callback.writer.flush()

    @staticmethod
    def _gradients_norm(model):
        """
        Create Keras function to compute Euclidean norm of gradient vector
        :param model: Keras model for which the norm will be computed
        :return: Keras function to compute Euclidean norm of gradient vector
        """
        grads = K.gradients(model.total_loss, model.trainable_weights)
        summed_squares = tf.stack([K.sum(K.square(g)) for g in grads])
        norm = K.sqrt(K.sum(summed_squares))

        input_tensors = [model.inputs[0],  # input data
                         model.sample_weights[0],  # how much to weight each sample by
                         model.targets[0],  # labels
                         K.learning_phase(),  # train or test mode
                         ]

        return K.function(inputs=input_tensors, outputs=[norm])

    def train(self, epochs, file_name=None, batch_size=32):
        """
        Trains the GAN
        :param epochs: Number of epochs
        :param file_name: Name of the .h5 file in checkpoints. If None, the model won't be saved
        :param batch_size: Batch size
        """
        # sess = tf.Session()
        # K.set_session(sess)
        # sess.run(tf.global_variables_initializer())

        # Best score for correlation of distance gene matrices
        best_gdxdz = 0

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Select random samples
            idxs = np.random.randint(0, self._nb_samples, batch_size)
            x_real = self._data[idxs, :]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self._latent_dim))
            x_fake = self.generate_batch(batch_size)  # self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            valid = np.random.uniform(low=0.7, high=1, size=(batch_size, 1))  # Label smoothing
            fake = np.zeros(
                shape=(batch_size, 1))  # np.random.uniform(low=0, high=0.3, size=(batch_size, 1))  # Label smoothing
            d_loss_real = self.discriminator.train_on_batch(x_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(x_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Add discriminator loss to TensorBoard
            self._write_log(self._callback_discr, ['train_loss'], [d_loss], epoch)

            # Train discr. using replay buffer
            if self._max_replay_len is not None:
                idxs = np.random.choice(len(self._replay_buffer), batch_size)
                d_loss_replay = self.discriminator.train_on_batch(self._replay_buffer[idxs, :], fake)

                # Add discriminator loss to TensorBoard
                self._write_log(self._callback_discr, ['replay_loss'], [d_loss_replay], epoch)

                # Add data to replay buffer
                if len(self._replay_buffer) < self._max_replay_len:
                    self._replay_buffer = np.concatenate((self._replay_buffer, x_fake), axis=0)
                elif np.random.randint(low=0, high=1) < 0.999 ** epoch:
                    idxs = np.random.choice(len(self._replay_buffer), batch_size, replace=False)
                    self._replay_buffer[idxs, :] = x_fake

            # ----------------------
            #  Train Generator
            # ----------------------

            # Train the generator
            valid = np.random.uniform(low=0.7, high=1, size=(batch_size, 1))  # np.ones(shape=(batch_size, 1))
            g_loss = self.combined.train_on_batch(noise, valid)

            # Add generator loss to TensorBoard
            self._write_log(self._callback_gen, ['train_loss'], [g_loss], epoch)

            # ----------------------
            #  Report gradient norms
            # ----------------------

            # Gradient norm TensorBoard log
            if epoch % 100 == 0:
                # Get discriminator gradients
                grad_inputs = [noise,  # data
                               np.ones(shape=(batch_size,)),  # Sample weights
                               valid,  # labels
                               0  # set learning phase in TEST mode
                               ]
                g_gen = self._gradients_gen(grad_inputs)[0]

                # Get discriminator gradients
                grad_inputs = [x_real,  # data
                               np.ones(shape=(batch_size,)),  # Sample weights
                               valid,  # labels
                               0  # set learning phase in TEST mode
                               ]
                g_discr_real = self._gradients_discr(grad_inputs)[0]
                grad_inputs = [x_fake,  # data
                               np.ones(shape=(batch_size,)),  # Sample weights
                               fake,  # labels
                               0  # set learning phase in TEST mode
                               ]
                g_discr_fake = self._gradients_discr(grad_inputs)[0]
                g_discr = (g_discr_real + g_discr_fake) / 2

                # Add information to TensorBoard log
                self._write_log(self._callback_discr, ['gradient_norm'], [g_discr], epoch)
                self._write_log(self._callback_gen, ['gradient_norm'], [g_gen], epoch)

            # ----------------------
            #  Plot the progress
            # ----------------------
            print('{} [D loss: {:.4f}] [G loss: {:.4f}]'.format(epoch, d_loss, g_loss))

            # ----------------------
            # Evaluate and save the model when good gamma(D^X, D^Z)
            # ----------------------
            if (epoch+1) % 100 == 0:
                s_expr = self.generate_batch(self._data.shape[0])
                gamma_dx_dz, gamma_dx_tx, gamma_dz_tz, gamma_tx_tz = gamma_coefficients(self._data, s_expr)
                r_tf_tg_corr_flat, r_tg_tg_corr_flat = compute_tf_tg_corrs(self._data, self._gene_symbols, flat=False)
                s_tf_tg_corr_flat, s_tg_tg_corr_flat = compute_tf_tg_corrs(s_expr, self._gene_symbols, flat=False)
                psi_dx_dz = psi_coefficient(r_tf_tg_corr_flat, s_tf_tg_corr_flat)
                theta_dx_dz = theta_coefficient(r_tg_tg_corr_flat, s_tg_tg_corr_flat)

                self._write_log(self._callback_gen,
                                ['Gamma(D^X, D^Z)', 'Gamma(D^Z, T^Z)', 'Gamma(T^X, T^Z)', 'Psi(D^X, D^Z)', 'Phi(D^X, D^Z)'],
                                [gamma_dx_dz, gamma_dz_tz, gamma_tx_tz, psi_dx_dz, theta_dx_dz],
                                epoch)
                print('Gamma(D^X, D^Z): {}'.format(gamma_dx_dz))
                if file_name is not None and gamma_dx_dz > best_gdxdz:
                    best_gdxdz = gamma_dx_dz
                    print('Saving model ...')
                    self.save_model(file_name)

        # Print noise layer norm
        # noise_layer_norm = sess.run(self._noise_layer.get_weights_norm())
        # print('Noise layer norm: {}'.format(noise_layer_norm))
        # sess.close()

    def generate_batch(self, batch_size=32):
        """
        Generate a batch of samples using the generator
        :param batch_size: Batch size
        :return: Artificial samples generated by the generator. Shape=(batch_size, nb_genes)
        """
        noise = np.random.normal(0, 1, (batch_size, self._latent_dim))
        pred = self.generator.predict(noise)

        return pred

    def discriminate(self, expr):
        """
        Discriminates a batch of samples
        :param expr: expressions matrix. Shape=(nb_samples, nb_genes)
        :return: for each sample, probability that it comes from the real distribution
        """
        return self.discriminator.predict(expr)

    def save_model(self, name):
        """
        Saves model to CHECKPOINTS_DIR
        :param name: model id
        """
        self.discriminator.trainable = True
        self.discriminator.save('{}/discr/{}.h5'.format(CHECKPOINTS_DIR, name))
        self.discriminator.trainable = False
        for layer in self.discriminator.layers:  # https://github.com/keras-team/keras/issues/9589
            layer.trainable = False
        self.generator.save('{}/gen/{}.h5'.format(CHECKPOINTS_DIR, name), include_optimizer=False)
        self.combined.save('{}/gan/{}.h5'.format(CHECKPOINTS_DIR, name))

    def load_model(self, name):
        """
        Loads model from CHECKPOINTS_DIR
        :param name: model id
        """
        self.discriminator = load_model('{}/discr/{}.h5'.format(CHECKPOINTS_DIR, name),
                                        custom_objects={'ClipWeights': ClipWeights})
        self.generator = load_model('{}/gen/{}.h5'.format(CHECKPOINTS_DIR, name),
                                    custom_objects={'GeneWiseNoise': GeneWiseNoise},
                                    compile=False)
        self.combined = load_model('{}/gan/{}.h5'.format(CHECKPOINTS_DIR, name),
                                   custom_objects={'GeneWiseNoise': GeneWiseNoise,
                                                   'ClipWeights': ClipWeights})
        self._latent_dim = self.generator.input_shape[-1]


def normalize(expr, kappa=1):
    """
    Normalizes expressions to make each gene have mean 0 and std kappa^-1
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param kappa: kappa^-1 is the gene std
    :return: normalized expressions
    """
    mean = np.mean(expr, axis=0)
    std = np.std(expr, axis=0)
    return (expr - mean) / (kappa * std)


def restore_scale(expr, mean, std):
    """
    Makes each gene j have mean_j and std_j
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param mean: vector of gene means. Shape=(nb_genes,)
    :param std: vector of gene stds. Shape=(nb_genes,)
    :return: Rescaled gene expressions
    """
    return expr * std + mean


def clip_outliers(expr, r_min, r_max):
    """
    Clips expression values to make them be between r_min and r_max
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param r_min: minimum expression value (float)
    :param r_max: maximum expression value (float)
    :return: Clipped expression matrix
    """
    expr_c = np.copy(expr)
    expr_c[expr_c < r_min] = r_min
    expr_c[expr_c > r_max] = r_max
    return expr_c


def generate_data(gan, size, mean, std, r_min, r_max):
    """
    Generates size samples from generator
    :param gan: Keras GAN
    :param size: Number of samples to generate
    :param mean: vector of gene means. Shape=(nb_genes,)
    :param std: vector of gene stds. Shape=(nb_genes,)
    :param r_min: vector of minimum expression values for each gene
    :param r_max: vector of maximum expression values for each gene
    :return: matrix of expressions
    """
    expr = gan.generate_batch(size)
    expr = normalize(expr)
    expr = restore_scale(expr, mean, std)
    expr = clip_outliers(expr, r_min, r_max)
    return expr


if __name__ == '__main__':
    # Load data
    root_gene = None  # Set to 'CRP' to select the CRP hierarchy. Set to None to use full set of genes
    minimum_evidence = 'weak'
    max_depth = np.inf
    expr, gene_symbols, sample_names = load_data(root_gene=root_gene,
                                                 minimum_evidence=minimum_evidence,
                                                 max_depth=max_depth)
    file_name = 'EColi_n{}_r{}_e{}_d{}'.format(len(gene_symbols), root_gene, minimum_evidence, max_depth)

    # Split data into train and test sets
    train_idxs, test_idxs = split_train_test(sample_names)
    expr_train = expr[train_idxs, :]
    expr_test = expr[test_idxs, :]

    # Standardize data
    data = normalize(expr_train, kappa=2)

    # Train GAN
    ggan = gGAN(data, gene_symbols, latent_dim=LATENT_DIM, noise_rate=DEFAULT_NOISE_RATE)
    ggan.train(epochs=4000, file_name=file_name, batch_size=32)

    # Generate synthetic data
    mean = np.mean(expr_train, axis=0)
    std = np.std(expr_train, axis=0)
    r_min = expr_train.min()
    r_max = expr_train.max()
    s_expr = generate_data(ggan, expr_train.shape[0], mean, std, r_min, r_max)

    # Save generated data
    save_synthetic(file_name, s_expr, gene_symbols)

    # Save model
    # ggan.save_model(file_name)
    # NOTE: The model is saved while training now
