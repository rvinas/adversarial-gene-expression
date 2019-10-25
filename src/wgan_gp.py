from keras.layers import Input, Dense, Dropout, LeakyReLU, Activation, Lambda, BatchNormalization, Concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop
import keras.backend as K
from keras.callbacks import TensorBoard
import numpy as np
from data_pipeline import load_data, save_synthetic, reg_network, split_train_test, clip_outliers, normalize
from utils import *
import datetime
import tensorflow as tf
import warnings
from layers import GeneWiseNoise, ClipWeights, MinibatchDiscrimination, CorrDiscr
from keras.layers.merge import _Merge
from functools import partial

warnings.filterwarnings('ignore', message='Discrepancy between')

RETRAIN = False
LATENT_DIM = 10
DEFAULT_NOISE_RATE = 10
CHECKPOINTS_DIR = '../checkpoints'


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class gGAN:
    def __init__(self, data, gene_symbols, latent_dim=LATENT_DIM, noise_rate=DEFAULT_NOISE_RATE,
                 discriminate_batch=False,
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
        if discriminate_batch:
            raise NotImplementedError

        self._latent_dim = latent_dim
        self._noise_rate = noise_rate
        self._data = data
        self._gene_symbols = gene_symbols
        self._nb_samples, self._nb_genes = data.shape
        self._discriminate_batch = discriminate_batch
        self._max_replay_len = max_replay_len

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build players
        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        self.generator.trainable = False

        # Inputs
        z_disc = Input(shape=(self._latent_dim,))
        real_inp = Input(shape=(self._nb_genes,))

        # Generate image based of noise (fake sample)
        fake_inp = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.discriminator(fake_inp)
        valid = self.discriminator(real_inp)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_inp, fake_inp])
        # Determine validity of weighted sample
        validity_interpolated = self.discriminator(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_inp, z_disc],
                                  outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, 10])

        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.discriminator.trainable = False
        self.generator.trainable = True
        # self._noise_layer.trainable = False

        # Sampled noise for input to generator
        z_gen = Input(shape=(self._latent_dim,))
        # Generate images based of noise
        fake_inp = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.discriminator(fake_inp)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

        # Noise model
        """
        self.generator.trainable = False
        self._noise_layer.trainable = True
        self.noise_model = Model(z_gen, valid)
        self.noise_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        """

        # -----------------------------
        # Prepare TensorBoard callbacks
        # -----------------------------

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

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def _build_generator(self):
        """
        Build the generator
        """
        # rate = 0.3
        # nb_initial_units = int((1 - rate) * self._latent_dim)
        noise = Input(shape=(self._latent_dim,))
        # input_1 = Lambda(lambda x: x[:, :nb_initial_units])(noise)
        # h = Dropout(0.5)(h)
        h = Dense(64)(noise)
        h = BatchNormalization()(h)
        h = LeakyReLU(0.3)(h)
        h = Dropout(0.5)(h)
        h = Dense(256)(h)
        h = BatchNormalization()(h)
        h = LeakyReLU(0.3)(h)
        h = Dropout(0.5)(h)
        h = Dense(256)(h)
        h = BatchNormalization()(h)
        h = LeakyReLU(0.3)(h)
        h = Dropout(0.5)(h)
        # input_2 = Lambda(lambda x: x[:, nb_initial_units:])(noise)
        # h = Concatenate()([h, input_2])
        h = Dense(self._nb_genes)(h)
        # h = Activation('tanh')(h)
        # h = LeakyReLU(0.3)(h)
        # self._noise_layer = GeneWiseNoise(self._noise_rate)
        # h = self._noise_layer(h)
        # h = Activation('tanh')(h)
        model = Model(inputs=noise, outputs=h)
        model.summary()
        return model

    def _build_discriminator(self, clipvalue=0.01):
        """
        Build the discriminator
        """
        expressions_input = Input(shape=(self._nb_genes,))
        h = expressions_input
        # h = Dropout(0.1)(h)
        batch_features_2 = MinibatchDiscrimination(2, 5)(h)
        h = Concatenate()([h, batch_features_2])
        # batch_features_1 = CorrDiscr()(h)
        h = Dense(100)(h)
        h = LeakyReLU(0.3)(h)
        h = Dropout(0.5)(h)
        h = Dense(100)(h)
        h = LeakyReLU(0.3)(h)
        h = Dropout(0.5)(h)
        h = Dense(1)(h)
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

        # Best score for correlation of distance gene matrices
        best_gdxdz = 0

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty

        for epoch in range(epochs):

            d_losses = []
            for _ in range(self.n_critic):
                # ----------------------
                #  Train Discriminator
                # ----------------------

                # Select random samples
                idxs = np.random.randint(0, self._nb_samples, batch_size)
                x_real = self._data[idxs, :]

                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (batch_size, self._latent_dim))

                # Train the discriminator (real classified as ones and generated as zeros)
                d_loss = self.critic_model.train_on_batch([x_real, noise], [valid, fake, dummy])

                d_losses.append(d_loss[0])

            # Add discriminator loss to TensorBoard
            d_loss = np.mean(d_losses)
            self._write_log(self._callback_discr, ['train_loss'], [d_loss], epoch)

            # ----------------------
            #  Report gradient norms
            # ----------------------

            # Gradient norm TensorBoard log
            if (epoch + 1) % 100 == 0:

                # ----------------------
                # Evaluate and save the model when good gamma(D^X, D^Z)
                # ----------------------
                s_expr = self.generate_batch(self._data.shape[0])
                gamma_dx_dz, gamma_dx_tx, gamma_dz_tz, gamma_tx_tz = gamma_coefficients(self._data, s_expr)
                r_tf_tg_corr_flat, r_tg_tg_corr_flat = compute_tf_tg_corrs(self._data, self._gene_symbols,
                                                                           flat=False)
                s_tf_tg_corr_flat, s_tg_tg_corr_flat = compute_tf_tg_corrs(s_expr, self._gene_symbols, flat=False)
                psi_dx_dz = psi_coefficient(r_tf_tg_corr_flat, s_tf_tg_corr_flat)
                phi_dx_dz = phi_coefficient(r_tg_tg_corr_flat, s_tg_tg_corr_flat)

                self._write_log(self._callback_gen,
                                ['Gamma(D^X, D^Z)', 'Gamma(D^Z, T^Z)', 'Gamma(T^X, T^Z)', 'Psi(D^X, D^Z)',
                                 'Phi(D^X, D^Z)'],
                                [gamma_dx_dz, gamma_dz_tz, gamma_tx_tz, psi_dx_dz, phi_dx_dz],
                                epoch)
                print('Gamma(D^X, D^Z): {}'.format(gamma_dx_dz))
                if file_name is not None and gamma_dx_dz > best_gdxdz:
                    best_gdxdz = gamma_dx_dz
                    print('Saving model ...')
                    self.save_model(file_name)

            # ----------------------
            #  Train Generator
            # ----------------------

            # Train the generator
            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Train noise model
            # n_loss = self.noise_model.train_on_batch(noise, valid)
            # g_loss = (g_loss + n_loss) / 2

            # Add generator loss to TensorBoard
            self._write_log(self._callback_gen, ['train_loss'], [g_loss], epoch)

            # ----------------------
            #  Plot the progress
            # ----------------------
            print('{} [D loss: {:.4f}] [G loss: {:.4f}]'.format(epoch, d_loss, g_loss))

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

    def load_model(self, name):
        """
        Loads model from CHECKPOINTS_DIR
        :param name: model id
        """
        """self.discriminator = load_model('{}/discr/{}.h5'.format(CHECKPOINTS_DIR, name),
                                        custom_objects={'ClipWeights': ClipWeights,
                                                        'MinibatchDiscrimination': MinibatchDiscrimination})"""
        self.generator = load_model('{}/gen/{}.h5'.format(CHECKPOINTS_DIR, name),
                                    custom_objects={'GeneWiseNoise': GeneWiseNoise},
                                    compile=False)
        self._latent_dim = self.generator.input_shape[-1]


if __name__ == '__main__':
    # Load data
    root_gene = 'CRP'  # Set to 'CRP' to select the CRP hierarchy. Set to None to use full set of genes
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
    data = normalize(expr_train)

    # Train GAN
    ggan = gGAN(data, gene_symbols, latent_dim=LATENT_DIM, noise_rate=DEFAULT_NOISE_RATE)
    if RETRAIN:
        ggan.load_model(file_name)
    ggan.train(epochs=4000, file_name=file_name, batch_size=32)

    # Generate synthetic data
    mean = np.mean(expr_train, axis=0)
    std = np.std(expr_train, axis=0)
    r_min = expr_train.min()
    r_max = expr_train.max()
    s_expr = ggan.generate_batch(expr_train.shape[0])

    # Save generated data
    save_synthetic(file_name, s_expr, gene_symbols)

    # Save model
    # ggan.save_model(file_name)
    # NOTE: The model is saved while training now
