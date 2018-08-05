from keras.layers import Input, Dense, Dropout, LeakyReLU, Activation, Lambda
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from data_pipeline import load_data, save_synthetic, reg_network, split_train_test, clip_outliers
from utils import *
import keras.backend as K
from keras.callbacks import TensorBoard
import datetime
import tensorflow as tf
import warnings
import scipy.stats as stats
from layers import ExperimentalNoise
warnings.filterwarnings('ignore', message='Discrepancy between')

LATENT_DIM = 20


class BioGAN:
    def __init__(self, data, latent_dim=LATENT_DIM, discriminate_batch=False, max_replay_len=None):
        """
        Initialize GAN
        :param data: expression matrix. Shape=(nb_samples, nb_genes)
        :param latent_dim: number input noise units for the generator
        :param discriminate_batch: whether to discriminate a whole batch of samples rather than discriminating samples
               individually
        :param max_replay_len: size of the replay buffer
        """
        self._latent_dim = latent_dim
        self._data = data
        self._nb_samples, self._nb_genes = data.shape
        self._discriminate_batch = discriminate_batch
        self._max_replay_len = max_replay_len

        # Build and compile the discriminator
        self.discriminator = self._build_discriminator()
        optimizer = Adam(0.0002, 0.5)
        loss = 'binary_crossentropy'
        if self._discriminate_batch:
            loss = self._minibatch_binary_crossentropy

        self.discriminator.compile(loss=loss,
                                   optimizer=optimizer)
        self._gradients_discr = self._gradients_norm(self.discriminator)

        # Build the generator
        self.generator = self._build_generator()
        z = Input(shape=(self._latent_dim,))
        gen_out = self.generator(z)

        # Build the combined model
        optimizer = Adam(0.0002, 0.5, clipvalue=0.1)
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
        h = Dense(1000)(noise)
        h = LeakyReLU(0.3)(h)
        h = Dropout(0.5)(h)
        h = Dense(self._nb_genes)(h)
        # h = Activation('tanh')(h)
        # h = LeakyReLU(0.3)(h)
        self._noise_layer = ExperimentalNoise()
        h = self._noise_layer(h)
        # h = Activation('tanh')(h)
        model = Model(inputs=noise, outputs=h)
        model.summary()
        return model

    def _build_discriminator(self):
        """
        Build the discriminator
        """
        expressions_input = Input(shape=(self._nb_genes,))
        h = Dense(1000)(expressions_input)
        h = LeakyReLU(0.3)(h)
        h = Dropout(0.5)(h)
        h = Dense(1, activation='sigmoid')(h)

        if self._discriminate_batch:
            h = Lambda(lambda x: K.mean(x, keepdims=True))(h)  # Discriminates the whole batch. Shape=(1,)

        model = Model(inputs=expressions_input, outputs=h)
        model.summary()
        return model

    @staticmethod
    def _minibatch_binary_crossentropy(y_true, y_pred):
        """
        Loss function for minibatch discrimination
        :param y_true: tensor with true label. Shape=(batch_size, 1)
        :param y_pred: tensor with probability P(x_batch=1). Shape=(1, 1)
        :return: minibatch binary crossentropy
        """
        print(y_true.get_shape())
        y_true_batch = y_true[0, None]
        print(y_true_batch.get_shape())
        return K.binary_crossentropy(y_true_batch, y_pred)

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

    def train(self, epochs, batch_size=32):
        """
        Trains the GAN
        :param epochs: Number of epochs
        :param batch_size: Batch size
        """
        # sess = tf.Session()
        # K.set_session(sess)
        # sess.run(tf.global_variables_initializer())

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
            fake = np.random.uniform(low=0, high=0.3, size=(batch_size, 1))  # Label smoothing
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

    # Clip outliers
    mean = np.mean(expr, axis=0)
    std = np.std(expr, axis=0)
    std_clip = 2
    # expr_train = clip_outliers(expr_train, mean, std, std_clip)

    # Standardize data
    r_min = expr.min(axis=0)
    r_max = expr.max(axis=0)
    expr_train = (expr_train - mean) / (std_clip * std)
    # expr_train = np.tanh(expr_train)

    # Scale standard deviations (0 to 1)
    std_min = std.min()
    std_max = std.max()
    scaled_stds = (std - std_min)/(std_max - std_min)

    # Train GAN
    biogan = BioGAN(expr_train, latent_dim=LATENT_DIM)
    biogan.train(epochs=3000, batch_size=32)

    # Generate synthetic data
    s_expr = biogan.generate_batch(expr_train.shape[0])  # expr_test.shape[0]
    # s_expr = np.arctanh(s_expr)
    s_expr = std_clip * s_expr * std + mean

    # Save generated data
    file_name = 'EColi_n{}_r{}_e{}_d{}'.format(len(gene_symbols), root_gene, minimum_evidence, max_depth)
    save_synthetic(file_name, s_expr, gene_symbols)
