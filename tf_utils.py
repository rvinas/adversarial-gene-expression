import tensorflow as tf
import os
import datetime
import numpy as np

tfk = tf.keras
tfkl = tf.keras.layers


CHECKPOINTS_DIR = '../checkpoints/'


# ------------------
# LIMIT GPU USAGE
# ------------------

def limit_gpu(gpu_idx=0, mem=2 * 1024):
    """
    Limits gpu usage
    :param gpu_idx: Use this gpu
    :param mem: Maximum memory in bytes
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            # Use a single gpu
            tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')

            # Limit memory
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem)])  # 2 GB
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


# ------------------
# WGAN-GP
# ------------------

def make_generator(x_dim, vocab_sizes, nb_numeric, h_dims=None, z_dim=10):
    """
    Make generator
    :param x_dim: Number of genes
    :param vocab_sizes: List of ints. Size of vocabulary for each categorical covariate
    :param nb_numeric: Number of numerical covariates
    :param h_dims: Number of units for each hidden layer
    :param z_dim: Number of input units
    :return: generator
    """
    # Define inputs
    z = tfkl.Input((z_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.int32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)

    embed_cats = []
    total_emb_dim = 0

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)
        total_emb_dim += emb_dim
    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    embeddings = tfkl.Concatenate(axis=-1)([num, embeddings])
    total_emb_dim += nb_numeric

    def make_generator_emb(x_dim, emb_dim, h_dims=None, z_dim=10):
        if h_dims is None:
            h_dims = [256, 256]

        z = tfkl.Input((z_dim,))
        t_emb = tfkl.Input((emb_dim,), dtype=tf.float32)
        h = tfkl.Concatenate(axis=-1)([z, t_emb])
        for d in h_dims:
            h = tfkl.Dense(d)(h)
            h = tfkl.ReLU()(h)
        h = tfkl.Dense(x_dim)(h)
        model = tfk.Model(inputs=[z, t_emb], outputs=h)
        return model

    gen_emb = make_generator_emb(x_dim=x_dim,
                                 emb_dim=total_emb_dim,
                                 h_dims=h_dims,
                                 z_dim=z_dim)
    model = tfk.Model(inputs=[z, cat, num], outputs=gen_emb([z, embeddings]))
    model.summary()
    return model


def make_discriminator(x_dim, vocab_sizes, nb_numeric, h_dims=None):
    """
    Make discriminator
    :param x_dim: Number of genes
    :param vocab_sizes: List of ints. Size of vocabulary for each categorical covariate
    :param nb_numeric: Number of numerical covariates
    :param h_dims: Number of units for each hidden layer
    :return: discriminator
    """
    if h_dims is None:
        h_dims = [256, 256]

    x = tfkl.Input((x_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.int32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)

    embed_cats = []

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)

    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    h = tfkl.Concatenate(axis=-1)([x, num, embeddings])
    for d in h_dims:
        h = tfkl.Dense(d)(h)
        h = tfkl.ReLU()(h)
    h = tfkl.Dense(1)(h)
    model = tfk.Model(inputs=[x, cat, num], outputs=h)
    return model


def wasserstein_loss(y_true, y_pred):
    """
    Wasserstein loss
    """
    return tf.reduce_mean(y_true * y_pred)


def generator_loss(fake_output):
    """
    Generator loss
    """
    return wasserstein_loss(-tf.ones_like(fake_output), fake_output)


def gradient_penalty(f, real_output, fake_output):
    """
    Gradient penalty of WGAN-GP
    :param f: discriminator function without sample covariates as input
    :param real_output: real data
    :param fake_output: fake data
    :return: gradient penalty
    """
    alpha = tf.random.uniform([real_output.shape[0], 1], 0., 1.)
    diff = fake_output - real_output
    inter = real_output + (alpha * diff)
    with tf.GradientTape() as t:
        t.watch(inter)
        pred = f(inter)
    grad = t.gradient(pred, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))  # real_output
    gp = tf.reduce_mean((slopes - 1.) ** 2)
    return gp


def discriminator_loss(real_output, fake_output):
    """
    Critic loss
    """
    real_loss = wasserstein_loss(-tf.ones_like(real_output), real_output)
    fake_loss = wasserstein_loss(tf.ones_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_disc(x, z, cc, nc, gen, disc, disc_opt, grad_penalty_weight=10, p_aug=0, norm_scale=0.5):
    """
    Train critic
    :param x: Batch of expression data
    :param z: Batch of latent variables
    :param cc: Batch of categorical covariates
    :param nc: Batch of numerical covariates
    :param gen: Generator
    :param disc: Critic
    :param disc_opt: Critic optimizer
    :param grad_penalty_weight: Weight for the gradient penalty
    :return: Critic loss
    """
    bs = z.shape[0]
    nb_genes = gen.output.shape[-1]
    augs = np.random.binomial(1, p_aug, bs)

    with tf.GradientTape() as disc_tape:
        # Generator forward pass
        x_gen = gen([z, cc, nc], training=False)

        # Perform augmentations
        x_gen = x_gen + augs[:, None] * np.random.normal(0, norm_scale, nb_genes)
        x = x + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))

        # Forward pass on discriminator
        disc_out = disc([x_gen, cc, nc], training=True)
        disc_real = disc([x, cc, nc], training=True)

        # Compute losses
        disc_loss = discriminator_loss(disc_real, disc_out) \
                    + grad_penalty_weight * gradient_penalty(lambda x: disc([x, cc, nc], training=True), x, x_gen)

    disc_grad = disc_tape.gradient(disc_loss, disc.trainable_variables)
    disc_opt.apply_gradients(zip(disc_grad, disc.trainable_variables))

    return disc_loss


@tf.function
def train_gen(z, cc, nc, gen, disc, gen_opt, p_aug=0, norm_scale=1):
    """
    Train generator
    :param z: Batch of latent variables
    :param cc: Batch of categorical covariates
    :param nc: Batch of numerical covariates
    :param gen: Generator
    :param disc: Critic
    :param gen_opt: Generator optimiser
    :return: Generator loss
    """
    bs = z.shape[0]
    nb_genes = gen.output.shape[-1]
    augs = np.random.binomial(1, p_aug, bs)

    with tf.GradientTape() as gen_tape:
        # Generator forward pass
        x_gen = gen([z, cc, nc], training=True)

        # Perform augmentations
        x_gen = x_gen + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))

        # Forward pass on discriminator
        disc_out = disc([x_gen, cc, nc], training=False)

        # Compute losses
        gen_loss = generator_loss(disc_out)

    gen_grad = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grad, gen.trainable_variables))

    return gen_loss


def train(dataset, cat_covs, num_covs, z_dim, epochs, batch_size, gen, disc, score_fn, save_fn,
          gen_opt=None, disc_opt=None, nb_critic=5, verbose=True, checkpoint_dir='./checkpoints/cpkt',
          log_dir='./logs/', patience=10, p_aug=0, norm_scale=0.5):
    """
    Train model
    :param dataset: Numpy matrix with data. Shape=(nb_samples, nb_genes)
    :param cat_covs: Categorical covariates. Shape=(nb_samples, nb_cat_covs)
    :param num_covs: Numerical covariates. Shape=(nb_samples, nb_num_covs)
    :param z_dim: Int. Latent dimension
    :param epochs: Number of training epochs
    :param batch_size: Batch size
    :param gen: Generator model
    :param disc: Critic model
    :param gen_opt: Generator optimiser
    :param disc_opt: Critic optimiser
    :param score_fn: Function that computes the score: Generator => score.
    :param save_fn:  Function that saves the model.
    :param nb_critic: Number of critic updates for each generator update
    :param verbose: Print details
    :param checkpoint_dir: Where to save checkpoints
    :param log_dir: Where to save logs
    :param patience: Number of epochs without improving after which the training is halted
    """
    # Optimizers
    if gen_opt is None:
        gen_opt = tfk.optimizers.RMSprop(5e-4)
    if disc_opt is None:
        disc_opt = tfk.optimizers.RMSprop(5e-4)

    # Set up logs and checkpoints
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    gen_log_dir = log_dir + current_time + '/gen'
    disc_log_dir = log_dir + current_time + '/disc'
    gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
    disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)

    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                     discriminator_optimizer=disc_opt,
                                     generator=gen,
                                     discriminator=disc)

    gen_losses = tfk.metrics.Mean('gen_loss', dtype=tf.float32)
    disc_losses = tfk.metrics.Mean('disc_loss', dtype=tf.float32)
    best_score = -np.inf
    initial_patience = patience

    for epoch in range(epochs):
        for i in range(0, len(dataset), batch_size):
            x = dataset[i: i + batch_size, :]
            cc = cat_covs[i: i + batch_size, :]
            nc = num_covs[i: i + batch_size, :]

            # Train critic
            disc_loss = None
            for _ in range(nb_critic):
                z = tf.random.normal([x.shape[0], z_dim])
                disc_loss = train_disc(x, z, cc, nc, gen, disc, disc_opt, p_aug=p_aug, norm_scale=norm_scale)
            disc_losses(disc_loss)

            # Train generator
            z = tf.random.normal([x.shape[0], z_dim])
            gen_loss = train_gen(z, cc, nc, gen, disc, gen_opt, p_aug=p_aug, norm_scale=norm_scale)
            gen_losses(gen_loss)

        # Logs
        with disc_summary_writer.as_default():
            tf.summary.scalar('loss', disc_losses.result(), step=epoch)
        with gen_summary_writer.as_default():
            tf.summary.scalar('loss', gen_losses.result(), step=epoch)

        # Save the model
        if epoch % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

            score = score_fn(gen)
            if score > best_score:
                print('Saving model ...')
                save_fn()
                best_score = score
                patience = initial_patience
            else:
                patience -= 1

            if verbose:
                print('Score: {:.3f}'.format(score))

        if verbose:
            print('Epoch {}. Gen loss: {:.2f}. Disc loss: {:.2f}'.format(epoch + 1,
                                                                         gen_losses.result(),
                                                                         disc_losses.result()))
        gen_losses.reset_states()
        disc_losses.reset_states()

        if patience == 0:
            break


def predict(cc, nc, gen, z=None, training=False):
    """
    Make predictions
    :param cc: Categorical covariates
    :param nc: Numerical covariates
    :param gen: Generator model
    :param z: Latent input
    :param training: Whether training
    :return: Sampled data
    """
    nb_samples = cc.shape[0]
    if z is None:
        z_dim = gen.input[0].shape[-1]
        z = tf.random.normal([nb_samples, z_dim])
    out = gen([z, cc, nc], training=training)
    if not training:
        return out.numpy()
    return out
