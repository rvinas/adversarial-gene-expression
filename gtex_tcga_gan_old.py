from utils import *
from tf_utils import *
from collections import Counter



EPOCHS = 1000
BATCH_SIZE = 32
LATENT_DIM = 64
MODELS_DIR = 'checkpoints/models/'



if __name__ == '__main__':
    # Load dataset
    expr_df, info_df = rnaseqdb_load()
    x = expr_df.values.T
    symbols = expr_df.index.levels[0].values
    sampl_ids = expr_df.columns.values
    tissues = info_df['TISSUE_GTEX'].values
    datasets = info_df['DATASET'].values

    print(Counter(datasets))
    print(Counter(tissues))
    print('Nb. genes: ', x.shape[-1])

    # Process categorical metadata
    cat_dicts = []
    tissues_dict_inv = np.array(list(sorted(set(tissues))))
    tissues_dict = {t: i for i, t in enumerate(tissues_dict_inv)}
    tissues = np.vectorize(lambda t: tissues_dict[t])(tissues)
    cat_dicts.append(tissues_dict_inv)
    dataset_dict_inv = np.array(list(sorted(set(datasets))))
    dataset_dict = {d: i for i, d in enumerate(dataset_dict_inv)}
    datasets = np.vectorize(lambda t: dataset_dict[t])(datasets)
    cat_dicts.append(dataset_dict_inv)
    cat_covs = np.concatenate((tissues[:, None], datasets[:, None]), axis=-1)

    cat_covs = np.int32(cat_covs)
    print('Cat covs: ', cat_covs.shape)

    # Process numerical metadata. NOTE: temporarily ignoring numerical covs
    # num_cols = ['AGE']  # 'AGE'
    # num_covs = df_metadata.loc[sampl_ids, num_cols].values
    # num_covs = standardize(num_covs)
    # num_covs = np.float32(num_covs)
    num_covs = np.zeros((x.shape[0], 1), dtype=np.float32)
    # num_covs = np.zeros_like(cat_covs).astype(np.float32)

    # Log-transform data
    x = np.log(1 + x)
    x = np.float32(x)

    # Train/test split
    np.random.seed(0)
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x = x[idx, :]
    num_covs = num_covs[idx, :]
    cat_covs = cat_covs[idx, :]

    x_train, x_test = split_train_test(x)
    num_covs_train, num_covs_test = split_train_test(num_covs)
    cat_covs_train, cat_covs_test = split_train_test(cat_covs)

    # Normalise data
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_train = standardize(x_train, mean=x_mean, std=x_std)
    x_test = standardize(x_test, mean=x_mean, std=x_std)

    # GPU limit
    limit_gpu(2)

    # Define model
    vocab_sizes = [len(c) for c in cat_dicts]
    print('Vocab sizes: ', vocab_sizes)
    nb_numeric = num_covs.shape[-1]
    x_dim = x.shape[-1]
    gen = make_generator(x_dim, vocab_sizes, nb_numeric)
    disc = make_discriminator(x_dim, vocab_sizes, nb_numeric)

    # Evaluation metrics
    def score_fn(x_test, cat_covs_test, num_covs_test):
        def _score(gen):
            x_gen = predict(cc=cat_covs_test,
                            nc=num_covs_test,
                            gen=gen)

            gamma_dx_dz = gamma_coef(x_test, x_gen)
            return gamma_dx_dz
            # score = (x_test - x_gen) ** 2
            # return -np.mean(score)

        return _score

    # Function to save models
    def save_fn(models_dir=MODELS_DIR):
        gen.save(models_dir + 'gen_rnaseqdb.h5')


    # Train model
    gen_opt = tfk.optimizers.RMSprop(5e-4)
    disc_opt = tfk.optimizers.RMSprop(5e-4)

    train(dataset=x_train,
          cat_covs=cat_covs_train,
          num_covs=num_covs_train,
          z_dim=LATENT_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          gen=gen,
          disc=disc,
          gen_opt=gen_opt,
          disc_opt=disc_opt,
          score_fn=score_fn(x_test, cat_covs_test, num_covs_test),
          save_fn=save_fn)

    # Evaluate data
    score = score_fn(x_test, cat_covs_test, num_covs_test)(gen)
    print('Gamma(Dx, Dz): {:.2f}'.format(score))
