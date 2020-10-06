from utils import *
from tf_utils import *

EPOCHS = 500
BATCH_SIZE = 32
LATENT_DIM = 10
MODELS_DIR = 'checkpoints/models/'

if __name__ == '__main__':
    # GPU limit
    limit_gpu()

    # Constants
    root_gene = 'CRP'  # Set to 'CRP' to select the CRP hierarchy. Set to None to use full set of genes
    minimum_evidence = 'weak'
    max_depth = np.inf
    retrain = False
    interactions = None

    # Load data
    expr, gene_symbols, sample_names = load_ecoli(root_gene=root_gene,
                                                  minimum_evidence=minimum_evidence,
                                                  max_depth=max_depth)

    # Load conditions
    encoded_conditions = None
    vocab_dicts = None

    _, encoded_conditions, vocab_dicts = ecoli_discrete_conditions(sample_names)
    print('Conditions: ', encoded_conditions)
    encoded_conditions = np.int32(encoded_conditions)

    file_name = 'EColi_n{}_r{}_e{}_d{}'.format(len(gene_symbols), root_gene, minimum_evidence, max_depth)
    print('Filename: ', file_name)

    # Split data into train and test sets
    train_idxs, test_idxs = split_train_test_v3(sample_names)
    expr = np.float32(expr)
    expr_train = expr[train_idxs, :]
    expr_test = expr[test_idxs, :]
    cond_train = encoded_conditions[train_idxs, :]
    cond_test = encoded_conditions[test_idxs, :]

    # Normalise data
    x_mean = np.mean(expr_train, axis=0)
    x_std = np.std(expr_train, axis=0)
    expr_train = standardize(expr_train, mean=x_mean, std=x_std)
    expr_test = standardize(expr_test, mean=x_mean, std=x_std)

    # Define model
    vocab_sizes = [len(c) for c in vocab_dicts]
    print('Vocab sizes: ', vocab_sizes)
    num_covs_train = np.zeros(shape=[cond_train.shape[0], 1])  # Dummy
    num_covs_test = np.zeros(shape=[cond_test.shape[0], 1])  # Dummy
    nb_numeric = num_covs_train.shape[-1]
    x_dim = expr_train.shape[-1]
    gen = make_generator(x_dim, vocab_sizes, nb_numeric, z_dim=LATENT_DIM, h_dims=[128, 128])
    disc = make_discriminator(x_dim, vocab_sizes, nb_numeric, h_dims=[128, 128])


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
        gen.save(models_dir + 'gen_ecoli.h5')


    # Train model
    train(dataset=expr_train,
          cat_covs=cond_train,
          num_covs=num_covs_train,
          z_dim=LATENT_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          gen=gen,
          disc=disc,
          p_aug=0.25,
          score_fn=score_fn(expr_test, cond_test, num_covs_test),
          save_fn=save_fn)

    # Evaluate data
    score = score_fn(expr_test, cond_test, num_covs_test)(gen)
    print('Gamma(Dx, Dz): {:.2f}'.format(score))
