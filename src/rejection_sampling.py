from biogan import BioGAN
from data_pipeline import load_data, save_synthetic, split_train_test
import numpy as np
import scipy

DEFAULT_GRID_SIZE = 1000
DEFAULT_BATCH_SIZE = 10000


def rejection_sampling(generate_data, nb_samples, gene_idx, expr_train, grid_size=DEFAULT_GRID_SIZE,
                       batch_size=DEFAULT_BATCH_SIZE):
    """
    Performs rejection sampling. Generates nb_samples samples by attempting to match the synthetic distribution
    of gene idx
    :param generate_data: Function that takes takes an integer n and samples from a synthetic distribution.
                          Returns an array of n samples.
    :param nb_samples: Number of samples to generate
    :param gene_idx: Gene index for which its synthetic distribution will be "matched" to the real distribution.
    :param expr_train: Samples from the real distribution. Shape=(nb_training_samples, nb_genes)
    :param grid_size: Size of the grid used to find the rejection sampling normalizing constant
    :param batch_size: Batch size used to call generate_data during rejection sampling
    :return: Rejection samples for gene_idx. Shape=(nb_samples, nb_genes)
    """
    # Kernel Density Estimation and finding rejection sampling constant
    r_expr_gene = expr_train[:, gene_idx].ravel()
    r_kde_gene = scipy.stats.gaussian_kde(r_expr_gene)
    grid = np.linspace(np.min(r_expr_gene), np.max(r_expr_gene), grid_size)
    s_expr = generate_data(500*expr_train.shape[0])
    s_expr_gene = s_expr[:, gene_idx].ravel()
    s_kde_gene = scipy.stats.gaussian_kde(s_expr_gene)
    norm_const = max(r_kde_gene(grid) / (s_kde_gene(grid) + 1e-5)) + 1

    # WARNING: Rejection sampling might be very inefficient for a large norm_const
    print('Rejection sampling normalization constant: {}'.format(norm_const))

    # Rejection sampling. Generates nb_samples expression arrays
    print('Performing rejection sampling...')
    accepted_samples = np.empty(shape=(nb_samples, expr_train.shape[1]))
    nb_accepted = 0
    while nb_accepted < nb_samples:
        s_batch = generate_data(batch_size)
        s_expr_gene = s_batch[:, gene_idx].ravel()
        r_probs = r_kde_gene.pdf(s_expr_gene)
        s_probs = s_kde_gene.pdf(s_expr_gene)
        probs = r_probs / (norm_const * s_probs)
        rand = np.random.uniform(0, 1, (batch_size,))
        accepted_mask = (rand < probs)
        to_add = s_batch[accepted_mask, :]
        nb_add = min(nb_samples - nb_accepted, to_add.shape[0])
        to_add = to_add[:nb_add, :]
        accepted_samples[nb_accepted:(nb_accepted + nb_add), :] = to_add
        nb_accepted += nb_add
        print('{}/{}'.format(nb_accepted, nb_samples))

    return np.array(accepted_samples)


if __name__ == '__main__':
    # Load data
    root_gene = 'CRP'
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

    mean = np.mean(expr, axis=0)
    std = np.std(expr, axis=0)
    std_clip = 2

    # Standardize data
    r_min = expr.min(axis=0)
    r_max = expr.max(axis=0)
    expr_train = (expr_train - mean) / (std_clip * std)
    restore_norm = lambda expr: std_clip * expr * std + mean

    # Load GAN
    biogan = BioGAN(expr_train)
    biogan.load_model(file_name)

    # Generate data
    generate_data = lambda size: restore_norm(biogan.generate_batch(size))
    s_expr = generate_data(expr_train.shape[0])

    # Perform rejection sampling
    crp_idx = np.argwhere(np.array(gene_symbols) == 'crp')
    r_expr_crp = expr_train[:, crp_idx].ravel()
    s_expr_crp = s_expr[:, crp_idx].ravel()
    s_expr_rej = rejection_sampling(generate_data=generate_data,
                                    nb_samples=expr_train.shape[0],
                                    gene_idx=crp_idx,
                                    expr_train=expr[train_idxs, :]  # Unnormalized train expression data
                                    )

    # Save data
    rej_file_name = '{}_{}'.format('rej', file_name)
    save_synthetic(rej_file_name, s_expr_rej, gene_symbols)
