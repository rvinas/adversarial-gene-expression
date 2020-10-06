from ggan import gGAN, generate_data, normalize
from data_pipeline import load_data, save_synthetic, split_train_test, reg_network, reverse_edges
import numpy as np
import scipy

DEFAULT_GRID_SIZE = 500
DEFAULT_BATCH_SIZE = 1024
DEFAULT_REJ_DIR = 'rej/'


def sampling(generate_data, nb_samples):
    """
    Generates nb_samples from generate_data
    :param generate_data: Function that takes takes an integer n and samples from a synthetic distribution.
                          Returns an array of n samples.
    :param nb_samples: Number of samples to generate
    :return: Samples. Shape=(nb_samples, nb_genes)
    """
    return generate_data(nb_samples)


def rejection_sampling(generate_data, nb_samples, gene_idxs, expr_train, grid_size=DEFAULT_GRID_SIZE,
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
    # Kernel Density Estimation and finding rejection sampling normalization constant
    r_expr_genes = expr_train[:, gene_idxs]
    r_kde_genes = scipy.stats.gaussian_kde(r_expr_genes.T)
    grids = []
    gene_mins = np.min(r_expr_genes, axis=0)
    gene_maxs = np.max(r_expr_genes, axis=0)
    for g_min, g_max in zip(gene_mins, gene_maxs):
        grid = np.linspace(g_min, g_max, grid_size)
        grids.append(grid)
    grids = np.array(grids)

    s_expr = generate_data(1000 * expr_train.shape[0])
    s_expr_genes = s_expr[:, gene_idxs]
    s_kde_genes = scipy.stats.gaussian_kde(s_expr_genes.T)
    norm_const = max(r_kde_genes(grids) / (s_kde_genes(grids) + 1e-5)) + 1

    # WARNING: Rejection sampling might be very inefficient for a large norm_const
    print('Rejection sampling normalization constant: {}'.format(norm_const))

    # Rejection sampling. Generates nb_samples expression arrays
    accepted_samples = np.empty(shape=(nb_samples, expr_train.shape[1]))
    nb_accepted = 0
    while nb_accepted < nb_samples:
        s_batch = generate_data(batch_size)
        s_expr_genes = s_batch[:, gene_idxs]
        r_probs = r_kde_genes.pdf(s_expr_genes.T)
        s_probs = s_kde_genes.pdf(s_expr_genes.T)
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
    root_gene = None  # 'CRP'
    minimum_evidence = 'weak'
    max_depth = np.inf
    expr, gene_symbols, sample_names = load_data(root_gene=root_gene,
                                                 minimum_evidence=minimum_evidence,
                                                 max_depth=max_depth)
    file_name = 'EColi_n{}_r{}_e{}_d{}'.format(len(gene_symbols), root_gene, minimum_evidence, max_depth)
    print('File: {}'.format(file_name))

    # Split data into train and test sets
    train_idxs, test_idxs = split_train_test(sample_names)
    expr_train = expr[train_idxs, :]
    expr_test = expr[test_idxs, :]

    # Load GAN
    ggan = gGAN(normalize(expr_train), gene_symbols)
    ggan.load_model(file_name)

    # Generate synthetic data
    mean = np.mean(expr_train, axis=0)
    std = np.std(expr_train, axis=0)
    r_min = expr_train.min()
    r_max = expr_train.max()
    gen_data = lambda size: generate_data(ggan, size, mean, std, r_min, r_max)

    # Perform normal sampling
    print('Performing normal sampling...')
    s_expr = sampling(generate_data=gen_data,
                      nb_samples=expr_train.shape[0])

    # Save data
    print('Saving generated data...')
    save_synthetic(file_name, s_expr, gene_symbols)

    # Get regulatory network
    nodes, edges = reg_network(gene_symbols,
                               root_gene=root_gene,
                               break_loops=False)
    r_edges = reverse_edges(edges)

    # Perform rejection sampling
    gene_list = ['xylr', 'yeil', 'soxs', 'fur',
                 'cadc', 'gutm', 'hyfr', 'acs',
                 'acna', 'aer', 'caia', 'nagc',
                 'gyra', 'lpd', 'mdh', 'dpia']
    for i, gene_name in enumerate(gene_list):
        print('{}/{}...'.format(i, len(gene_list)))
        tfs = r_edges[gene_name].keys()
        tf_idxs = np.squeeze([np.argwhere(np.array(gene_symbols) == g) for g in tfs if g != gene_name])
        print('Performing rejection sampling for genes {} (TFs of `{}`)...'.format([gene_symbols[g] for g in tf_idxs], gene_name))

        s_expr_rej = rejection_sampling(generate_data=gen_data,
                                        nb_samples=expr_train.shape[0],
                                        gene_idxs=tf_idxs,
                                        expr_train=expr_train)

        # Save data
        print('Saving rejection sampling data...')
        rej_file_name = '{}/{}_{}'.format(DEFAULT_REJ_DIR, gene_name, file_name)
        save_synthetic(rej_file_name, s_expr_rej, gene_symbols)