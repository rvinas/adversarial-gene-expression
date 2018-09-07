from biogan import BioGAN, generate_data, normalize
from data_pipeline import load_data, load_synthetic, split_train_test, reg_network, reverse_edges
import numpy as np
from syntren import *
from gnw import *


def binary_crossentropy(labels, pred, eps=1e-7):
    """
    Computes binary crossentropy
    :param labels: list of labels. Shape=(nb_samples,)
    :param pred: discriminator's predictions. Shape=(nb_samples,)
    :param eps: epsilon
    :return: binary crossentropy
    """
    return -np.mean(labels * np.log(pred + eps) + (1 - labels) * np.log(1 - pred + eps))


if __name__ == '__main__':
    root_gene = 'CRP'
    minimum_evidence = 'weak'
    max_depth = np.inf
    r_expr, gene_symbols, sample_names = load_data(root_gene=root_gene,
                                                   minimum_evidence=minimum_evidence,
                                                   max_depth=max_depth)

    # Split data into train and test sets
    train_idxs, test_idxs = split_train_test(sample_names)
    expr_train = r_expr[train_idxs, :]
    expr_test = r_expr[test_idxs, :]
    nb_train, nb_genes = expr_train.shape

    # Find minimum and maximum expression values
    r_min = np.min(expr_train)
    r_max = np.max(expr_train)

    # Find mean and std of each gene
    r_mean = np.mean(expr_train, axis=0)
    r_std = np.std(expr_train, axis=0)

    ### GAN data ###
    file_name = 'EColi_n{}_r{}_e{}_d{}'.format(len(gene_symbols), root_gene, minimum_evidence, max_depth)
    s_expr, s_gs = load_synthetic(file_name)
    assert (np.array(gene_symbols) == np.array(s_gs)).all()

    gan_expr = normalize(s_expr, kappa=2)

    ### SynTReN data ###
    s_expr, s_gene_symbols = syntren_results(minimum_evidence='Weak',
                                             nb_background=0)

    # Align synthetic gene symbols
    idxs = [s_gene_symbols.index(gene) for gene in gene_symbols]
    s_expr = s_expr[:nb_train, idxs]
    assert (np.array(s_gene_symbols)[idxs] == np.array(gene_symbols)).all()

    syn_expr = normalize(s_expr, kappa=2)

    ### GNW data ###
    s_expr, s_gene_symbols = gnw_results(minimum_evidence='Weak',
                                         break_loops=False)

    # Align synthetic gene symbols
    idxs = [s_gene_symbols.index(gene) for gene in gene_symbols]
    s_expr = s_expr[:nb_train, idxs]
    assert (np.array(s_gene_symbols)[idxs] == np.array(gene_symbols)).all()

    gnw_expr = normalize(s_expr, kappa=2)

    ### Test data ###
    test_expr = normalize(expr_test, kappa=2)

    # Load GAN
    biogan = BioGAN(normalize(expr_train, kappa=2))
    biogan.load_model(file_name)

    # Discriminate datasets
    syn_out = biogan.discriminate(syn_expr)
    gnw_out = biogan.discriminate(gnw_expr)
    gan_out = biogan.discriminate(gan_expr)
    test_out = biogan.discriminate(test_expr)

    syn_ce = binary_crossentropy(np.zeros_like(syn_out), syn_out)
    gnw_ce = binary_crossentropy(np.zeros_like(gnw_out), gnw_out)
    gan_ce = binary_crossentropy(np.zeros_like(gan_out), gan_out)
    test_ce = binary_crossentropy(np.ones_like(test_out), test_out)

    syn_acc = np.mean(np.zeros_like(syn_out) == np.round(syn_out))
    gnw_acc = np.mean(np.zeros_like(gnw_out) == np.round(gnw_out))
    gan_acc = np.mean(np.zeros_like(gan_out) == np.round(gan_out))
    test_acc = np.mean(np.ones_like(test_out) == np.round(test_out))

    print('Cross entropies:\nSynTReN: {}\nGNW: {}\nGAN: {}\nTest: {}'
          .format(syn_ce, gnw_ce, gan_ce, test_ce))

    print('\nAccuracies:\nSynTReN: {}\nGNW: {}\nGAN: {}\nTest: {}'
          .format(syn_acc, gnw_acc, gan_acc, test_acc))
