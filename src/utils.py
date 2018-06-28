import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_pipeline import tf_tg_interactions
import scipy.stats
from statsmodels.stats.multitest import multipletests


def pearson_correlation(x, y):
    """
    Computes similarity measure between each pair of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :return: Matrix with shape (nb_genes_1, nb_genes_2) containing the similarity coefficients
    """

    def standardize(a):
        a_off = np.mean(a, axis=0)
        a_std = np.std(a, axis=0)
        return (a - a_off) / a_std

    assert x.shape[0] == y.shape[0]
    x_ = standardize(x)
    y_ = standardize(y)
    return np.dot(x_.T, y_) / x.shape[0]


def upper_diag_list(m_):
    """
    Returns the condensed list of all the values in the upper-diagonal of m_
    :param m_: numpy array of float. Shape=(N, N)
    :return: list of values in the upper-diagonal of m_ (from top to bottom and from
             left to right). Shape=(N*(N-1)/2,)
    """
    m = np.triu(m_, k=1)  # upper-diagonal matrix
    tril = np.zeros_like(m_) + np.nan
    tril = np.tril(tril)
    m += tril
    m = np.ravel(m)
    return m[~np.isnan(m)]


def correlations_list(x, y, corr_fun=pearson_correlation):
    """
    Generates correlation list between all pairs of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :param corr_fun: correlation function taking x and y as inputs
    """
    corr = corr_fun(x, y)
    return upper_diag_list(corr)


def compute_tf_tg_corrs(expr, gene_symbols, tf_tg=None):
    """
    Computes the lists of TF-TG and TG-TG correlations
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param gene_symbols: list of gene symbols matching the expr matrix. Shape=(nb_genes,)
    :param tf_tg: dict with TF symbol as key and list of TGs' symbols as value
    :return: flat np.array of TF-TG correlations and flat np.array of TG-TG correlations
    """
    if tf_tg is None:
        tf_tg = tf_tg_interactions()
    gene_symbols = np.array(gene_symbols)

    tf_tg_corr = []
    tg_tg_corr = []
    for tf, tgs in tf_tg.items():
        if tf in gene_symbols:
            # TG-TG correlations
            present_tgs = [tg for tg in tgs if tg in gene_symbols]
            tg_idxs = np.searchsorted(gene_symbols, present_tgs)
            expr_tgs = expr[:, tg_idxs]
            corr = correlations_list(expr_tgs, expr_tgs)
            tg_tg_corr += corr.tolist()

            # TF-TG correlations
            tf_idx = np.argwhere(gene_symbols == tf)[0]
            expr_tf = expr[:, tf_idx]
            corr = correlations_list(expr_tf[:, None], expr_tgs)
            tf_tg_corr += corr.tolist()

    return np.array(tf_tg_corr), np.array(tg_tg_corr)


# PLOTTING UTILITIES
def plot_intensities(expr, plot_quantiles=True):
    """
    Plot intensities histogram
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param plot_quantiles: whether to plot the 5 and 95% intensity gene quantiles
    """
    x = np.ravel(expr)
    ax = sns.distplot(x,
                      hist=False,
                      kde_kws={'color': 'royalblue', 'linewidth': 2, 'bw': .15},
                      label='E. coli M3D')

    if plot_quantiles:
        stds = np.std(expr, axis=-1)
        idxs = np.argsort(stds)
        cut_point = int(0.05 * len(idxs))

        q95_idxs = idxs[-cut_point]
        x = np.ravel(expr[q95_idxs, :])
        ax = sns.distplot(x,
                          ax=ax,
                          hist=False,
                          kde_kws={'linestyle': ':', 'color': 'royalblue', 'linewidth': 2, 'bw': .15},
                          label='High variance E. coli M3D')

        q5_idxs = idxs[:cut_point]
        x = np.ravel(expr[q5_idxs, :])
        sns.distplot(x,
                     ax=ax,
                     hist=False,
                     kde_kws={'linestyle': '--', 'color': 'royalblue', 'linewidth': 2, 'bw': .15},
                     label='Low variance E. coli M3D')
        plt.legend()
        plt.xlabel('Absolute levels')
        plt.ylabel('Density')


def plot_gene_ranges(expr):
    """
    Plot gene ranges histogram
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    """
    nb_samples, nb_genes = expr.shape
    sorted_expr = [np.sort(expr[:, gene]) for gene in range(nb_genes)]
    sorted_expr = np.array(sorted_expr)  # Shape=(nb_genes, nb_samples)
    cut_point = int(0.05 * nb_samples)
    diffs = sorted_expr[:, -cut_point] - sorted_expr[:, cut_point]

    ax = sns.distplot(diffs,
                      hist=False,
                      kde_kws={'color': 'royalblue', 'linewidth': 2, 'bw': .1},
                      label='E. coli M3D')

    plt.xlabel('Range of gene lavels')
    plt.ylabel('Density')


def plot_difference_histogram(interest_distr, background_distr, xlabel, left_lim=-1, right_lim=1):
    """
    Plots a difference between a distribution of interest and a background distribution.
    Approximates these distributions with Kernel Density Estimation using a Gaussian kernel
    :param interest_distr: list containing the values of the distribution of interest.
    :param background_distr: list containing the values of the background distribution.
    :param right_lim: histogram left limit
    :param left_lim: histogram right limit
    """
    # Estimate distributions
    kde_back = scipy.stats.gaussian_kde(background_distr)
    kde_corr = scipy.stats.gaussian_kde(interest_distr)

    # Plot difference histogram
    grid = np.linspace(left_lim, right_lim, 1000)
    # plt.plot(grid, kde_back(grid), label="kde A")
    # plt.plot(grid, kde_corr(grid), label="kde B")
    plt.plot(grid, kde_corr(grid) - kde_back(grid),
             'royalblue',
             label='E. coli M3D')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Density difference')


def plot_tf_activity_histogram(expr, gene_symbols, tf_tg=None):
    """
    Plots the TF activity histogram. It is computed according to the Wilcoxon's non parametric rank-sum method, which tests
    whether TF targets exhibit significant rank differences in comparison with other non-target genes. The obtained
    p-values are corrected via Benjamini-Hochberg's procedure to account for multiple testing.
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param gene_symbols: list of gene_symbols. Shape=(nb_genes,)
    :param tf_tg: dict with TF symbol as key and list of TGs' symbols as value
    """
    nb_samples, nb_genes = expr.shape

    if tf_tg is None:
        tf_tg = tf_tg_interactions()
    gene_symbols = np.array(gene_symbols)

    # Normalize expression data
    expr_norm = (expr - np.mean(expr, axis=0)) / np.std(expr, axis=0)

    # For each TF, check whether its target genes exhibit significant rank differences in comparison with other
    # non-target genes.
    active_tfs = []
    for tf, tgs in tf_tg.items():
        if tf in gene_symbols:
            # Find expressions of TG regulated by TF
            present_tgs = [tg for tg in tgs if tg in gene_symbols]
            tg_idxs = np.searchsorted(gene_symbols, present_tgs)
            expr_tgs = expr_norm[:, tg_idxs]

            # Find expressions of other genes
            non_tg_idxs = list(set(range(nb_genes)) - set(tg_idxs))
            expr_non_tgs = expr_norm[:, non_tg_idxs]

            # Compute Wilcoxon's p-value for each sample
            p_values = []
            for i in range(nb_samples):
                statistic, p_value = scipy.stats.mannwhitneyu(expr_tgs[i, :], expr_non_tgs[i, :],
                                                              alternative='two-sided')
                p_values.append(p_value)

            # Correct the independent p-values to account for multiple testing with Benjamini-Hochberg's procedure
            reject, p_values_c, _, _ = multipletests(pvals=p_values,
                                                     alpha=0.05,
                                                     method='fdr_bh')
            chip_rate = np.sum(reject) / nb_samples
            active_tfs.append(chip_rate)

    # Plot histogram
    values = active_tfs
    bins = np.logspace(-10, 1, 20, base=2)
    bins[0] = 0
    fig, ax = plt.subplots()
    plt.hist(values, bins=bins)
    ax.set_xscale('log', basex=2)
    ax.set_xlim(2 ** -10, 1)
    ax.set_xlabel('Fraction of chips. TF activity')
    ax.set_ylabel('Density')
    # from matplotlib.ticker import FormatStrFormatter
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.show()
