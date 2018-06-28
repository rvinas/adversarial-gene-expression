import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
from sklearn.metrics import silhouette_score
from utils import *
from data_pipeline import load_data, split_replicates
import matplotlib.pyplot as plt
import seaborn as sns


class Cluster:
    """
    Auxiliary class to store binary clusters
    """
    def __init__(self, c_left=None, c_right=None, index=None):
        assert (index is not None) ^ (c_left is not None and c_right is not None)
        self._c_left = c_left
        self._c_right = c_right
        if index is not None:
            self._indices = [index]
        else:
            self._indices = c_left.indices + c_right.indices

    @property
    def indices(self):
        return self._indices

    @property
    def c_left(self):
        return self._c_left

    @property
    def c_right(self):
        return self._c_right


def hierarchical_clustering(data, corr_fun=pearson_correlation):
    """
    Performs hierarchical clustering to cluster genes according to a gene similarity
    metric.
    Reference: Cluster analysis and display of genome-wide expression patterns
    :param data: numpy array. Shape=(nb_samples, nb_genes)
    :param corr_fun: function that computes the pairwise correlations between each pair
                     of genes in data
    :return scipy linkage matrix
    """
    # Perform hierarchical clustering
    y = 1 - correlations_list(data, data, corr_fun)
    l_matrix = linkage(y, 'complete')  # 'correlation'
    return l_matrix


def compute_silhouette(data, l_matrix):
    """
    Computes silhouette scores of the dendrogram given by l_matrix
    :param data: numpy array. Shape=(nb_samples, nb_genes)
    :param l_matrix: Scipy linkage matrix. Shape=(nb_genes-1, 4)
    :return: list of Silhouette scores
    """
    nb_samples, nb_genes = data.shape

    # Form dendrogram and compute Silhouette score at each node
    clusters = {i: Cluster(index=i) for i in range(nb_genes)}
    scores = []
    for i, z in enumerate(l_matrix):
        c1, c2, dist, n_elems = z
        clusters[nb_genes + i] = Cluster(c_left=clusters[c1],
                                         c_right=clusters[c2])
        c1_indices = clusters[c1].indices
        c2_indices = clusters[c2].indices
        labels = [0] * len(c1_indices) + [1] * len(c2_indices)
        if len(labels) == 2:
            scores.append(0)
        else:
            expr = data[:, clusters[nb_genes + i].indices]
            m = 1 - pearson_correlation(expr, expr)
            score = silhouette_score(m, labels, metric='precomputed')
            scores.append(score)

    return scores


def dendrogram_distance(l_matrix, condensed=True):
    """
    Computes the distances between each pair of genes according to the scipy linkage
    matrix.
    :param l_matrix: Scipy linkage matrix. Shape=(nb_genes-1, 4)
    :param condensed: whether to return the distances as a flat array containing the
           upper-triangular of the distance matrix
    :return: distances
    """
    nb_genes = l_matrix.shape[0] + 1

    # Fill distance matrix m
    clusters = {i: Cluster(index=i) for i in range(nb_genes)}
    m = np.zeros((nb_genes, nb_genes))
    for i, z in enumerate(l_matrix):
        c1, c2, dist, n_elems = z
        clusters[nb_genes + i] = Cluster(c_left=clusters[c1],
                                         c_right=clusters[c2])
        c1_indices = clusters[c1].indices
        c2_indices = clusters[c2].indices

        for c1_idx in c1_indices:
            for c2_idx in c2_indices:
                m[c1_idx, c2_idx] = dist
                m[c2_idx, c1_idx] = dist

    # Return flat array if condensed
    if condensed:
        return upper_diag_list(m)

    return m


def compare_cophenetic(l_matrix1, l_matrix2):
    """
    Computes the cophenic distance between two dendrograms given as scipy linkage matrices
    :param l_matrix1: Scipy linkage matrix. Shape=(nb_genes-1, 4)
    :param l_matrix2: Scipy linkage matrix. Shape=(nb_genes-1, 4)
    :return: cophenic distance between two dendrograms
    """
    dists1 = dendrogram_distance(l_matrix1, condensed=True)
    dists2 = dendrogram_distance(l_matrix2, condensed=True)

    return pearson_correlation(dists1, dists2)


if __name__ == '__main__':
    expr, gene_symbols, sample_names = load_data()

    # Perform hierarchical clustering on full expression matrix
    l_matrix = hierarchical_clustering(expr)
    silhouettes = compute_silhouette(expr, l_matrix)
    y = 1 - correlations_list(expr, expr)
    c, d = cophenet(l_matrix, y)
    print('Cophenic coefficient full dendrogram wrt. the original distance matrix: {}'.format(c))

    # Perform hierarchical clustering on 2 mutually exclusive subset of samples
    idxs1, idxs2 = split_replicates(sample_names)
    l_matrix1 = hierarchical_clustering(expr[idxs1, :])
    l_matrix2 = hierarchical_clustering(expr[idxs2, :])
    c = compare_cophenetic(l_matrix1, l_matrix2)
    print('Cophenetic coefficient dendrogram 1 wrt. dendrogram 2: {}'.format(c))
    c, d = cophenet(l_matrix1, y)
    print('Cophenetic coefficient dendrogram 1 wrt. the original distance matrix: {}'.format(c))
    c, d = cophenet(l_matrix2, y)
    print('Cophenetic coefficient dendrogram 2 wrt. the original distance matrix: {}'.format(c))
    dn1 = dendrogram(l_matrix1, truncate_mode='level', p=4)
    plt.show()
    dn2 = dendrogram(l_matrix2, truncate_mode='level', p=4)
    plt.show()

    # Plot Silhouette histogram for full dendrogram
    sns.distplot(silhouettes)
    plt.show()

    # Plot Silhouette histogram for random dendrogram
    random_noise = np.random.uniform(4, 14, expr.shape)
    l_matrix_rand = hierarchical_clustering(random_noise)
    silhouettes_rand = compute_silhouette(random_noise, l_matrix_rand)
    sns.distplot(silhouettes_rand)
    plt.show()