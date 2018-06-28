import numpy as np
import collections
from sklearn.metrics import silhouette_score
from src.utils import pearson_correlation
import sys

sys.setrecursionlimit(5000)


class Cluster:
    def __init__(self, child_clusters=None, indices=None, expressions=None):
        """

        Creates a cluster for hierarchical clustering. Provide either a list of sub_clusters or
        a list of element indices + element expressions
        :param child_clusters: None or list of 2 clusters
        :param indices: None or list of element indices
        :param expressions: None or list of element expressions
        """
        assert child_clusters is not None or (indices is not None and expressions is not None)
        assert child_clusters is None or len(child_clusters) == 2
        self._child_clusters = child_clusters
        self._indices = indices
        self._expressions = expressions
        self._weight = None
        self._profile = None
        self._silhouette = None

    @property
    def weight(self):
        if self._weight is None:
            self._weight = len(self.indices)
        return self._weight

    @property
    def indices(self):
        if self._indices is None:
            self._indices = [idx for c in self._child_clusters for idx in c.indices]
        return np.array(self._indices)

    @property
    def n_elements(self):
        return len(self.indices)

    @property
    def expressions(self):
        if self._expressions is None:
            self._expressions = [expr for c in self._child_clusters for expr in c.expressions]
        return np.array(self._expressions)

    @property
    def profile(self):
        if self._profile is None:
            if self._expressions is not None:
                self._expressions = np.array(self._expressions)
                self._profile = np.mean(self._expressions, axis=0)
            else:
                child_weights = np.array([c.weight for c in self._child_clusters])
                total_weight = np.sum(child_weights, dtype='float')
                child_weights = child_weights / total_weight
                child_profiles = np.array([c.profile for c in self._child_clusters])
                self._profile = np.sum(child_weights[:, None] * child_profiles, axis=0)
        return self._profile

    @property
    def silhouette(self):
        if self._silhouette is None:
            c_left, c_right = self._child_clusters
            labels = [0] * c_left.n_elements + [1] * c_right.n_elements
            if len(labels) == 2:
                self._silhouette = 0
            else:
                m = 1-pearson_correlation(self.expressions.T, self.expressions.T)
                self._silhouette = silhouette_score(m, labels, metric='precomputed')
        return self._silhouette


def hierarchical_clustering(data):
    """
    Performs hierarchical clustering to cluster genes according to a gene similarity
    metric.
    Reference: Cluster analysis and display of genome-wide expression patterns
    :param data: numpy array with shape (nb_samples, nb_genes)
    """
    nb_samples, nb_genes = data.shape

    # Initialize clusters
    data_ = np.array(data)
    clusters = [Cluster(indices=[i], expressions=[data_[:, i]]) for i in range(nb_genes)]

    # Silhouette coefficients
    sil_coeffs = []

    print('Clustering genes ...')
    for k in range(nb_genes - 1):
        if k % 50 == 0:
            print(k)

        # Compute similarities between clusters
        s = pearson_correlation(data_, data_)

        # Mask lower-diagonal similarity matrix with -np.inf
        s = np.triu(s, k=1)  # Upper-diagonal similarity matrix
        tril = np.zeros_like(s) - np.inf
        tril = np.tril(tril)
        s += tril

        # Find 2 most correlated clusters
        idx = np.argmax(s)
        i, j = np.unravel_index(idx, dims=s.shape)
        a, b = min(i, j), max(i, j)

        print('Pearson corr: {}'.format(s[i, j]))

        # Form cluster
        cluster = Cluster(child_clusters=[clusters[a], clusters[b]])
        clusters[a] = cluster
        del clusters[b]

        # Compute silhouette coefficients
        sil_coeffs.append(cluster.silhouette)

        # Update expressions
        data_[:, a] = cluster.profile
        keep_cols = [x for x in range(nb_genes - k) if x != b]
        data_ = data_[:, keep_cols]

    return clusters[0], sil_coeffs


from src.data_pipeline import load_data
import seaborn as sns
import matplotlib.pyplot as plt

expr, _, _ = load_data()

c, sil_coeffs = hierarchical_clustering(expr[:, :1000])
print(c.silhouette)
print(sil_coeffs)

sns.distplot(sil_coeffs)
plt.show()


