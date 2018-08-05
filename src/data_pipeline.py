import numpy as np
import pandas as pd
import pickle
import random
import csv

DATA_DIR = '../data/E_coli_v4_Build_6'
SYNTHETIC_DIR = '../data/artificial'
SUBSET_DIR = '../data/subsets'
REGULATORY_NET_DIR = '../data/regulatory_networks'
CSV_DIR = '../data/csv'
DEFAULT_DATAFILE = 'E_coli_v4_Build_6_chips907probes4297.tab'
PROBE_DESCRIPTIONS = 'E_coli_v4_Build_6.probe_set_descriptions'
DEFAULT_REGULATORY_INTERACTIONS = 'regulatory_interactions'
DEFAULT_TF_TG = 'tf_tg'
DEFAULT_ROOT_GENE = 'CRP'
DEFAULT_EVIDENCE = 'Weak'


def _parse(lines):
    """
    Parse lines from expression file
    :param lines: list of lines. First row and column have the sample and the gene names, respectively.
    :return: expression np.array with Shape=(nb_samples, nb_genes), gene names with Shape=(nb_genes,) and sample
             names with Shape=(nb_samples,)
    """
    # Parse lines containing gene expressions
    lines_iter = iter(lines)
    line = next(lines_iter, None)
    sample_names = line.split()[1:]
    gene_names = []
    expression_values = []
    line = next(lines_iter, None)
    while line:
        split = line.split()
        gene_name, expr = split[0], split[1:]
        gene_names.append(gene_name)
        expression_values.append(expr)
        line = next(lines_iter, None)
    expression_values = np.array(expression_values, dtype=np.float64).T

    return expression_values, gene_names, sample_names


def _transform_to_gene_symbols(gene_names):
    """
    Transforms list of gene names to gene symbols (more common names) according to PROBE_DESCRIPTIONS
    :param gene_names: list of gene names with Shape=(nb_genes,)
    :return: list of gene symbols (lowercase) for each gene name, with Shape=(nb_genes,)
    """
    # Transform gene names
    descr = pd.read_csv('{}/{}'.format(DATA_DIR, PROBE_DESCRIPTIONS),
                        delimiter='\t')
    names = descr['probe_set_name'].values
    symbols = descr['gene_symbol'].values
    symbols_dict = {name: symbol for name, symbol in zip(names, symbols)}

    return [symbols_dict[name].lower() for name in gene_names]


def _reg_int_df(file_name, minimum_evidence=DEFAULT_EVIDENCE):
    """
    Generates Pandas dataframe containing the regulatory interactions, with columns 'tf', 'tg', 'effect',
    'evidences', 'strength'
    :param file_name: name of the file where the interactions will be read from. The file has
           the RegulonDB format
    :param minimum_evidence: Interactions with a strength below this level will be discarded.
           Possible values: 'confirmed', 'strong', 'weak'
    :return: Pandas dataframe containing the regulatory interactions
    """
    # Parse regulatory interactions
    df = pd.read_csv(file_name,
                     delimiter='\t',
                     comment='#',
                     header=None,
                     names=['tf', 'tg', 'effect', 'evidences', 'strength'])

    # Set minimum evidence level
    accepted_evidences = ['Confirmed']
    if minimum_evidence.lower() != 'confirmed':
        accepted_evidences.append('Strong')
        if minimum_evidence.lower() != 'strong':
            accepted_evidences.append('Weak')
    df = df[df['strength'].isin(accepted_evidences)]
    df['tf'] = df['tf'].str.lower()
    df['tg'] = df['tg'].str.lower()

    return df


def _gene_subset(root_gene, reg_int, minimum_evidence=DEFAULT_EVIDENCE, max_depth=np.inf):
    """
    Obtains list of gene symbols of genes that are in the hierarchy of root_gene (genes that are directly or
    indirectly regulated by root_gene, including root_gene itself).
    :param root_gene: Gene on top of the hierarchy
    :param reg_int: Name of the RegulonDB file in SUBSET_DIR where the regulatory interactions will be read from
    :param minimum_evidence: Interactions with a strength below this level will be discarded.
           Possible values: 'confirmed', 'strong', 'weak'
    :param max_depth: Ignores genes that are not in the first max_depth levels of the hierarchy
    :return: list of genes in the root_gene hierarchy
    """
    # Get dataframe of regulatory interactions
    file_name = '{}/{}'.format(SUBSET_DIR, reg_int)
    df = _reg_int_df(file_name, minimum_evidence)

    # Find genes under root_gene in the regulatory network
    depth = 0
    gene_set = set()
    new_gene_set = {root_gene.lower()}
    while len(gene_set) != len(new_gene_set) and depth <= max_depth:
        gene_set = new_gene_set
        df_tfs = df[df['tf'].isin(gene_set)]
        tgs = df_tfs['tg'].values
        new_gene_set = gene_set.union(tgs)
        depth += 1

    return sorted(list(gene_set))


def _load_data(name, root_gene=None, reg_int=None, minimum_evidence=None, max_depth=None):
    """
    Loads data from the file with the given name in DATA_DIR. Selects genes from the hierarchy of root_gene (genes
    that are directly or indirectly regulated by root_gene, including root_gene itself) according to RegulonDB file
    reg_int, selecting only interactions with at least an evidence of minimum_evidence down to a certain level
    max_depth.
    :param name: name from file in DATA_DIR containing the expression data
    :param root_gene: Gene on top of the hierarchy. If None, the full dataset is returned
    :param reg_int: Name of the RegulonDB file in SUBSET_DIR where the regulatory interactions will be read from
    :param minimum_evidence: Interactions with a strength below this level will be discarded.
           Possible values: 'confirmed', 'strong', 'weak'
    :param max_depth: Ignores genes that are not in the first max_depth levels of the hierarchy
    :return: expression np.array with Shape=(nb_samples, nb_genes), gene symbols with Shape=(nb_genes,) and sample
             names with Shape=(nb_samples,)
    """
    # Parse data file
    with open('{}/{}'.format(DATA_DIR, name), 'r') as f:
        lines = f.readlines()
    expr, gene_names, sample_names = _parse(lines)
    print('Found {} genes in datafile'.format(len(gene_names)))

    # Transform gene names to gene symbols
    gene_symbols = _transform_to_gene_symbols(gene_names)

    # Select subset of genes
    if root_gene is not None:
        gene_subset = _gene_subset(root_gene, reg_int, minimum_evidence, max_depth)
        print('Found {} genes in {} regulatory network'.format(len(gene_subset), root_gene))
        gene_idxs = []
        symb = []
        for gene in gene_subset:
            try:
                gene_idxs.append(gene_symbols.index(gene))
                symb.append(gene)
            except ValueError:
                pass
                # print('Warning: Gene {} not in gene list'.format(gene))
        print('{} genes not in gene subset. Selecting {} genes ...'.format(expr.shape[1], len(gene_idxs)))
        expr = expr[:, gene_idxs]
        gene_symbols = symb

    return expr, gene_symbols, sample_names


def tf_tg_interactions(tf_tg_file=DEFAULT_TF_TG, minimum_evidence=DEFAULT_EVIDENCE):
    """
    Get dictionary of TF-TG interactions
    :param tf_tg_file: name of RegulonDB file in REGULATORY_NET_DIR containing the interactions
    :param minimum_evidence: Interactions with a strength below this level will be discarded.
           Possible values: 'confirmed', 'strong', 'weak'
    :return: dictionary with containing the list of TGs (values) for each TF (keys)
    """
    # Parse TF-TG regulatory interactions
    file_name = '{}/{}'.format(REGULATORY_NET_DIR, tf_tg_file)
    df = _reg_int_df(file_name, minimum_evidence)

    # Select TF-TG interactions with minimum evidence level
    tf_tg = {}
    for tf, tg in df[['tf', 'tg']].values:
        if tf in tf_tg:
            if tg not in tf_tg[tf]:
                tf_tg[tf].append(tg)
        else:
            tf_tg[tf] = [tg]

    return tf_tg


def get_gene_symbols(name=DEFAULT_DATAFILE):
    """
    Returns gene symbols from the file with the given name in DATA_DIR
    :param name: name from file in DATA_DIR containing the expression data
    :return: list of gene symbols with Shape=(nb_genes,)
    """
    # Parse data file
    with open('{}/{}'.format(DATA_DIR, name), 'r') as f:
        lines = f.readlines()
    expr, gene_names, sample_names = _parse(lines)
    print('Found {} genes in datafile'.format(len(gene_names)))

    # Transform gene names to gene symbols
    return _transform_to_gene_symbols(gene_names)


def reg_network(gene_symbols, root_gene=DEFAULT_ROOT_GENE, minimum_evidence=DEFAULT_EVIDENCE, max_depth=np.inf,
                break_loops=True, reg_int=DEFAULT_REGULATORY_INTERACTIONS):
    """
    Returns the regulatory network of the genes in gene_symbols.
    :param gene_symbols: List of genes to be included in the network.
    :param root_gene: Gene on top of the hierarchy
    :param minimum_evidence: Interactions with a strength below this level will be discarded.
           Possible values: 'confirmed', 'strong', 'weak'
    :param max_depth: Ignores genes that are not in the first max_depth levels of the hierarchy
    :param break_loops: Whether to break the loops from lower (or equal) to upper levels in the hierarchy.
           If True, the resulting network is a Directed Acyclic Graph (DAG).
    :param reg_int: Name of the RegulonDB file in SUBSET_DIR where the regulatory interactions will be read from
    :return: list of nodes and dictionary of edges (keys: *from* node, values: dictionary with key *to* node and
             value regulation type)
    """
    # Get dataframe of regulatory interactions
    file_name = '{}/{}'.format(SUBSET_DIR, reg_int)
    df = _reg_int_df(file_name, minimum_evidence)
    df = df[df['tf'].isin(gene_symbols)]
    df = df[df['tg'].isin(gene_symbols)]

    # Construct network
    depth = 0
    nodes = set()
    nodes_new = {root_gene.lower()}
    edges = {}
    while len(nodes) != len(nodes_new) and depth <= max_depth:
        df_tfs = df[~df['tf'].isin(nodes)]
        nodes = nodes_new
        df_tfs = df_tfs[df_tfs['tf'].isin(nodes_new)]
        tfs = df_tfs['tf'].values
        tgs = df_tfs['tg'].values
        reg_types = df_tfs['effect'].values
        nodes_new = nodes.union(tgs)
        depth += 1
        for tf, tg, reg_type in zip(tfs, tgs, reg_types):
            # We might want to avoid loops. TODO: Take evidence into account?
            if break_loops and tg in edges:
                print('Warning: Breaking loop: {}->{}'.format(tf, tg))
            elif break_loops and tf == tg:
                print('Warning: Breaking autoregulation loop: {}->{}'.format(tf, tg))
            else:
                if tf in edges:
                    # If edge is present, check if it is necessary to change its regulation type
                    if tg in edges[tf] and (
                            (edges[tf][tg] == '+' and reg_type == '-') or (edges[tf][tg] == '-' and reg_type == '+')):
                        edges[tf][tg] = '+-'
                    elif tg not in edges[tf]:
                        edges[tf][tg] = reg_type
                else:
                    edges[tf] = {tg: reg_type}

    return nodes, edges


def split_replicates(sample_names, split_point=-1):
    """
    Splits sample_names into 2 sets. The first set contains replicates 1..split_point
    and the second has replicates split_point+1..infinity.
    :param sample_names: list of sample names
    :param split_point: where to make the split. Set split_point=-1 to include the last replicate
                        in the second set
    :return: indices of first and second subset of samples, respectively.
    """
    indices_first = []
    indices_second = []
    repl_ant = -1
    for i, name in enumerate(sample_names):
        repl_nb = int(name.split('_')[-1][1:])
        if split_point == -1:
            if repl_ant != -1:
                if repl_ant >= repl_nb:
                    indices_second.append(i - 1)
                else:
                    indices_first.append(i - 1)
            repl_ant = repl_nb
        elif repl_nb <= split_point:
            indices_first.append(i)
        else:
            indices_second.append(i)

    # Last replicate goes to second set if split_point=-1
    if split_point == -1:
        indices_second.append(len(sample_names) - 1)

    assert len(set(indices_first + indices_second)) == len(sample_names)
    return indices_first, indices_second


def split_train_test(sample_names, train_rate=0.75, seed=0):
    """
    Split data into a train and a test sets keeping replicates within the same set
    :param sample_names: list of sample names
    :param train_rate: percentage of training samples
    :param seed: random seed
    :return: lists of train and test sample indices
    """
    # Set random seed
    random.seed(seed)

    # Find replicate segments
    replicate_ranges = {}
    for i, name in enumerate(sample_names):
        repl_nb = int(name.split('_')[-1][1:])
        n = len(replicate_ranges)
        if repl_nb == 1:  # First replicate sample
            replicate_ranges[n] = {'start': i, 'end': i}
        else:
            replicate_ranges[n - 1]['end'] = i

    # Permute unique samples
    unique_sample_idxs = list(replicate_ranges.keys())
    random.shuffle(unique_sample_idxs)

    # Split data
    nb_unique = len(unique_sample_idxs)
    split_point = int(train_rate * nb_unique)
    unique_train = unique_sample_idxs[:split_point]

    # Recover replicates
    train_idxs = []
    test_idxs = []
    for i in range(nb_unique):
        for j in range(replicate_ranges[i]['start'], replicate_ranges[i]['end'] + 1):
            if i in unique_train:
                train_idxs.append(j)
            else:
                test_idxs.append(j)

    assert len(set(train_idxs + test_idxs)) == len(sample_names)
    return train_idxs, test_idxs


def clip_outliers(expr, gene_means=None, gene_stds=None, std_clip=2):
    """
    Clips gene expressions of samples in which the gene deviates more than std_clip standard deviations from the gene mean.
    :param expr: np.array of expression data with Shape=(nb_samples, nb_genes)
    :param gene_means: np.array with the mean of each gene. Shape=(nb_genes,). If None, it is computed from expr
    :param gene_stds: np.array with the std of each gene. Shape=(nb_genes,). If None, it is computed from expr
    :param std_clip: Number of standard deviations for which the expression of a gene will be clipped
    :return: clipped expression matrix
    """
    nb_samples, nb_genes = expr.shape

    # Find gene means and stds
    if gene_means is None:
        gene_means = np.mean(expr, axis=0)
    if gene_stds is None:
        gene_stds = np.std(expr, axis=0)

    # Clip samples for which a gene is not within [gene_mean - std_clip * gene_std, gene_mean + std_clip * gene_std]
    clipped_expr = np.array(expr)
    for sample_idx in range(nb_samples):
        for gene_idx in range(nb_genes):
            if expr[sample_idx, gene_idx] > (gene_means + std_clip * gene_stds)[gene_idx]:
                clipped_expr[sample_idx, gene_idx] = (gene_means + std_clip * gene_stds)[gene_idx]
            elif expr[sample_idx, gene_idx] < (gene_means - std_clip * gene_stds)[gene_idx]:
                clipped_expr[sample_idx, gene_idx] = (gene_means - std_clip * gene_stds)[gene_idx]

    return clipped_expr


def load_data(name=DEFAULT_DATAFILE, root_gene=DEFAULT_ROOT_GENE, minimum_evidence=DEFAULT_EVIDENCE, max_depth=np.inf,
              reg_int=DEFAULT_REGULATORY_INTERACTIONS):
    """
    Loads data from the file with the given name in DATA_DIR. Selects genes from the hierarchy of root_gene (genes
    that are directly or indirectly regulated by root_gene, including root_gene itself) according to RegulonDB file
    reg_int, selecting only interactions with at least an evidence of minimum_evidence down to a certain level
    max_depth.
    :param name: name from file in DATA_DIR containing the expression data
    :param root_gene: Gene on top of the hierarchy. If None, the full dataset is returned
    :param reg_int: Name of the RegulonDB file in SUBSET_DIR where the regulatory interactions will be read from
    :param minimum_evidence: Interactions with a strength below this level will be discarded.
           Possible values: 'confirmed', 'strong', 'weak'
    :param max_depth: Ignores genes that are not in the first max_depth levels of the hierarchy
    :return: expression np.array with Shape=(nb_samples, nb_genes), gene symbols with Shape=(nb_genes,) and sample
             names with Shape=(nb_samples,)
    """
    return _load_data(name, root_gene, reg_int, minimum_evidence, max_depth)


def save_synthetic(name, expr, gene_symbols):
    """
    Saves expression data with Shape=(nb_samples, nb_genes) to pickle file with the given name in SYNTHETIC_DIR.
    :param name: name of the file in SYNTHETIC_DIR where the expression data will be saved
    :param expr: np.array of expression data with Shape=(nb_samples, nb_genes)
    :param gene_symbols: list of gene symbols matching the columns of expr
    """
    file = '{}/{}.pkl'.format(SYNTHETIC_DIR, name)
    data = {'expr': expr,
            'gene_symbols': gene_symbols}
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_synthetic(name):
    """
    Loads expression data from pickle file with the given name (produced by save_synthetic function)
    :param name: name of the pickle file in SYNTHETIC_DIR containing the expression data
    :return: np.array of expression with Shape=(nb_samples, nb_genes) and list of gene symbols matching the columns
    of expr
    """
    file = '{}/{}.pkl'.format(SYNTHETIC_DIR, name)
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data['expr'], data['gene_symbols']


def write_csv(name, expr, gene_symbols, sample_names=None, nb_decimals=5):
    """
    Writes expression data to a CSV file
    :param name: file name
    :param expr: expression matrix. Shape=(nb_samples, nb_genes)
    :param gene_symbols: list of gene symbols. Shape=(nb_genes,)
    :param sample_names: list of gene samples or None. Shape=(nb_samples,)
    :param nb_decimals: number of decimals for expression values
    """
    expr_rounded = np.around(expr, decimals=nb_decimals)
    with open('{}/{}'.format(CSV_DIR, name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow([' '] + [g for g in gene_symbols])
        for i, e in enumerate(expr_rounded):
            sample_name = ' '
            if sample_names is not None:
                sample_name = sample_names[i]
            writer.writerow([sample_name] + list(e))


if __name__ == '__main__':
    expr, gene_symbols, sample_names = load_data()
    # tf_tg_interactions()

    idxs_first, idxs_second = split_replicates(sample_names)
    print(idxs_first)
    print(idxs_second)
    print(len(idxs_first))
    assert len(set(idxs_first + idxs_second)) == len(sample_names)

    idxs_train, idxs_test = split_train_test(sample_names, train_rate=0.7)
    print(idxs_train)
    print(idxs_test)
