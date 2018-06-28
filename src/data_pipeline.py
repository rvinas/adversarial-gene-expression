import numpy as np
import pandas as pd

DATA_DIR = '../data/E_coli_v4_Build_6'
SUBSET_DIR = '../data/subsets'
REGULATORY_NET_DIR = '../data/regulatory_networks'
DEFAULT_DATAFILE = 'E_coli_v4_Build_6_chips907probes4297.tab'
PROBE_DESCRIPTIONS = 'E_coli_v4_Build_6.probe_set_descriptions'
DEFAULT_SUBSET = 'crp'
DEFAULT_TF_TG = 'tf_tg'


def _parse(lines):
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
    # Transform gene names
    descr = pd.read_csv('{}/{}'.format(DATA_DIR, PROBE_DESCRIPTIONS),
                        delimiter='\t')
    names = descr['probe_set_name'].values
    symbols = descr['gene_symbol'].values
    symbols_dict = {name: symbol for name, symbol in zip(names, symbols)}

    return [symbols_dict[name].lower() for name in gene_names]



def _load_data(name=DEFAULT_DATAFILE, subset=DEFAULT_SUBSET):
    # Parse data file
    with open('{}/{}'.format(DATA_DIR, name), 'r') as f:
        lines = f.readlines()
    expr, gene_names, sample_names = _parse(lines)
    print('Found {} genes in datafile'.format(len(gene_names)))

    # Transform gene names to gene symbols
    gene_symbols = _transform_to_gene_symbols(gene_names)

    # Select subset of genes
    if subset is not None:
        with open('{}/{}'.format(SUBSET_DIR, subset), 'r') as f:
            gene_subset = f.readline() \
                .replace(u'\xa0', '') \
                .rstrip() \
                .lower() \
                .split(',')
            print('Found {} genes in {} subset'.format(len(gene_subset), subset))
        gene_idxs = []
        symb = []
        for gene in gene_subset:
            try:
                gene_idxs.append(gene_symbols.index(gene))
                symb.append(gene)
            except ValueError:
                print('Warning: Gene {} not in gene list'.format(gene))
        print('Selecting {} genes ...'.format(len(gene_idxs)))
        expr = expr[:, gene_idxs]
        gene_symbols = symb

    return expr, gene_symbols, sample_names


def tf_tg_interactions(tf_tg_file=DEFAULT_TF_TG, minimum_evidence='Strong'):
    # Parse TF-TG regulatory interactions
    df = pd.read_csv('{}/{}'.format(REGULATORY_NET_DIR, tf_tg_file),
                     delimiter='\t',
                     comment='#',
                     header=None,
                     names=['tf', 'tg', 'effect', 'evidences', 'strength'])

    # Set minimum evidence level
    accepted_evidences = ['Confirmed']
    if minimum_evidence != 'Confirmed':
        accepted_evidences.append('Strong')
        if minimum_evidence != 'Strong':
            accepted_evidences.append('Weak')

    # Select TF-TG interactions with minimum evidence level
    df = df[df['strength'].isin(accepted_evidences)]
    tf_tg = {}
    for tf, tg in df[['tf', 'tg']].values:
        tf, tg = tf.lower(), tg.lower()
        if tf in tf_tg:
            tf_tg[tf].append(tg)
        else:
            tf_tg[tf] = [tg]

    return tf_tg


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
        indices_second.append(len(sample_names)-1)

    assert len(set(indices_first + indices_second)) == len(sample_names)
    return indices_first, indices_second


def load_data(name=DEFAULT_DATAFILE, subset=DEFAULT_SUBSET):
    return _load_data(name, subset)

if __name__ == '__main__':
    expr, gene_symbols, sample_names = load_data()
    # tf_tg_interactions()

    idxs_first, idxs_second = split_replicates(sample_names)
    print(idxs_first)
    print(idxs_second)
    print(len(idxs_first))
    assert len(set(idxs_first+idxs_second)) == len(sample_names)
