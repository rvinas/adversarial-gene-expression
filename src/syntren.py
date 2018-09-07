from data_pipeline import reg_network, get_gene_symbols
import numpy as np
import pandas as pd

DEFAULT_ROOT_GENE = 'CRP'
DEFAULT_EVIDENCE = 'Weak'
DEFAULT_DEPTH = np.inf
SYNTREN_DATA_DIR = '../data/syntren/'  # '../../syntren1.2release/data/'
NETWORK_FILE = SYNTREN_DATA_DIR + 'source_networks/EColi_n{}_r{}_e{}_d{}.sif'
RESULTS_FILE = SYNTREN_DATA_DIR + 'results/nn{}_nbgr{}_hop{}_bionoise{}_expnoise{}_corrnoise{' \
                                  '}_neighAdd_{}_dataset.txt'
DEFAULT_BACKGROUND_NODES = 0
DEFAULT_HOP = 0.3  # Probability for complex 2-regulator interactions
DEFAULT_BIONOISE = 0.1
DEFAULT_EXPNOISE = 0.1
DEFAULT_CORRNOISE = 0.1


def dump_network(root_gene=DEFAULT_ROOT_GENE, minimum_evidence=DEFAULT_EVIDENCE, depth=DEFAULT_DEPTH, break_loops=True):
    """
    Writes network for root_gene in SyNTReN format (directory: SYNTREN_DATA_DIR)
    :param root_gene: Gene on top of the hierarchy
    :param minimum_evidence: Interactions with a strength below this level will be discarded.
           Possible values: 'confirmed', 'strong', 'weak'
    :param depth: Ignores genes that are not in the first max_depth levels of the hierarchy
    :param break_loops: Whether to break the loops from lower (or equal) to upper levels in the hierarchy.
           If True, the resulting network is a Directed Acyclic Graph (DAG).
    """
    gs = get_gene_symbols()
    nodes, edges = reg_network(gs, root_gene, minimum_evidence, depth, break_loops)
    nb_nodes = len(nodes)
    file_name = NETWORK_FILE.format(nb_nodes, root_gene, minimum_evidence, depth)
    with open(file_name, 'w') as f:
        for tf, tg_dict in edges.items():
            for tg, reg_type in tg_dict.items():
                s_reg_type = 'ac'
                if reg_type == '-':
                    s_reg_type = 're'
                elif reg_type == '+-':
                    s_reg_type = 'du'
                elif reg_type == '?':
                    s_reg_type = 'du'
                    print('Warning: Unknown regulation type found for interaction {}->{}'.format(tf, tg))
                elif reg_type != '+':
                    raise ValueError('Invalid regulation type')
                line = '{}\t{}\t{}\n'.format(tf, s_reg_type, tg)
                f.write(line)
    print('Finished preparing SynTReN network\nFile:{}'.format(file_name))


def syntren_results(root_gene=DEFAULT_ROOT_GENE, minimum_evidence=DEFAULT_EVIDENCE, depth=DEFAULT_DEPTH,
                    break_loops=True, nb_background=DEFAULT_BACKGROUND_NODES, hop=DEFAULT_HOP,
                    bionoise=DEFAULT_BIONOISE, expnoise=DEFAULT_EXPNOISE, corrnoise=DEFAULT_CORRNOISE,
                    normalized=True):
    """
    Reads SynTReN results
    :param root_gene: Gene on top of the hierarchy
    :param minimum_evidence: Interactions with a strength below this level will be discarded.
           Possible values: 'confirmed', 'strong', 'weak'
    :param depth: Ignores genes that are not in the first max_depth levels of the hierarchy
    :param break_loops: Whether to break the loops from lower (or equal) to upper levels in the hierarchy.
           If True, the resulting network is a Directed Acyclic Graph (DAG).
    :param nb_background: Number of background nodes
    :param hop: Probability for complex 2-regulator interactions
    :param bionoise: Biological noise [0, 1]
    :param expnoise: Experimental noise [0, 1]
    :param corrnoise: Noise on correlated inputs [0, 1]
    :param normalized: Whether to get SynTReN normalized or unnormalized data
    :return: expression matrix with Shape=(nb_samples, nb_genes), and list of gene symbols
    """
    # Read results
    gs = get_gene_symbols()
    nodes, edges = reg_network(gs, root_gene, minimum_evidence, depth, break_loops)
    nb_nodes = len(nodes)
    norm = 'maxExpr1'
    if not normalized:
        norm = 'unnormalized'
    file_name = RESULTS_FILE.format(nb_nodes, nb_background, hop, bionoise, expnoise, corrnoise, norm)
    df = pd.read_csv(file_name, delimiter='\t')

    # Get data and discard background genes
    symbols = df['GENE'].values
    gene_symbols, backgr_symbols = symbols[:nb_nodes], symbols[nb_nodes:]
    expr = df.loc[:, df.columns != 'GENE'].values.T
    expr, expr_background = expr[:, :nb_nodes], expr[:, nb_nodes:]

    return expr, list(gene_symbols)


if __name__ == '__main__':
    dump_network(minimum_evidence='Weak')
    expr, gene_symbols = syntren_results(minimum_evidence='Weak', nb_background=0)
