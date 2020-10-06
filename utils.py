import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
import pickle
import random
from statsmodels.stats.multitest import multipletests
from matplotlib.collections import EllipseCollection
import scipy

RNAS_DIR = 'data/RNAseqDB/normalized'
RNAS_EXPR_FILE = '{}/{}'.format(RNAS_DIR, '_expr.csv')
RNAS_INFO_FILE = '{}/{}'.format(RNAS_DIR, '_info.csv')
RNAS_METADATA_FILE = '{}/../{}'.format(RNAS_DIR, '_metadata.csv')
RNAS_TISSUE_MAPPING_FILE = '{}/../{}'.format(RNAS_DIR, 'tissue_mapping.txt')
RNAS_GTEX_TISSUE_MAPPING_FILE = '{}/../{}'.format(RNAS_DIR, 'gtex_tissue_mapping.txt')
RNAS_GTEX_TISSUE_MAPPING_RNAS_TISSUE = 'rnaseqdb_tissue'
RNAS_GTEX_TISSUE_MAPPING_SLIDE_TISSUE = 'slide_tissue'
RNAS_TISSUE_MAPPING_GTEX_TISSUE = 'GTEx_Tissue'
RNAS_TISSUE_MAPPING_TCGA_TISSUE = 'TCGA_Tissue'
RNAS_INFO_SAMPLE_ID = 'SAMPID'
RNAS_INFO_TISSUE = 'TISSUE_GTEX'  # Default format: GTEx
RNAS_INFO_TISSUE_TCGA = 'TISSUE_TCGA'
RNAS_INFO_DATASET = 'DATASET'
RNAS_METADATA_SAMPLE_ID = RNAS_INFO_SAMPLE_ID
RNAS_METADATA_TILES_DIRS = 'TILES_DIR'
RNAS_METADATA_TISSUE = RNAS_INFO_TISSUE
RNAS_METADATA_TISSUE_TCGA = RNAS_INFO_TISSUE_TCGA
RNAS_METADATA_DATASET = RNAS_INFO_DATASET
RNAS_METADATA_SPLIT_GROUP = 'SPLIT'
RNAS_ENTREZ_ID = 'Entrez_Gene_Id'
RNAS_HUGO_SYMBOL = 'Hugo_Symbol'

ECOLI_DIR = 'data/E_coli_v4_Build_6'
ECOLI_SUBSET_DIR = 'data/subsets'
ECOLI_REGULATORY_NET_DIR = 'data/regulatory_networks'
ECOLI_DEFAULT_DATAFILE = 'E_coli_v4_Build_6_chips907probes4297.tab'
ECOLI_PROBE_DESCRIPTIONS = 'E_coli_v4_Build_6.probe_set_descriptions'
ECOLI_EXPERIMENT_CONDITIONS = 'E_coli_v4_Build_6.experiment_descriptions'
ECOLI_DEFAULT_REGULATORY_INTERACTIONS = 'regulatory_interactions'
ECOLI_DEFAULT_TF_TG = 'tf_tg'
ECOLI_DEFAULT_ROOT_GENE = 'CRP'
ECOLI_DEFAULT_EVIDENCE = 'Weak'


# ------------------
# RNASeqDB
# ------------------

def rnaseqdb_df(file=RNAS_EXPR_FILE):
    """
    Loads RNASeqDB expression dataset
    :param file: RNASeqDB expression file name
    :return: Pandas dataframe with RNA-Seq values (rows: genes, cols: samples)
    """
    df = pd.read_csv(file,
                     delimiter='\t',
                     index_col=[0, 1])
    # df = df + eps
    # df = df.apply(np.log2)
    return df


def rnaseqdb_gene_symbols(file=RNAS_EXPR_FILE):
    """
    Reads RNASeqDB gene symbols
    :param file: RNASeqDB expression file name
    :return: list of gene symbols
    """
    df = pd.read_csv(file,
                     delimiter='\t',
                     usecols=[RNAS_HUGO_SYMBOL])
    return df[RNAS_HUGO_SYMBOL].values


def rnaseqdb_info_df(file=RNAS_INFO_FILE):
    """
    Loads RNASeqDB info dataframe
    :param file: RNASeqDB info file name
    :return: Pandas dataframe with information about each sample
    """
    df = pd.read_csv(file,
                     delimiter='\t',
                     index_col=0)
    return df


def rnaseqdb_tcga_barcode_to_sample_id(barcode):
    """
    Selects sample ID from TCGA barcode
    :param barcode: TCGA barcode
    :return: sample ID
    """
    split = barcode.split('-')[:4]
    return '-'.join(split).upper()[:-1]


def rnaseqdb_gtex_barcode_to_sample_id(barcode):
    """
    Selects sample ID from GTEx barcode
    :param barcode: GTEx barcode
    :return: sample ID
    """
    # https://sites.google.com/broadinstitute.org/gtex-faqs/home
    split = barcode.split('-')[:3]
    return '-'.join(split)[:-1].upper()  # Note: discarding last digit to match images


def rnaseqdb_barcodes_to_sample_ids(barcodes):
    """
    Selects sample IDs from GTEx or TCGA barcodes
    :param barcodes: array of GTEx or TCGA barcodes
    :return: sample IDs
    """
    sample_ids = []
    for b in barcodes:
        if b.startswith('GTEX'):
            sample_id = rnaseqdb_gtex_barcode_to_sample_id(b)
        elif b.startswith('TCGA'):
            sample_id = rnaseqdb_tcga_barcode_to_sample_id(b)
        else:
            raise ValueError('Unknown barcode: {}'.format(b))
        sample_ids.append(sample_id)
    return np.array(sample_ids)


def rnaseqdb_tissue_mapping_df(file=RNAS_TISSUE_MAPPING_FILE):
    """
    Loads TCGA to GTEx mapping dataframe
    :param file: mapping file name
    :return: Pandas dataframe with tissue mapping
    """
    df = pd.read_csv(file,
                     delimiter='\t')
    return df


def rnaseqdb_tissue_mapping(file=RNAS_TISSUE_MAPPING_FILE):
    """
    Maps TCGA tissues to GTEx tissues
    :param file: RNASeqDB issue mapping file name
    :return: dict with key: TCGA tissue, value: GTEx tissue
    """
    df = rnaseqdb_tissue_mapping_df(file)
    gtex_tissues = df[RNAS_TISSUE_MAPPING_GTEX_TISSUE].values
    tcga_tissues = df[RNAS_TISSUE_MAPPING_TCGA_TISSUE].values
    tcga_gtex_map = {k: v for k, v in zip(tcga_tissues, gtex_tissues)}
    gtex_tcga_map = {k: v for k, v in zip(gtex_tissues, tcga_tissues)}
    return tcga_gtex_map, gtex_tcga_map


def rnaseqdb_join_datasets(dir=RNAS_DIR, mapping_file=RNAS_TISSUE_MAPPING_FILE):
    """
    Joins the RNASeqDB individual datasets into a single dataframe
    :param dir: RNASeqDB directory
    :param mapping_file: RNASeqDB issue mapping file name
    :return: full dataset
    """
    # Find all files
    files = os.listdir(dir)
    regex = re.compile(r'.txt$')
    files = filter(regex.search, files)

    # Load and join datasets
    info_df_cols = [RNAS_INFO_SAMPLE_ID, RNAS_INFO_TISSUE, RNAS_INFO_TISSUE_TCGA, RNAS_INFO_DATASET]
    sample_info_df = pd.DataFrame(columns=info_df_cols)
    sample_info_df.set_index(RNAS_INFO_SAMPLE_ID, inplace=True)
    joined_df = None
    tcga_gtex_map, gtex_tcga_map = rnaseqdb_tissue_mapping(mapping_file)  # WARNING: Not 1-to-1 maps!!
    for file in files:
        # Find tissue and dataset information
        split = file.split('-')
        tissue = split[0]
        end_groups = split[3:]
        dataset_name = '-'.join(end_groups).split('.')[0]
        print('File: {}. Tissue: {}. Dataset: {}'.format(file, tissue, dataset_name))

        # Make tissue names uniform
        tissue_gtex = tissue
        tissue_tcga = tissue
        if dataset_name == 'gtex':
            tissue_tcga = gtex_tcga_map[tissue]
        else:
            tissue_gtex = tcga_gtex_map[tissue]

        # Join datasets
        df = rnaseqdb_df('{}/{}'.format(dir, file))
        if joined_df is None:
            joined_df = df
        else:
            print('Index: ', joined_df.index)
            joined_df = pd.merge(joined_df,  # .reset_index(),
                                 df,  # .reset_index(),
                                 left_index=True, right_index=True)
            # on=[RNAS_HUGO_SYMBOL, RNAS_ENTREZ_ID])
        print('Shape: {}. Memory usage: {}'.format(joined_df.shape,
                                                   joined_df.memory_usage(deep=True).values.sum() / 1024 ** 3))

        # Fill sample info in dataframe
        sample_ids = df.columns.values
        nb_samples = len(sample_ids)
        info_df = pd.DataFrame({RNAS_INFO_SAMPLE_ID: sample_ids,
                                RNAS_INFO_TISSUE: [tissue_gtex] * nb_samples,
                                RNAS_INFO_TISSUE_TCGA: [tissue_tcga] * nb_samples,
                                RNAS_INFO_DATASET: [dataset_name] * nb_samples})
        info_df.set_index(RNAS_INFO_SAMPLE_ID, inplace=True)

        sample_info_df = sample_info_df.append(info_df)
        print('Info shape: {}'.format(sample_info_df.shape))

    return joined_df, sample_info_df


def rnaseqdb_save(expr_df, info_df, expr_file=RNAS_EXPR_FILE, info_file=RNAS_INFO_FILE):
    """
    Saves RNASeqDB
    :param expr_df: expression dataframe. Shape=(nb_genes, nb_samples)
    :param info_df: sample information dataframe (sample ID, tissue, original dataset). Shape=(nb_samples, 3)
    :param expr_file: expressions file name
    :param info_file: sample information file name
    """
    print('... saving RNASeqDB dataset')
    expr_df.to_csv(expr_file,
                   sep='\t')
    info_df.to_csv(info_file,
                   sep='\t')


def rnaseqdb_load(expr_file=RNAS_EXPR_FILE, info_file=RNAS_INFO_FILE):
    """
    Loads RNASeqDB
    :param expr_file: expressions file name
    :param info_file: sample information file name
    """
    print('... loading RNASeqDB dataset')
    expr_df = rnaseqdb_df(expr_file)
    info_df = rnaseqdb_info_df(info_file)
    return expr_df, info_df


def rnaseqdb_gtex_tissue_mapping_df(file=RNAS_GTEX_TISSUE_MAPPING_FILE):
    """
    Loads dataframe to match RNASeqDB-GTEx tissues to GTEx slide tissues
    :param file: file path
    :return: Pandas dataframe with tissue mapping
    """
    df = pd.read_csv(file,
                     delimiter='\t',
                     index_col=0)
    return df


# ---------------------
# DATA UTILITIES
# ---------------------

def standardize(x, mean=None, std=None):
    """
    Shape x: (nb_samples, nb_vars)
    """
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    return (x - mean) / std


def split_train_test(x, train_rate=0.75):
    """
    Split data into a train and a test sets
    :param train_rate: percentage of training samples
    :return: x_train, x_test
    """
    nb_samples = x.shape[0]
    split_point = int(train_rate * nb_samples)
    x_train = x[:split_point]
    x_test = x[split_point:]

    return x_train, x_test


def split_train_test_v2(x, sampl_ids, train_rate=0.75):
    """
    Avoids patient leak between train/test set
    Split data into a train and a test sets
    :param train_rate: percentage of training samples
    :return: x_train, x_test
    """
    nb_samples = x.shape[0]
    sample_ids_rev = np.array([s[::-1] for s in sampl_ids])
    split_point = int(train_rate * nb_samples)
    idxs = np.argsort(sample_ids_rev)
    sample_ids_rev_sorted = sample_ids_rev[idxs]

    p = split_point
    while p == nb_samples and sample_ids_rev_sorted[p - 1] == sample_ids_rev_sorted[p - 2]:
        p += 1
    if p == nb_samples:
        raise Exception('Error: Cannot split samples into train and test sets')

    x_train = x[idxs[:p]]
    x_test = x[idxs[p:]]
    sample_ids_train = sampl_ids[idxs[:p]]
    sample_ids_test = sampl_ids[idxs[p:]]

    return x_train, x_test, sample_ids_train, sample_ids_test


def split_train_test_v3(sample_names, train_rate=0.75, seed=0):
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


def save_synthetic(name, data, symbols, datadir):
    """
    Saves data with Shape=(nb_samples, nb_genes) to pickle file with the given name in datadir.
    :param name: name of the file in SYNTHETIC_DIR where the expression data will be saved
    :param data: np.array of data with Shape=(nb_samples, nb_genes)
    :param symbols: list of gene symbols matching the columns of data
    """
    file = '{}/{}.pkl'.format(datadir, name)
    data = {'data': data,
            'symbols': symbols}
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_synthetic(name, datadir):
    """
    Loads data from pickle file with the given name (produced by save_synthetic function)
    :param name: name of the pickle file in datadir containing the expression data
    :return: np.array of expression with Shape=(nb_samples, nb_genes) and list of gene symbols matching the columns
    of data
    """
    file = '{}/{}.pkl'.format(datadir, name)
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data['data'], data['symbols']


def reg_network(gene_symbols, root_gene=ECOLI_DEFAULT_ROOT_GENE, minimum_evidence=ECOLI_DEFAULT_EVIDENCE,
                max_depth=np.inf,
                break_loops=True, reg_int=ECOLI_DEFAULT_REGULATORY_INTERACTIONS):
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
    # IMPORTANT NOTE: This function assumes that the TF in the following file are sorted alphabetically
    file_name = '{}/{}'.format(ECOLI_SUBSET_DIR, reg_int)
    df = _reg_int_df(file_name, minimum_evidence)
    df = df[df['tf'].isin(gene_symbols)]
    df = df[df['tg'].isin(gene_symbols)]

    # Select all genes
    if root_gene is None:
        assert not break_loops  # break_loops not supported. Which loops would we break?
        nodes = gene_symbols
        edges = {}
        tfs = df['tf'].values
        tgs = df['tg'].values
        reg_types = df['effect'].values
        for tf, tg, reg_type in zip(tfs, tgs, reg_types):
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

    # Construct network
    depth = 0
    nodes = set()
    nodes_new = {root_gene.lower()}
    edges = {}
    ancestors = {root_gene.lower(): set()}
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
            if break_loops and tf == tg:
                print('Warning: Breaking autoregulatory loop ({}->{})'.format(tf, tg))
            elif break_loops and tg in ancestors[tf]:
                print('Warning: Breaking loop ({}->{})'.format(tf, tg))
            else:
                if tg in ancestors:
                    ancestors[tg] = ancestors[tg].union({tf})
                    ancestors[tg] = ancestors[tg].union(ancestors[tf])
                else:
                    ancestors[tg] = {tf}.union(ancestors[tf])

                if tf in edges:
                    # If edge is present, check if it is necessary to change its regulation type
                    if tg in edges[tf] and (
                            (edges[tf][tg] == '+' and reg_type == '-') or (edges[tf][tg] == '-' and reg_type == '+')):
                        edges[tf][tg] = '+-'
                    elif tg not in edges[tf]:
                        edges[tf][tg] = reg_type
                else:
                    edges[tf] = {tg: reg_type}

    return list(nodes), edges


# ------------------
# E. coli M3D
# ------------------


def load_ecoli(name=ECOLI_DEFAULT_DATAFILE, root_gene=ECOLI_DEFAULT_ROOT_GENE, minimum_evidence=ECOLI_DEFAULT_EVIDENCE,
               max_depth=np.inf,
               reg_int=ECOLI_DEFAULT_REGULATORY_INTERACTIONS):
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
    return _load_ecoli(name, root_gene, reg_int, minimum_evidence, max_depth)


def _transform_to_gene_symbols(gene_names):
    """
    Transforms list of gene names to gene symbols (more common names) according to PROBE_DESCRIPTIONS
    :param gene_names: list of gene names with Shape=(nb_genes,)
    :return: list of gene symbols (lowercase) for each gene name, with Shape=(nb_genes,)
    """
    # Transform gene names
    descr = pd.read_csv('{}/{}'.format(ECOLI_DIR, ECOLI_PROBE_DESCRIPTIONS),
                        delimiter='\t')
    names = descr['probe_set_name'].values
    symbols = descr['gene_symbol'].values
    symbols_dict = {name: symbol for name, symbol in zip(names, symbols)}

    return [symbols_dict[name].lower() for name in gene_names]


def _reg_int_df(file_name, minimum_evidence=ECOLI_DEFAULT_EVIDENCE):
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


def _gene_subset(root_gene, reg_int, minimum_evidence=ECOLI_DEFAULT_EVIDENCE, max_depth=np.inf):
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
    file_name = '{}/{}'.format(ECOLI_SUBSET_DIR, reg_int)
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


def _load_ecoli(name, root_gene=None, reg_int=None, minimum_evidence=None, max_depth=None):
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
    with open('{}/{}'.format(ECOLI_DIR, name), 'r') as f:
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

    return expr, np.array(gene_symbols), np.array(sample_names)


def ecoli_discrete_conditions(sample_names, conditions=None):
    """
    Loads discrete conditions of E. coli data
    :param sample_names: Sample names
    :param conditions: Condition names
    :return: data, int encoded data and vocabulary dicts
    """
    # pH changes, antibiotics, heat shock, varying glucose and oxygen concentration

    if conditions is None:
        conditions = ['growth_phase',
                      'glucose',
                      'ampicillin',
                      'culture_ph',
                      'aeration',
                      'culture_o2',
                      'culture_temperature']
        # continuous: 'cell_density', 'culture_volume', 'culture_ph',
    with open('{}/{}'.format(ECOLI_DIR, ECOLI_EXPERIMENT_CONDITIONS), 'r', encoding='utf-8',
              errors='ignore') as f:  # encoding='utf-8', errors='ignore'
        lines = f.readlines()

    data = []
    s_names = []
    vocab_dicts = [{None: 0} for _ in conditions]
    for line in lines[1:]:
        splits = line.split('\t')
        sample_data = []
        s_name = splits[0]
        s_names.append(s_name)
        splits = splits[-1].split(', ')

        for i, cond in enumerate(conditions):
            word = None
            for split in splits:
                if split.startswith(cond):
                    word = split.split(':')[-1].strip()
                    sample_data.append(word)
                    break
            if word is None:
                sample_data.append(None)
            if word not in vocab_dicts[i]:
                dict_len = len(vocab_dicts[i])
                vocab_dicts[i][word] = dict_len

        data.append(sample_data)

    data = np.array(data)
    encoded_data = []
    for r in data:
        encoded_row = []
        for i, c in enumerate(r):
            encoded_row.append(vocab_dicts[i][c])
        encoded_data.append(encoded_row)
    encoded_data = np.array(encoded_data)

    # Vocab
    s_names = np.array(s_names)
    idxs = [np.argwhere(s_names == s).ravel()[0] for s in sample_names]
    data = np.array(data)[idxs]
    encoded_data = np.array(encoded_data)[idxs]

    return data, encoded_data, vocab_dicts


def ecoli_continuous_conditions(sample_names, conditions=None):
    """
    Loads continuous conditions of E. coli data
    :param sample_names: Sample names
    :param conditions: Condition names
    :return: data
    """
    # pH changes, antibiotics, heat shock, varying glucose and oxygen concentration

    if conditions is None:
        conditions = ['cell_density', 'culture_volume', 'culture_ph']
    with open('{}/{}'.format(ECOLI_DIR, ECOLI_EXPERIMENT_CONDITIONS), 'r') as f:
        lines = f.readlines()

    data = []
    s_names = []
    for line in lines[1:]:
        splits = line.split('\t')
        sample_data = []
        s_name = splits[0]
        s_names.append(s_name)
        splits = splits[-1].split(', ')

        for i, cond in enumerate(conditions):
            word = None
            for split in splits:
                if split.startswith(cond):
                    word = split.split(':')[-1].strip()
                    sample_data.append(word)
                    break
            if word is None:
                sample_data.append(None)
        data.append(sample_data)

    s_names = np.array(s_names)
    idxs = [np.argwhere(s_names == s).ravel()[0] for s in sample_names]
    data = np.array(data)[idxs]

    return data


def ecoli_get_gene_idxs(gene_symbols, selected_genes=None):
    """
    Get index of each selected gene
    :param gene_symbols: list of gene symbols
    :param selected_genes: list of selected genes
    :return: List of gene indexes
    """
    if selected_genes is None:
        selected_genes = ['crp', 'fur', 'soxs', 'acrr', 'acna', 'fnr', 'soxr', 'arca']
    gene_idxs = []
    for g in selected_genes:
        if not g in gene_symbols:
            print('Warning: Gene {} not found in the list of gene symbols'.format(g))
        else:
            idx = np.argwhere(gene_symbols == g)[0][0]
            gene_idxs.append(idx)
    return gene_idxs
    # return np.ravel([np.argwhere(gene_symbols == g) for g in selected_genes])


def ecoli_get_gene_symbols(name=ECOLI_DEFAULT_DATAFILE):
    """
    Returns gene symbols from the file with the given name in DATA_DIR
    :param name: name from file in DATA_DIR containing the expression data
    :return: list of gene symbols with Shape=(nb_genes,)
    """
    # Parse data file
    with open('{}/{}'.format(ECOLI_DIR, name), 'r') as f:
        lines = f.readlines()
    expr, gene_names, sample_names = _parse(lines)
    # print('Found {} genes in datafile'.format(len(gene_names)))

    # Transform gene names to gene symbols
    return _transform_to_gene_symbols(gene_names)


def tf_tg_interactions(tf_tg_file=ECOLI_DEFAULT_TF_TG, minimum_evidence=ECOLI_DEFAULT_EVIDENCE):
    """
    Get dictionary of TF-TG interactions
    :param tf_tg_file: name of RegulonDB file in REGULATORY_NET_DIR containing the interactions
    :param minimum_evidence: Interactions with a strength below this level will be discarded.
           Possible values: 'confirmed', 'strong', 'weak'
    :return: dictionary with containing the list of TGs (values) for each TF (keys)
    """
    # Parse TF-TG regulatory interactions
    file_name = '{}/{}'.format(ECOLI_REGULATORY_NET_DIR, tf_tg_file)
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


# ---------------------
# CORRELATION UTILITIES
# ---------------------

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


def cosine_similarity(x, y):
    """
    Computes cosine similarity between vectors x and y
    :param x: Array of numbers. Shape=(n,)
    :param y: Array of numbers. Shape=(n,)
    :return: cosine similarity between vectors
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


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


def correlations_list(x, y, corr_fn=pearson_correlation):
    """
    Generates correlation list between all pairs of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :param corr_fn: correlation function taking x and y as inputs
    """
    corr = corr_fn(x, y)
    return upper_diag_list(corr)


def gamma_coef(x, y):
    """
    Compute gamma coefficients for two given expression matrices
    :param x: matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
    :param y: matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    :return: Gamma(D^X, D^Z)
    """
    dists_x = 1 - correlations_list(x, x)
    dists_y = 1 - correlations_list(y, y)
    gamma_dx_dy = pearson_correlation(dists_x, dists_y)
    return gamma_dx_dy


def gamma_coefficients(expr_x, expr_z):
    """
    Compute gamma coefficients for two given expression matrices
    :param expr_x: matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
    :param expr_z: matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    :return: Gamma(D^X, D^Z), Gamma(D^X, T^X), Gamma(D^Z, T^Z), Gamma(T^X, T^Z)
             where D^X and D^Z are the distance matrices of expr_x and expr_z (respectively),
             and T^X and T^Z are the dendrogrammatic distance matrices of expr_x and expr_z (respectively).
             Gamma(A, B) is a function that computes the correlation between the elements in the upper-diagonal
             of A and B.
    """
    # Compute Gamma(D^X, D^Z)
    dists_x = 1 - correlations_list(expr_x, expr_x)
    dists_z = 1 - correlations_list(expr_z, expr_z)
    gamma_dx_dz = pearson_correlation(dists_x, dists_z)

    # Compute Gamma(D^X, T^X)
    xl_matrix = hierarchical_clustering(expr_x)
    gamma_dx_tx, _ = cophenet(xl_matrix, dists_x)

    # Compute Gamma(D^Z, T^Z)
    zl_matrix = hierarchical_clustering(expr_z)
    gamma_dz_tz, _ = cophenet(zl_matrix, dists_z)

    # Compute Gamma(T^X, T^Z)
    gamma_tx_tz = compare_cophenetic(xl_matrix, zl_matrix)

    return gamma_dx_dz, gamma_dx_tx, gamma_dz_tz, gamma_tx_tz


def compute_tf_tg_corrs(expr, gene_symbols, tf_tg=None, flat=True):
    """
    Computes the lists of TF-TG and TG-TG correlations
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param gene_symbols: list of gene symbols matching the expr matrix. Shape=(nb_genes,)
    :param tf_tg: dict with TF symbol as key and list of TGs' symbols as value
    :param flat: whether to return flat lists
    :return: lists of TF-TG and TG-TG correlations, respectively
    """
    if tf_tg is None:
        tf_tg = tf_tg_interactions()
    gene_symbols = np.array(gene_symbols)

    tf_tg_corr = []
    tg_tg_corr = []
    for tf, tgs in tf_tg.items():
        tg_idxs = np.array([np.where(gene_symbols == tg)[0] for tg in tgs if tg in gene_symbols]).ravel()

        if tf in gene_symbols and len(tg_idxs) > 0:
            # TG-TG correlations
            expr_tgs = expr[:, tg_idxs]
            corr = correlations_list(expr_tgs, expr_tgs)
            tg_tg_corr += [corr.tolist()]

            # TF-TG correlations
            tf_idx = np.argwhere(gene_symbols == tf)[0]
            expr_tf = expr[:, tf_idx]
            corr = pearson_correlation(expr_tf[:, None], expr_tgs).ravel()
            tf_tg_corr += [corr.tolist()]

    # Flatten list
    if flat:
        tf_tg_corr = [c for corr_l in tf_tg_corr for c in corr_l]
        tg_tg_corr = [c for corr_l in tg_tg_corr for c in corr_l]

    return tf_tg_corr, tg_tg_corr


def psi_coefficient(tf_tg_x, tf_tg_z, weights_type='nb_genes'):
    """
    Computes the psi TF-TG correlation coefficient
    :param tf_tg_x: list of TF-TG correlations, returned by compute_tf_tg_corrs with flat=False
    :param tf_tg_z: list of TF-TG correlations, returned by compute_tf_tg_corrs with flat=False
    :param weights_type: for 'nb_genes' the weights for each TF are proportional to the number
                    target genes that it regulates. For 'ones' the weights are all one.
    :return: psi correlation coefficient
    """
    weights_sum = 0
    total_sum = 0
    for cx, cz in zip(tf_tg_x, tf_tg_z):
        weight = 1
        if weights_type == 'nb_genes':
            weight = len(cx)  # nb. of genes regulated by the TF
        weights_sum += weight
        cx = np.array(cx)
        cz = np.array(cz)
        total_sum += weight * cosine_similarity(cx, cz)
    return total_sum / weights_sum


def phi_coefficient(tg_tg_x, tg_tg_z, weights_type='nb_genes'):
    """
    Computes the theta TG-TG correlation coefficient
    :param tf_tg_x: list of TG-TG correlations, returned by compute_tf_tg_corrs with flat=False
    :param tf_tg_z: list of TG-TG correlations, returned by compute_tf_tg_corrs with flat=False
    :param weights_type: for 'nb_genes' the weights for each TF are proportional to the number
                    target genes that it regulates. For 'ones' the weights are all one.
    :return: theta correlation coefficient
    """
    weights_sum = 0
    total_sum = 0
    for cx, cz in zip(tg_tg_x, tg_tg_z):
        if len(cx) > 0:  # In case a TF only regulates one gene, the list will be empty
            weight = 1
            if weights_type == 'nb_genes':
                x = len(cx)  # nb_genes * (nb_genes + 1) = 2*weight
                roots = np.roots([1, 1, -2 * x])
                weight = max(roots)  # nb. of genes regulated by the TF
            weights_sum += weight
            cx = np.array(cx)
            cz = np.array(cz)
            total_sum += weight * cosine_similarity(cx, cz)
    return total_sum / weights_sum


def compute_scores(expr_x, expr_z, gene_symbols):
    """
    Computes evaluation scores
    :param expr_x: real data. Shape=(nb_samples_1, nb_genes)
    :param expr_z: synthetic data. Shape=(nb_samples_2, nb_genes)
    :param gene_symbols: list of gene symbols (the genes dimension is sorted according to this list). Shape=(nb_genes,)
    :return: list of evaluation coefficients (S_dist, S_dend, S_sdcc, S_tftg, S_tgtg)
    """
    # Gamma coefficients
    gamma_dx_dz, gamma_dx_tx, gamma_dz_tz, gamma_tx_tz = gamma_coefficients(expr_x, expr_z)

    # Psi and phi coefficients
    r_tf_tg_corr, r_tg_tg_corr = compute_tf_tg_corrs(expr_x, gene_symbols, flat=False)
    s_tf_tg_corr, s_tg_tg_corr = compute_tf_tg_corrs(expr_z, gene_symbols, flat=False)
    psi_dx_dz = psi_coefficient(r_tf_tg_corr, s_tf_tg_corr)
    phi_dx_dz = phi_coefficient(r_tg_tg_corr, s_tg_tg_corr)

    return [gamma_dx_dz,
            gamma_tx_tz,
            (gamma_dx_tx - gamma_dz_tz) ** 2,
            psi_dx_dz,
            phi_dx_dz]


# ---------------------
# CLUSTERING UTILITIES
# ---------------------

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


# ---------------------
# PLOTTING UTILITIES
# ---------------------

def plot_distribution(data, label, color='royalblue', linestyle='-', ax=None, plot_legend=True,
                      xlabel=None, ylabel=None):
    """
    Plot a distribution
    :param data: data for which the distribution of its flattened values will be plotted
    :param label: label for this distribution
    :param color: line color
    :param linestyle: type of line
    :param ax: matplotlib axes
    :param plot_legend: whether to plot a legend
    :param xlabel: label of the x axis (or None)
    :param ylabel: label of the y axis (or None)
    :return matplotlib axes
    """
    x = np.ravel(data)
    ax = sns.distplot(x,
                      hist=False,
                      kde_kws={'linestyle': linestyle, 'color': color, 'linewidth': 2, 'bw': .15},
                      label=label,
                      ax=ax)
    if plot_legend:
        plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    return ax


def plot_distance_matrix(dist_m, v_min, v_max, symbols, title='Distance matrix'):
    ax = plt.gca()
    im = ax.imshow(dist_m, vmin=v_min, vmax=v_max)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(symbols)))
    ax.set_yticks(np.arange(len(symbols)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(symbols)
    ax.set_yticklabels(symbols)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            text = ax.text(j, i, '{:.2f}'.format(dist_m[i, j]),
                           ha="center", va="center", color="w")
    ax.set_title(title)


def plot_distance_matrices(x, y, symbols, corr_fn=pearson_correlation):
    """
    Plots distance matrices of both datasets x and y.
    :param x: matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
    :param y: matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    :symbols: array of gene symbols. Shape=(nb_genes,)
    :param corr_fn: 2-d correlation function
    """

    dist_x = 1 - np.abs(corr_fn(x, x))
    dist_y = 1 - np.abs(corr_fn(y, y))
    v_min = min(np.min(dist_x), np.min(dist_y))
    v_max = min(np.max(dist_x), np.max(dist_y))

    # fig = plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plot_distance_matrix(dist_x, v_min, v_max, symbols, title='Distance matrix, real')
    plt.subplot(1, 2, 2)
    plot_distance_matrix(dist_y, v_min, v_max, symbols, title='Distance matrix, synthetic')
    # fig.tight_layout()
    return plt.gca()


def plot_individual_distrs(x, y, symbols, nrows=4, xlabel='X', ylabel='Y'):
    """
    Plots individual distributions for each gene
    """
    nb_symbols = len(symbols)
    ncols = 1 + (nb_symbols - 1) // nrows

    # plt.figure(figsize=(18, 12))
    plt.subplots_adjust(left=0, bottom=0, right=None, top=1.3, wspace=None, hspace=None)
    for r in range(nrows):
        for c in range(ncols):
            idx = (nrows - 1) * r + c
            plt.subplot(nrows, ncols, idx + 1)

            plt.title(symbols[idx])
            plot_distribution(x[:, idx], xlabel='', ylabel='', label=xlabel, color='black')
            plot_distribution(y[:, idx], xlabel='', ylabel='', label=ylabel, color='royalblue')

            if idx + 1 == nb_symbols:
                break


def plot_intensities(expr, plot_quantiles=True, dataset_name='E. coli M3D', color='royalblue', ax=None):
    """
    Plot intensities histogram
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param plot_quantiles: whether to plot the 5 and 95% intensity gene quantiles
    :param dataset_name: name of the dataset
    :param color: line color
    :param ax: matplotlib axes
    :return matplotlib axes
    """
    x = np.ravel(expr)
    ax = sns.distplot(x,
                      hist=False,
                      kde_kws={'color': color, 'linewidth': 2, 'bw': .15},
                      label=dataset_name,
                      ax=ax)

    if plot_quantiles:
        stds = np.std(expr, axis=-1)
        idxs = np.argsort(stds)
        cut_point = int(0.05 * len(idxs))

        q95_idxs = idxs[-cut_point]
        x = np.ravel(expr[q95_idxs, :])
        ax = sns.distplot(x,
                          ax=ax,
                          hist=False,
                          kde_kws={'linestyle': ':', 'color': color, 'linewidth': 2, 'bw': .15},
                          label='High variance {}'.format(dataset_name))

        q5_idxs = idxs[:cut_point]
        x = np.ravel(expr[q5_idxs, :])
        sns.distplot(x,
                     ax=ax,
                     hist=False,
                     kde_kws={'linestyle': '--', 'color': color, 'linewidth': 2, 'bw': .15},
                     label='Low variance {}'.format(dataset_name))
    plt.legend()
    plt.xlabel('Absolute levels')
    plt.ylabel('Density')
    return ax


def plot_gene_ranges(expr, dataset_name='E. coli M3D', color='royalblue', ax=None):
    """
    Plot gene ranges histogram
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param dataset_name: name of the dataset
    :param color: line color
    :param ax: matplotlib axes
    :return matplotlib axes
    """
    nb_samples, nb_genes = expr.shape
    sorted_expr = [np.sort(expr[:, gene]) for gene in range(nb_genes)]
    sorted_expr = np.array(sorted_expr)  # Shape=(nb_genes, nb_samples)
    cut_point = int(0.05 * nb_samples)
    diffs = sorted_expr[:, -cut_point] - sorted_expr[:, cut_point]

    ax = sns.distplot(diffs,
                      hist=False,
                      kde_kws={'color': color, 'linewidth': 2, 'bw': .15},
                      label=dataset_name,
                      ax=ax)

    plt.xlabel('Gene ranges')
    plt.ylabel('Density')

    return ax


def plot_difference_histogram(interest_distr, background_distr, xlabel, left_lim=-1, right_lim=1,
                              dataset_name='E. coli M3D', color='royalblue', ax=None):
    """
    Plots a difference between a distribution of interest and a background distribution.
    Approximates these distributions with Kernel Density Estimation using a Gaussian kernel
    :param interest_distr: list containing the values of the distribution of interest.
    :param background_distr: list containing the values of the background distribution.
    :param xlabel: label on the x axis
    :param right_lim: histogram left limit
    :param left_lim: histogram right limit
    :param dataset_name: name of the dataset
    :param color: line color
    :param ax: matplotlib axes
    :return matplotlib axes
    """
    # Estimate distributions
    kde_back = scipy.stats.gaussian_kde(background_distr)
    kde_corr = scipy.stats.gaussian_kde(interest_distr)

    # Plot difference histogram
    grid = np.linspace(left_lim, right_lim, 1000)
    # plt.plot(grid, kde_back(grid), label="kde A")
    # plt.plot(grid, kde_corr(grid), label="kde B")
    ax = plt.plot(grid, kde_corr(grid) - kde_back(grid),
                  color,
                  label=dataset_name,
                  linewidth=2)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Density difference')
    return ax


def find_chip_rates(expr, gene_symbols, tf_tg=None):
    """
    Plots the TF activity histogram. It is computed according to the Wilcoxon's non parametric rank-sum method, which tests
    whether TF targets exhibit significant rank differences in comparison with other non-target genes. The obtained
    p-values are corrected via Benjamini-Hochberg's procedure to account for multiple testing.
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param gene_symbols: list of gene_symbols. Shape=(nb_genes,)
    :param tf_tg: dict with TF symbol as key and list of TGs' symbols as value
    :return np.array of chip rates, and weights (for each TF, number of TGs it regulates)
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
    weights = []
    for tf, tgs in tf_tg.items():
        tg_idxs = np.array([np.where(gene_symbols == tg)[0] for tg in tgs if tg in gene_symbols]).ravel()

        if tf in gene_symbols and len(tg_idxs) > 0:
            # Add weight
            weights.append(len(tg_idxs))

            # Find expressions of TG regulated by TF
            expr_tgs = expr_norm[:, tg_idxs]

            # Find expressions of other genes
            non_tg_idxs = list(set(range(nb_genes)) - set(tg_idxs.tolist()))
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

    return np.array(active_tfs), np.array(weights)


def plot_tf_activity_histogram(expr, gene_symbols, xlabel='Fraction of chips. TF activity', color='royalblue',
                               tf_tg=None):
    """
    Plots the TF activity histogram. It is computed according to the Wilcoxon's non parametric rank-sum method, which tests
    whether TF targets exhibit significant rank differences in comparison with other non-target genes. The obtained
    p-values are corrected via Benjamini-Hochberg's procedure to account for multiple testing.
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param gene_symbols: list of gene_symbols. Shape=(nb_genes,)
    :param tf_tg: dict with TF symbol as key and list of TGs' symbols as value
    :param xlabel: label on the x axis
    :param color: histogram color
    :return matplotlib axes
    """

    # Plot histogram
    values, _ = find_chip_rates(expr, gene_symbols, tf_tg)
    bins = np.logspace(-10, 1, 20, base=2)
    bins[0] = 0
    ax = plt.gca()
    plt.hist(values, bins=bins, color=color)
    ax.set_xscale('log', basex=2)
    ax.set_xlim(2 ** -10, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    return ax


def plot_corr_ellipses(data, ax=None, **kwargs):
    # https://stackoverflow.com/questions/34556180/how-can-i-plot-a-correlation-matrix-as-a-set-of-ellipses-similar-to-the-r-open
    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # set the relative sizes of the major/minor axes according to the strength of
    # the positive/negative correlation
    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    # if data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec


def plot_dendr(x):
    Z = hierarchical_clustering(x)
    plt.figure(figsize=(8, 4))
    plt.title('Gen')
    with plt.rc_context({'lines.linewidth': 0.5}):
        dn = dendrogram(Z, count_sort='ascending', labels=gene_list, leaf_rotation=45, link_color_func=lambda _: '#000000')
    ax = plt.gca()
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig('figures/gen_dendr.pdf');
    # plt.axis('off')
    # plt.tight_layout()


def tsne_2d(data, **kwargs):
    """
    Transform data to 2d tSNE representation
    :param data: expression data. Shape=(dim1, dim2)
    :param kwargs: tSNE kwargs
    :return:
    """
    print('... performing tSNE')
    tsne = TSNE(n_components=2, **kwargs)
    return tsne.fit_transform(data)


def plot_tsne_2d(data, labels, **kwargs):
    """
    Plots tSNE for the provided data, coloring the labels
    :param data: expression data. Shape=(dim1, dim2)
    :param labels: color labels. Shape=(dim1,)
    :param kwargs: tSNE kwargs
    :return: matplotlib axes
    """
    dim1, dim2 = data.shape

    # Prepare label dict and color map
    label_set = set(labels)
    label_dict = {k: v for k, v in enumerate(label_set)}

    # Perform tSNE
    if dim2 == 2:
        # print('plot_tsne_2d: Not performing tSNE. Shape of second dimension is 2')
        data_2d = data
    elif dim2 > 2:
        data_2d = tsne_2d(data, **kwargs)
    else:
        raise ValueError('Shape of second dimension is <2: {}'.format(dim2))

    # Plot scatterplot
    for k, v in label_dict.items():
        plt.scatter(data_2d[labels == v, 0], data_2d[labels == v, 1],
                    label=v)
    plt.legend()
    return plt.gca()


def scatter_2d(data_2d, labels, colors=None, **kwargs):
    """
    Scatterplot for the provided data, coloring the labels
    :param data: expression data. Shape=(dim1, dim2)
    :param labels: color labels. Shape=(dim1,)
    :param kwargs: tSNE kwargs
    :return: matplotlib axes
    """
    # Prepare label dict and color map
    label_set = list(set(labels))[::-1]
    label_dict = {k: v for k, v in enumerate(label_set)}

    # Plot scatterplot
    i = 0
    for k, v in label_dict.items():
        c = None
        if colors is not None:
            c = colors[i]
        plt.scatter(data_2d[labels == v, 0], data_2d[labels == v, 1],
                    label=v, color=c, **kwargs)
        i += 1
    lgnd = plt.legend(markerscale=3)
    return plt.gca()


def scatter_2d_cancer(data_2d, labels, cancer, colors=None, **kwargs):
    # Prepare label dict and color map
    label_set = list(set(labels))[::-1]
    label_dict = {k: v for k, v in enumerate(label_set)}

    # Plot scatterplot
    i = 0
    for k, v in label_dict.items():
        c = None
        if colors is not None:
            c = colors[i]

        idxs = np.logical_and(labels == v, cancer == 'normal')
        plt.scatter(data_2d[idxs, 0], data_2d[idxs, 1],
                    label=v, color=c, marker='o', s=7, **kwargs)
        idxs = np.logical_and(labels == v, cancer == 'cancer')
        plt.scatter(data_2d[idxs, 0], data_2d[idxs, 1], color=c, marker='+', **kwargs)
        i += 1
    lgnd = plt.legend(markerscale=3)
    return plt.gca()


if __name__ == '__main__':
    pass
