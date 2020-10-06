import pandas as pd
import os
import re

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


if __name__ == '__main__':
    # expr_df, info_df = rnaseqdb_join_datasets()
    # rnaseqdb_save(expr_df, info_df)
    expr_df, info_df = rnaseqdb_load()
    print(expr_df.head())