from data_pipeline import reg_network, gene_symbols
import numpy as np

DEFAULT_ROOT_GENE = 'CRP'
DEFAULT_EVIDENCE = 'Weak'
DEFAULT_DEPTH = np.inf
OUT_FILE = '../../syntren1.2release/data/sourceNetworks/EColi_n{}_r{}_e{}_d{}.sif'


def dump_network(root_gene=DEFAULT_ROOT_GENE, minimum_evidence=DEFAULT_EVIDENCE, depth=DEFAULT_DEPTH, break_loops=True):
    gs = get_gene_symbols()
    nodes, edges = reg_network(gs, root_gene, minimum_evidence, depth, break_loops)
    nb_nodes = len(nodes)
    file_name = OUT_FILE.format(nb_nodes, root_gene, minimum_evidence, depth)
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


if __name__ == '__main__':
    dump_network(minimum_evidence='Weak')
