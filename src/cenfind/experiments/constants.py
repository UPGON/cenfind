import re
from pathlib import Path

import pandas as pd

PREFIX_LOCAL = Path('/Users/buergy/Dropbox/epfl/datasets')
PREFIX_REMOTE = Path('/data1/centrioles/')

ROOT_DIR = Path('/home/buergy/projects/cenfind')
datasets = [
    'RPE1wt_CEP63+CETN2+PCNT_1',
    'RPE1wt_CP110+GTU88+PCNT_2',
    'RPE1wt_CEP152+GTU88+PCNT_1',
    'U2OS_CEP63+SAS6+PCNT_1',
    'RPE1p53+Cnone_CEP63+CETN2+PCNT_1',
]

protein_positions = {
    'RPE1wt_CEP63+CETN2+PCNT_1': ('DAPI', 'CEP63', 'CETN2', 'PCNT'),
    'RPE1wt_CP110+GTU88+PCNT_2': ('DAPI', 'CP110', 'GTU88', 'PCNT'),
    'RPE1wt_CEP152+GTU88+PCNT_1': ('DAPI', 'CEP152', 'GTU88', 'PCNT'),
    'U2OS_CEP63+SAS6+PCNT_1': ('DAPI', 'CEP63', 'SAS6', 'PCNT'),
    'RPE1p53+Cnone_CEP63+CETN2+PCNT_1': ('DAPI', 'CEP63', 'CETN2', 'PCNT'),
}

protein_names = {
    'CEP63': 'Cep63',
    'CEP152': 'Cep152',
    'GTU88': 'γ-Tubulin',
    'PCNT': 'Pericentrin',
    'CETN2': 'Centrin',
    'SAS6': 'SAS6',
    'CP110': 'Cp110',
    'AcTub': 'Ac. Tub',
    'Plk4': 'Plk4',
    'STIL': 'STIL',
}

celltype_names = {
    'RPE1wt': 'RPE-1 WT',
    'RPE1p53': 'RPE-1 p53-/-',
    'U2OS': 'U2-OS',
    'U2OSwt': 'U2-OS',
}

pattern_dataset = re.compile(
    "(?P<cell_type>[a-zA-Z0-9.-]+)([+_](?P<treatment>\w+))?_(?P<markers>[\w+]+)_(?P<replicate>\d)")

if __name__ == '__main__':
    datasets = pd.DataFrame(datasets)
    cell_types = pd.DataFrame().from_dict(celltype_names, orient='index')
    protein_positions = pd.DataFrame().from_dict(protein_positions, orient='index')
    proteins_names = pd.DataFrame.from_dict(protein_names, orient='index')

    datasets.to_csv('data/datasets.tsv', sep='\t')
    protein_positions.to_csv('data/protein_positions.tsv', sep='\t')
    proteins_names.to_csv('data/proteins_names.tsv', sep='\t')
    cell_types.to_csv('data/cell_types.tsv', sep='\t')
