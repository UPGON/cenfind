import re
from pathlib import Path

PREFIX_LOCAL = Path('/Users/buergy/Dropbox/epfl/datasets')
PREFIX_REMOTE = Path('/data1/centrioles/')

ROOT_DIR = Path('/home/buergy/projects/centrack')
datasets = [
    'RPE1wt_CEP63+CETN2+PCNT_1',
    'RPE1wt_CP110+GTU88+PCNT_2',
    'RPE1wt_CEP152+GTU88+PCNT_1',
    'U2OS_CEP63+SAS6+PCNT_1',
    'RPE1p53_Cnone_CEP63+CETN2+PCNT_1',
    'RPE1p53_PLK4flag_CEP63+CETN2+PCNT_1',
]

protein_names = {
    'CEP63': 'Cep63',
    'CEP152': 'Cep152',
    'GTU88': 'γ-Tubulin',
    'PCNT': 'Pericentrin',
    'CETN2': 'Centrin',
    'SAS6': 'SAS6',
    'CP110': 'Cp110'
}

celltype_names = {
    'RPE1wt': 'RPE-1 WT',
    'RPE1p53': 'RPE-1 p53-/-',
    'U2OS': 'U2-OS'
}

pattern_dataset = re.compile(
    "(?P<cell_type>[a-zA-Z0-9.-]+)(_(?P<treatment>\w+))?_(?P<markers>[\w+]+)_(?P<replicate>\d)")
