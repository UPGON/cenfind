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

pattern_dataset = re.compile(
    "(?P<cell_type>[a-zA-Z0-9.-]+)(_(?P<treatment>\w+))?_(?P<markers>[\w+]+)_(?P<replicate>\d)")
