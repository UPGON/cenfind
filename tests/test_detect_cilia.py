from pathlib import Path

import pandas as pd

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_cilia

ROOT_DIR = Path(__file__).parent.parent


class TestDataCilia:
    path_dataset = ROOT_DIR / "data/cilia"
    ds = Dataset(path_dataset)
    annotation = pd.read_csv(path_dataset / 'annotations.tsv', sep='\t', index_col=0)
    annotation = annotation.to_dict(orient='index')
    path_cilia = (ds.visualisation / "cilia")
    path_cilia.mkdir(exist_ok=True)

    def test_detect_cilia(self):
        for field in self.ds.fields:
            print(field.name)
            cilia = extract_cilia(field, channel=2, dst=self.path_cilia)
            groundtruth = self.annotation[field.name]['cilia']
            if groundtruth == 0:
                assert len(cilia) >= 0
            else:
                assert abs(len(cilia) / groundtruth) >= .8
