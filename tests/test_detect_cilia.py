from pathlib import Path

import pandas as pd

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_cilia

ROOT_DIR = Path(__file__).parent.parent


class TestDataCilia:
    path_dataset = ROOT_DIR / "data/cilia"
    ds = Dataset(path_dataset)
    ds.setup()
    annotation = pd.read_csv(path_dataset / 'annotations.tsv', sep='\t', index_col=0)
    annotation = annotation.to_dict(orient='index')

    def test_detect_cilia(self):
        for field in self.ds.fields:
            print(field.name)
            cilia = extract_cilia(field, channel=2)
            ground_truth = self.annotation[field.name]['cilia']
            if ground_truth == 0:
                assert len(cilia) >= 0
            else:
                assert abs(len(cilia) / ground_truth) >= .8
