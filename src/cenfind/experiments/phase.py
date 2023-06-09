"""
Collect all images in one dataset folder
Set up the code for learning
Training loop from notebook

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_nuclei

np.random.seed(0)


def main():
    data_path = Path('out/nuclei.tsv')
    if not data_path.exists():
        ds = Dataset('/Users/buergy/Dropbox/epfl/datasets/RPE1wt_CEP152+GTU88+PCNT_1', projection_suffix='_max')
        ds.setup()
        ds.write_fields()

        data = []
        for field in ds.fields:
            nuclei = extract_nuclei(field, channel=0, factor=256)
            for n in nuclei:
                data.append((n.intensity, n.area))

        df = pd.DataFrame.from_records(data)
        df.columns = ['intensity', 'surface_area']

        df.to_csv(data_path)
    else:
        print('Skipping segmentation')
        df = pd.read_csv(data_path)

    plt.plot(df['surface_area'] ** (1 / 2), np.log2(df['intensity']), 'b.', alpha=.3)
    plt.xlim(0)
    plt.show()


if __name__ == '__main__':
    main()
