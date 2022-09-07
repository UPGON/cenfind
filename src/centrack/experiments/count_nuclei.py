import argparse
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

from centrack.data.base import Dataset, Projection, Channel
from centrack.experiments.constants import datasets, PREFIX_REMOTE

config = dotenv_values('.env')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=str, default=None)
    args = parser.parse_args()

    statistics_path = Path(args.destination)
    statistics_path.touch()

    pad_lower = int(.1 * 2048)
    pad_upper = 2048 - pad_lower
    records = []
    for dataset in datasets:
        dataset = Dataset(PREFIX_REMOTE / dataset)
        for field in dataset.fields('_max.tif'):
            projection = Projection(dataset, field)

            channel = Channel(projection, 0)
            annot_nuclei = channel.mask(0)
            centres, contours = channel.extract_nuclei(annotation=annot_nuclei)
            for centre in centres:
                at_edge = True
                if all([pad_lower < c < pad_upper for c in centre.centre]):
                    at_edge = False
                records.append({'dataset': dataset.file_name,
                                'field': field.name,
                                'centre': centre.centre,
                                'at_edge': at_edge})
    df = pd.DataFrame(records)

    def frac(x):
        return x.sum() / len(x)

    summary = df.groupby(['dataset'])['at_edge'].agg(['count', sum, frac])
    summary.to_csv(statistics_path)


if __name__ == '__main__':
    main()
