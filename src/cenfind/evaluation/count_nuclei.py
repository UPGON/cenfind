import argparse
from pathlib import Path

import cv2
import pandas as pd
from dotenv import dotenv_values
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.measure import full_in_field, extract_nuclei
from cenfind.core.helpers import frac, flag
from cenfind.core.outline import create_vignette
from cenfind.experiments.constants import datasets, PREFIX_REMOTE

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

    records = []
    for dataset in datasets:
        dataset = Dataset(PREFIX_REMOTE / dataset)
        for field, channel in tqdm(dataset.pairs()):
            mask = field.mask(0)
            centres, contours = extract_nuclei(field, 0, annotation=mask)
            vignette = create_vignette(field, 1, 0)
            for centre, contour in zip(centres, contours):
                is_full = full_in_field(centre.centre, .05, mask)
                records.append({'dataset': dataset.path.name,
                                'field': field.name,
                                'centre': centre.centre,
                                'is_full': is_full})
                contour.draw(vignette, color=flag(is_full))

            cv2.imwrite(f'out/checks/{field.name}.png', vignette)
    df = pd.DataFrame(records)

    summary = df.groupby(['dataset'])['is_full'].agg(['count', sum, frac])
    summary.to_csv(statistics_path)


if __name__ == '__main__':
    main()
