from pathlib import Path

import cv2
import pandas as pd
from dotenv import dotenv_values
from tqdm import tqdm

from cenfind.constants import datasets
from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_nuclei
from cenfind.core.measure import flag
from cenfind.core.visualisation import draw_contour, create_vignette

PREFIX_REMOTE = Path("/data1/centrioles/canonical")
config = dotenv_values('.env')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def main():
    records = []
    for dataset in datasets:
        dataset = Dataset(PREFIX_REMOTE / dataset)
        for field in tqdm(dataset.fields):
            nuclei = extract_nuclei(field, 0)
            vignette = create_vignette(field, 1, 0)
            for nucleus in nuclei:
                centre = nucleus.centre
                is_full = nucleus.full_in_field
                records.append({'dataset': dataset.path.name,
                                'field': field.name,
                                'centre': centre,
                                'is_full': is_full})
                draw_contour(vignette, nucleus, color=flag(is_full))

            cv2.imwrite(f'out/checks/{field.name}.png', vignette)
    df = pd.DataFrame(records)

    summary = df.groupby(['dataset'])['is_full'].agg(['count', sum])
    summary = summary.reset_index()
    summary.columns = ['Dataset', 'Nuclei detected', 'Nuclei full']
    summary.to_csv('out/nuclei_counts.csv')


if __name__ == '__main__':
    main()
