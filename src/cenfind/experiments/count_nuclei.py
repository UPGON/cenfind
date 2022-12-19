import cv2
import pandas as pd
from dotenv import dotenv_values
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.measure import flag, full_in_field, extract_nuclei
from cenfind.core.outline import create_vignette
from cenfind.experiments.constants import datasets, PREFIX_REMOTE

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
            mask = field.mask(0)
            centres, contours = extract_nuclei(field, 0, annotation=mask, factor=256)
            vignette = create_vignette(field, 1, 0)
            for centre, contour in zip(centres, contours):
                is_full = full_in_field(centre.centre, mask, .05)
                records.append({'dataset': dataset.path.name,
                                'field': field.name,
                                'centre': centre.centre,
                                'is_full': is_full})
                contour.draw(vignette, color=flag(is_full))

            cv2.imwrite(f'out/checks/{field.name}.png', vignette)
    df = pd.DataFrame(records)

    summary = df.groupby(['dataset'])['is_full'].agg(['count', sum])
    summary = summary.reset_index()
    summary.columns = ['Dataset', 'Nuclei detected', 'Nuclei full']
    summary.to_csv('out/nuclei_counts.csv')


if __name__ == '__main__':
    main()
