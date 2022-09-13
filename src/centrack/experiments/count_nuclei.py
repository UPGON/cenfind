import argparse
from pathlib import Path
import cv2
import pandas as pd
from dotenv import dotenv_values
from tqdm import tqdm
from centrack.core.data import Dataset, Field
from centrack.experiments.constants import datasets, PREFIX_REMOTE
from centrack.core.measure import frac, full_in_field
from centrack.core.outline import create_vignette

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
        for field in tqdm(dataset.fields()):
            projection = Field(field, dataset)
            channel = field.channel(0)
            annot_nuclei = field.mask()

            centres, contours = channel.extract_nuclei(annotation=annot_nuclei)
            vignette = create_vignette(projection, 1, 0)
            for centre, contour in zip(centres, contours):
                is_full = full_in_field(centre.centre, .05, annot_nuclei)
                color = (0, 0, 255)
                if is_full:
                    color = (0, 255, 0)
                records.append({'dataset': dataset.path.name,
                                'field': field.name,
                                'centre': centre.centre,
                                'is_full': is_full})
                contour.draw(vignette, color=color)
            cv2.imwrite(f'out/checks/{field.name}.png', vignette)
    df = pd.DataFrame(records)

    summary = df.groupby(['dataset'])['is_full'].agg(['count', sum, frac])
    summary.to_csv(statistics_path)


if __name__ == '__main__':
    main()
