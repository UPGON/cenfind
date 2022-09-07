import argparse
from pathlib import Path
import cv2
import pandas as pd
from dotenv import dotenv_values
from tqdm import tqdm
from centrack.data.base import Dataset, Projection, Channel, generate_vignette
from centrack.experiments.constants import datasets, PREFIX_REMOTE
from centrack.data.measure import frac, at_edge

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
        for field in tqdm(dataset.fields('_max.tif')):
            projection = Projection(dataset, field)
            channel = Channel(projection, 0)
            annot_nuclei = channel.mask(0)
            centres, contours = channel.extract_nuclei(annotation=annot_nuclei)
            vignette = generate_vignette(projection, 1, 0)
            for centre, contour in zip(centres, contours):
                is_at_edge = at_edge(centre.centre, .05, annot_nuclei)
                color = (0, 0, 255)
                if is_at_edge:
                    color = (0, 255, 0)
                records.append({'dataset': dataset.file_name,
                                'field': field.name,
                                'centre': centre.centre,
                                'is_at_edge': is_at_edge})

                contour.draw(vignette, color=color)

            cv2.imwrite(f'out/checks/{field.name}.png', vignette)
    df = pd.DataFrame(records)

    summary = df.groupby(['dataset'])['is_at_edge'].agg(['count', sum, frac])
    summary.to_csv(statistics_path)


if __name__ == '__main__':
    main()
