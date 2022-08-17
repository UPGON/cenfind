import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from centrack.data.base import Dataset, Projection, generate_vignette


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('nuclei_index', type=int)
    args = parser.parse_args()

    dataset = Dataset(args.path)
    dataset.vignettes.mkdir(exist_ok=True)

    train_files = dataset.splits_for('train')
    test_files = dataset.splits_for('test')
    all_files = train_files + test_files

    for fov_name, channel_id in tqdm(all_files):
        channel_id = int(channel_id)
        projection = Projection(dataset, fov_name)
        vignette = generate_vignette(projection, channel_id, args.nuclei_index)

        dst = str(dataset.vignettes / f'{fov_name}_max_C{channel_id}.png')
        cv2.imwrite(dst, vignette)


if __name__ == '__main__':
    cli()
