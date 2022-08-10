import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from centrack.layout.dataset import DataSet, FieldOfView


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('nuclei_index', type=int)
    args = parser.parse_args()

    dataset = DataSet(args.path)
    dataset.vignettes.mkdir(exist_ok=True)

    train_files = dataset.split_images_channel('train')
    test_files = dataset.split_images_channel('test')
    all_files = train_files + test_files

    for fov_name, channel_id in tqdm(all_files):
        channel_id = int(channel_id)
        fov = FieldOfView(dataset, fov_name)
        vignette = fov.generate_vignette(channel_id,
                                         args.nuclei_index)

        dst = str(dataset.vignettes / f'{fov_name}_max_C{channel_id}.png')
        cv2.imwrite(dst, vignette)


if __name__ == '__main__':
    cli()
