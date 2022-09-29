import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from cenfind.core.data import Dataset, Field
from cenfind.core.outline import create_vignette


def main():
    parser = argparse.ArgumentParser(description="VIGNETTE: create png version of channel+nuclei for annotation tool")
    parser.add_argument('path', type=Path, help="the path to the dataset")
    parser.add_argument('nuclei_index', type=int, help="the index of the nuclei (often, 0 or 3, first or last)")
    parser.add_argument('projection_suffix',
                        type=str,
                        default='max',
                        help='the suffix indicating projection, e.g., `max` or `Projected`')
    args = parser.parse_args()

    dataset = Dataset(args.path, projection_suffix=args.projection_suffix)
    path_vignettes = Path(dataset.path / 'vignettes')
    path_vignettes.mkdir(exist_ok=True)

    pbar = tqdm(dataset.pairs())
    for fov_name, channel_id in pbar:
        pbar.set_description(f"{fov_name}: {channel_id}")
        field = Field(fov_name, dataset)
        vignette = create_vignette(field, channel_id, args.nuclei_index)

        dst = str(path_vignettes / f'{fov_name}_max_C{channel_id}.png')
        cv2.imwrite(dst, vignette)


if __name__ == '__main__':
    main()
