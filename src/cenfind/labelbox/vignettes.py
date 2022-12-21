import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.outline import create_vignette


def get_args():
    parser = argparse.ArgumentParser(description="VIGNETTE: create png version of channel+nuclei for annotation tool")
    parser.add_argument('path', type=Path, help="the path to the dataset")
    parser.add_argument('nuclei_index', type=int, help="the index of the nuclei (often, 0 or 3, first or last)")
    parser.add_argument('centrioles_index', type=int, nargs='+', help="the index of the channels")
    parser.add_argument('--projection_suffix',
                        type=str,
                        default='_max',
                        help='the suffix indicating projection, e.g., `_max` or `_Projected`')
    return parser.parse_args()


def main():
    args = get_args()

    dataset = Dataset(args.path, projection_suffix=args.projection_suffix)

    pbar = tqdm(dataset.fields)
    for field in pbar:
        for channel_id in args.centrioles_index:
            pbar.set_description(f"{field.name}: {channel_id}")
            vignette = create_vignette(field, channel_id, args.nuclei_index)
            dst = dataset.vignettes / f'{field.name}{args.projection_suffix}_C{channel_id}.png'
            cv2.imwrite(str(dst), vignette)


if __name__ == '__main__':
    main()
