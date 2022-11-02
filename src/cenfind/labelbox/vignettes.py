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
    parser.add_argument('projection_suffix',
                        type=str,
                        default='max',
                        help='the suffix indicating projection, e.g., `max` or `Projected`')
    parser.add_argument('--channel_index', type=int, help="the index of the channel (often, 1)")
    return parser.parse_args()


def main():
    args = get_args()

    dataset = Dataset(args.path, projection_suffix=args.projection_suffix)

    pbar = tqdm(dataset.pairs(channel_id=args.channel_index))
    for field, channel_id in pbar:
        pbar.set_description(f"{field.name}: {channel_id}")
        vignette = create_vignette(field, channel_id, args.nuclei_index)
        dst = dataset.vignettes / f'{field.name}_max_C{channel_id}.png'
        cv2.imwrite(dst, vignette)


if __name__ == '__main__':
    main()
