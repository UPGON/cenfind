import argparse
import logging
from pathlib import Path

import tensorflow as tf
import pandas as pd
from stardist.models import StarDist2D
from tqdm import tqdm

from centrack.core.data import Dataset, Field
from centrack.core.measure import get_model
from centrack.core.measure import score_summary
from centrack.core.measure import score_fov

tf.get_logger().setLevel(logging.ERROR)


def get_args():
    parser = argparse.ArgumentParser(
        description='CENTRACK: Automatic centriole scoring')

    parser.add_argument('path',
                        type=Path,
                        help='path to the ds')

    parser.add_argument('model',
                        type=Path,
                        help='absolute path to the model folder')

    parser.add_argument('channel_nuclei',
                        type=int,
                        help='channel id for nuclei segmentation, e.g., 0 or 3')

    parser.add_argument('channels',
                        nargs='+',
                        type=int,
                        help='channels to analyse, e.g., 1 2 3')

    args = parser.parse_args()

    if args.channel_nuclei in set(args.channels):
        raise ValueError('Nuclei channel cannot present in channels')

    if not args.model.exists():
        raise FileNotFoundError(f"{args.model} does not exist")

    return args


def main():
    args = get_args()

    dataset = Dataset(args.path)
    model_spotnet = get_model(args.model)
    model_stardist = StarDist2D.from_pretrained('2D_versatile_fluo')

    scores = []
    pbar = tqdm(dataset.fields())
    for field in pbar:
        pbar.set_description(f"{field}")
        field = Field(field, dataset)
        for ch in args.channels:
            score = score_fov(field=field,
                              nuclei_channel=args.channel_nuclei,
                              channel=ch,
                              model_foci=model_spotnet,
                              model_nuclei=model_stardist)
            scores.append(score)
    flattened = [leaf for tree in scores for leaf in tree]
    scores_df = pd.DataFrame(flattened)
    binned = score_summary(scores_df)
    binned.to_csv(dataset.statistics / f'statistics.csv')


if __name__ == '__main__':
    main()
