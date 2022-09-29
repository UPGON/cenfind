import argparse
import logging
from pathlib import Path

import pandas as pd
import tensorflow as tf
from stardist.models import StarDist2D
from tqdm import tqdm

from cenfind.core.data import Dataset, Field
from cenfind.core.measure import field_score
from cenfind.core.measure import field_score_frequency

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

    parser.add_argument('projection_suffix',
                        type=str,
                        default='max',
                        help='the suffix indicating projection, e.g., `max` or `Projected`')
    args = parser.parse_args()

    if args.channel_nuclei in set(args.channels):
        raise ValueError('Nuclei channel cannot present in channels')

    if not args.model.exists():
        raise FileNotFoundError(f"{args.model} does not exist")

    return args


def main():
    args = get_args()

    dataset = Dataset(args.path, projection_suffix=args.projection_suffix)
    model_stardist = StarDist2D.from_pretrained('2D_versatile_fluo')

    scores = []
    pbar = tqdm(dataset.pairs())
    for field_name, _ in pbar:
        pbar.set_description(f"{field_name}")
        field = Field(field_name, dataset)
        for ch in args.channels:
            score = field_score(field=field, model_nuclei=model_stardist, model_foci=args.model,
                                nuclei_channel=args.channel_nuclei, channel=ch)
            scores.append(score)

    flattened = [leaf for tree in scores for leaf in tree]

    scores_df = pd.DataFrame(flattened)
    scores_df.to_csv(dataset.statistics / f'scores_df.tsv', sep='\t', index=False)

    binned = field_score_frequency(scores_df)
    binned.to_csv(dataset.statistics / f'statistics.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
