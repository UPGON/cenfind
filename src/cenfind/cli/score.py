import argparse
from pathlib import Path

import cv2
import pandas as pd
import tifffile as tf
import tensorflow
from stardist.models import StarDist2D
from tqdm import tqdm

from cenfind.core.data import Dataset, Field
from cenfind.core.measure import field_score
from cenfind.core.measure import field_score_frequency
from cenfind.core.outline import create_vignette

## GLOBAL SEED ##
tensorflow.random.set_seed(3)

# tf.get_logger().setLevel(logging.ERROR)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_args():
    parser = argparse.ArgumentParser(
        description='CENFIND: Automatic centriole scoring')

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

    parser.add_argument('--projection_suffix',
                        type=str,
                        default='_max',
                        help='the suffix indicating projection, e.g., `_max` or `_Projected`, if not specified, set to _max')
    args = parser.parse_args()

    if args.channel_nuclei in set(args.channels):
        raise ValueError('Nuclei channel cannot present in channels')

    if not args.model.exists():
        raise FileNotFoundError(f"{args.model} does not exist")

    return args


def main():
    args = get_args()
    visualisation = True

    dataset = Dataset(args.path, projection_suffix=args.projection_suffix)
    model_stardist = StarDist2D.from_pretrained('2D_versatile_fluo')

    scores = []
    pbar = tqdm(dataset.pairs())
    for field, _ in pbar:
        pbar.set_description(f"{field.name}")
        for ch in args.channels:
            foci, nuclei, assigned, score = field_score(field=field, model_nuclei=model_stardist, model_foci=args.model,
                                                        nuclei_channel=args.channel_nuclei, channel=ch)
            pbar.set_postfix({'nuclei': len(nuclei), 'foci': len(foci)})
            scores.append(score)

            if visualisation:
                background = create_vignette(field, marker_index=ch, nuclei_index=0)
                for focus in foci:
                    background = focus.draw(background)
                for nucleus in nuclei:
                    background = nucleus.draw(background)
                for n_pos, c_pos in assigned:
                    for sub_c in c_pos:
                        if sub_c:

                            cv2.arrowedLine(background, sub_c.to_cv2(), n_pos.centre.to_cv2(), color=(0, 255, 0),
                                            thickness=1)
                tf.imwrite(args.path / 'visualisations' / f"{field.name}_C{ch}_pred.png", background)

    flattened = [leaf for tree in scores for leaf in tree]

    scores_df = pd.DataFrame(flattened)
    scores_df.to_csv(dataset.statistics / f'scores_df.tsv', sep='\t', index=False)

    binned = field_score_frequency(scores_df)
    binned.to_csv(dataset.statistics / f'statistics.tsv', sep='\t', index=True)


if __name__ == '__main__':
    main()
