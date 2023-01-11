from numpy.random import seed

seed(1)

import tensorflow as tf

tf.random.set_seed(2)
import argparse
import logging
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile as tf
from stardist.models import StarDist2D
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.measure import field_score, field_score_frequency
from cenfind.core.outline import Centre, create_vignette


def get_args():
    parser = argparse.ArgumentParser(
        description='CENFIND: Automatic centriole scoring')

    parser.add_argument('path',
                        type=Path,
                        help='Path to the dataset')

    parser.add_argument('model',
                        type=Path,
                        help='Absolute path to the model folder')

    parser.add_argument('channel_nuclei',
                        type=int,
                        help='Channel index for nuclei segmentation, e.g., 0 or 3')

    parser.add_argument('channels',
                        nargs='+',
                        type=int,
                        help='Channel indices to analyse, e.g., 1 2 3')

    parser.add_argument('--vicinity',
                        type=int,
                        default=-5,
                        help='Distance threshold in micrometer (default: -5 um)')

    parser.add_argument('--factor',
                        type=int,
                        default=256,
                        help='Factor to use: given a 2048x2048 image, 256 if 63x; 2048 if 20x:')

    parser.add_argument('--projection_suffix',
                        type=str,
                        default='_max',
                        help='Projection suffix (`_max` (default) or `_Projected`')

    parser.add_argument('--cpu',
                        action='store_true',
                        help='Only use the cpu')

    args = parser.parse_args()

    if args.channel_nuclei in set(args.channels):
        raise ValueError('Nuclei channel cannot be in channels')

    if not Path(args.model).exists():
        raise FileNotFoundError(f"{args.model} does not exist")

    return args


def save_foci(foci_list: list[Centre], dst: str, logger=None) -> None:
    if len(foci_list) == 0:
        array = np.array([])
        if logger is not None:
            logger.info('No centriole detected')
        else:
            print('No centriole detected')
    else:
        array = np.asarray(np.stack([c.to_numpy() for c in foci_list]))
        array = array[:, [1, 0]]
    np.savetxt(dst, array, delimiter=',', fmt='%u')


def main():
    args = get_args()
    visualisation = True

    path_logs = args.path / 'logs'
    path_logs.mkdir(exist_ok=True)

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(filename=path_logs / 'score.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    dataset = Dataset(args.path, projection_suffix=args.projection_suffix)

    channels, width, height = dataset.fields[0].projection.shape
    if args.channel_nuclei not in range(channels):
        logger.error("Index for nuclei (%s) out of index range" % args.channel_nuclei)
        sys.exit()

    if not set(args.channels).issubset(set(range(channels))):
        logger.error("Channels (%s) out of channel range %s" % args.channels, set(range(channels)))
        sys.exit()

    model_stardist = StarDist2D.from_pretrained('2D_versatile_fluo')

    scores = []
    pbar = tqdm(dataset.fields)
    for field in pbar:
        logger.info("Processing %s" % field.name)
        pbar.set_description(f"{field.name}")
        for ch in args.channels:
            logger.info("Processing %s / %d" % (field.name, ch))
            try:
                foci, nuclei, assigned, score = field_score(field=field,
                                                            model_nuclei=model_stardist,
                                                            model_foci=args.model,
                                                            nuclei_channel=args.channel_nuclei,
                                                            factor=args.factor,
                                                            vicinity=args.vicinity,
                                                            channel=ch)
                predictions_path = dataset.predictions / 'centrioles' / f"{field.name}{args.projection_suffix}_C{ch}.txt"
                save_foci(foci, predictions_path)
                logger.info("(%s), channel %s: nuclei: %s; foci: %s" % (field.name, ch, len(nuclei), len(foci)))
                pbar.set_postfix({'field': field.name, 'channel': ch, 'nuclei': len(nuclei), 'foci': len(foci)})
                scores.append(score)
                if visualisation:
                    logger.info("Writing visualisations for (%s), channel %s" % (field.name, ch))
                    background = create_vignette(field, marker_index=ch, nuclei_index=args.channel_nuclei)
                    for focus in foci:
                        background = focus.draw(background, annotation=False)
                    for nucleus in nuclei:
                        background = nucleus.draw(background, annotation=False)
                    for n_pos, c_pos in assigned:
                        nuc = Centre(n_pos, label='Nucleus')
                        for sub_c in c_pos:
                            if sub_c:
                                cv2.arrowedLine(background, sub_c.to_cv2(), nuc.to_cv2(), color=(0, 255, 0),
                                                thickness=1)
                    tf.imwrite(dataset.visualisation / f"{field.name}_C{ch}_pred.png", background)
            except ValueError as e:
                logger.warning('%s (%s)' % (e, field.name))
                continue

        logger.info("DONE (%s)" % field.name)

    flattened = [leaf for tree in scores for leaf in tree]
    scores_df = pd.DataFrame(flattened)
    scores_df.to_csv(dataset.statistics / f'scores_df.tsv', sep='\t', index=False)
    logger.info("Writing raw scores to scores_df.tsv")
    binned = field_score_frequency(scores_df)
    binned.to_csv(dataset.statistics / f'statistics.tsv', sep='\t', index=True)
    logger.info("Writing statistics to statistics.tsv")

    logger.info("All fields in (%s) have been processed" % dataset.path.name)


if __name__ == '__main__':
    main()
