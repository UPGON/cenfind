import argparse
import contextlib
import logging
import os
from pathlib import Path

import pandas as pd

from stardist.models import StarDist2D
from centrack.data.base import Dataset, Projection, Channel
from centrack.data.base import get_model
from centrack.data.measure import foci_prediction_prepare, assign, score_summary

logger_score = logging.getLogger()
logger_score.setLevel(logging.INFO)


def main():
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
                        help='channel id for nuclei segmentation, e.g., 0 or 3, default 0')

    args = parser.parse_args()

    path_dataset = Path(args.path)
    dataset = Dataset(path_dataset)

    path_predictions = path_dataset / 'predictions'
    path_visualisation = path_dataset / 'visualisations'
    path_statistics = path_dataset / 'statistics'

    path_predictions.mkdir(exist_ok=True)
    path_visualisation.mkdir(exist_ok=True)
    path_statistics.mkdir(exist_ok=True)

    if not args.model.exists():
        raise FileNotFoundError(f"{args.model} does not exist")
    model_spotnet = get_model(args.model)

    nuclei_channel = args.channel_nuclei
    if not dataset.projections.exists():
        raise FileExistsError(
            'Projection folder does not exist. Have you run `squash`?')

    scored = []

    for projection in dataset.fields('_max.tif'):
        logger_score.info('Loading %s', projection.name)

        projection = Projection(dataset, projection)
        channels, height, width = projection.data.shape
        channels = list(range(channels))
        channels.pop(nuclei_channel)

        nuclei = Channel(projection, nuclei_channel)
        model_stardist = StarDist2D.from_pretrained('2D_versatile_fluo')

        # This skips the print calls in spotipy
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            centres, nuclei = nuclei.extract_nuclei(model_stardist)

        for channel in channels:
            centrioles = Channel(projection, channel)

            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                foci = centrioles.detect_centrioles(model=model_spotnet)

            if foci:
                foci_df = foci_prediction_prepare(foci, channel)
                foci_df.to_csv(path_predictions / f"{projection.name}_foci_{channel}_preds.csv")

                logger_score.info('Detection in channel %s: %s nuclei, %s foci',
                                  channel, len(nuclei), len(foci))
            else:
                logger_score.warning(
                    'No object were detected in channel %s: skipping...', channel)

            assigned = assign(foci=foci, nuclei=nuclei, vicinity=-50)

            for pair in assigned:
                n, foci = pair
                scored.append({'fov': projection.name,
                               'channel': channel,
                               'nucleus': n.centre.position,
                               'score': len(foci),
                               })

    scores = pd.DataFrame(scored)
    binned = score_summary(scores)
    dst_statistics = str(path_statistics / f'statistics.csv')
    binned.to_csv(dst_statistics)
    logger_score.info('Saving statistics to %s' % path_statistics)
    logger_score.info('Analysis done.')


if __name__ == '__main__':
    main()
