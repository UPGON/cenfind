import argparse
from pathlib import Path

import pandas as pd
from stardist.models import StarDist2D

from centrack.core.data import Dataset, Field
from centrack.core.measure import field_score, get_model
from centrack.experiments.constants import datasets, PREFIX_REMOTE

stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    model_spotnet = get_model(args.model)
    model_stardist = StarDist2D.from_pretrained('2D_versatile_fluo')

    scored = []
    for dataset_name in datasets:
        dataset = Dataset(PREFIX_REMOTE / dataset_name)
        dataset.visualisation.mkdir(exist_ok=True)
        test_files = dataset.splits_for('test')

        for fov_name, channel in test_files:
            channel = int(channel)
            field = Field(fov_name, dataset)
            score = field_score(field=field, model_nuclei=model_stardist, model_foci=model_spotnet,
                                nuclei_channel=args.channel_nuclei, channel=channel)
            scored.append(score)

    scores = pd.DataFrame(scored)
    scores.to_csv('out/scores_test.csv')


if __name__ == '__main__':
    main()
