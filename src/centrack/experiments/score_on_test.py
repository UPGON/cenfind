import argparse
from pathlib import Path

import pandas as pd

from centrack.core.data import Dataset, Field, Projection, Channel
from centrack.core.data import get_model
from centrack.core.measure import assign
from centrack.experiments.constants import datasets, PREFIX_REMOTE
from stardist.models import StarDist2D

stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    args = parser.parse_args()

    path_model = args.model
    model = get_model(path_model)
    scored = []
    for dataset_name in datasets:
        path_dataset = PREFIX_REMOTE / dataset_name
        ds = Dataset(path_dataset)
        ds.visualisation.mkdir(exist_ok=True)
        test_files = ds.splits_for('test')

        for fov_name, channel_id in test_files:
            channel_id = int(channel_id)
            projection = Projection(ds, Field(fov_name))
            channel_foci = Channel(projection, channel_id)
            channel_nuclei = Channel(projection, 0)

            centrioles_preds = channel_foci.detect_centrioles(model)
            nuclei_preds_centre, nuclei_preds = channel_nuclei.extract_nuclei(stardist_model)
            assigned = assign(foci=centrioles_preds, nuclei=nuclei_preds, vicinity=-50)

            for pair in assigned:
                n, foci = pair
                scored.append({'fov': projection.name,
                               'channel': channel_id,
                               'nucleus': n.centre.position,
                               'score': len(foci),
                               })

    scores = pd.DataFrame(scored)
    scores.to_csv('out/scores_test.csv')


if __name__ == '__main__':
    main()
