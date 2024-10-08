import logging
from pathlib import Path

import numpy as np
import pandas as pd
from spotipy.utils import points_matching
from stardist.models import StarDist2D
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_nuclei
from cenfind.constants import datasets

PREFIX_REMOTE = Path("/data1/centrioles/canonical")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s: %(message)s")


def main():
    model_stardist = StarDist2D.from_pretrained('2D_versatile_fluo')
    accuracies = []
    preds = []
    actual = []
    for dataset in datasets:
        path_dataset = PREFIX_REMOTE / dataset
        dataset = Dataset(path_dataset)
        for field in tqdm(dataset.fields):
            nuclei_mask = field.mask(0)
            nuclei_preds = extract_nuclei(field=field, channel=0, model=model_stardist )
            nuclei_actual = extract_nuclei(field=field, annotation=nuclei_mask, channel=0)
            centres_preds = [c.centre for c in nuclei_preds]
            centres_actual = [c.centre for c in nuclei_actual]
            logging.info("Found %d nuclei instead of %d" % (len(centres_preds), len(centres_actual)))
            preds = preds + centres_preds
            actual = actual + centres_actual
        res = points_matching(preds, actual, cutoff_distance=50)
        accuracies.append({'DATASET': dataset.path.name,
                           'F1': np.round(res.f1, 3)})
    acc_df = pd.DataFrame(accuracies)
    acc_df.to_csv('out/_nuclei_acc.csv')


if __name__ == '__main__':
    main()
