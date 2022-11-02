import logging
import os

import numpy as np
import pandas as pd
from spotipy.utils import points_matching
from stardist.models import StarDist2D
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.experiments.constants import datasets, PREFIX_REMOTE

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s: %(message)s")


def main():
    model_stardist = StarDist2D.from_pretrained('2D_versatile_fluo')
    accuracies = []
    preds = []
    actual = []
    for dataset in datasets:
        path_dataset = PREFIX_REMOTE / dataset
        dataset = Dataset(path_dataset)
        for field, channel in tqdm(dataset.pairs('_max.tif')):
            nuclei = field.channel(0)
            nuclei_mask = field.mask(0)
            centres_preds, nuclei_preds = nuclei.extract_nuclei(model_stardist)
            centres_actual, nuclei_actual = nuclei.extract_nuclei(model_stardist, annotation=nuclei_mask)
            logging.info("Found %d nuclei instead of %d" % (len(centres_preds), len(centres_actual)))
            preds = preds + [p.position for p in centres_preds]
            actual = actual + [p.position for p in centres_actual]
        res = points_matching(preds, actual, cutoff_distance=50)
        accuracies.append({'DATASET': dataset.path.name,
                           'F1': np.round(res.f1, 3)})
    acc_df = pd.DataFrame(accuracies)
    acc_df.to_csv('out/nuclei_acc.csv')


if __name__ == '__main__':
    main()
