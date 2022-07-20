from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from centrack.commands.score import get_model
from centrack.utils.constants import datasets, PREFIX
from centrack.utils.visualisation import show_images
from spotipy.spotipy.utils import normalize_fast2d, points_to_label, points_to_prob, points_matching

if __name__ == '__main__':
    model = get_model(
        '/Users/buergy/Dropbox/epfl/projects/centrack/models/leo3_multiscale_True_mae_aug_1_sigma_1.5_split_2_batch_2_n_300')
    performances = []
    for dataset in datasets:
        path_visualisation = PREFIX / dataset / 'visualisation'
        path_visualisation.mkdir(exist_ok=True)
        with open(PREFIX / dataset / 'test_channels.txt') as f:
            test_files = f.read().splitlines()
        for fov in test_files:
            fov_name, chid = fov.split(',')
            chid = int(chid)
            projection = PREFIX / dataset / 'projections' / f"{fov_name}_max.tif"
            data = tifffile.imread(projection)
            channel = data[chid, :, :]
            inp = normalize_fast2d(channel)
            path_annotation = PREFIX / dataset / 'annotations' / 'centrioles' / f"{fov_name}_max_C{chid}.txt"
            try:
                annotation = np.loadtxt(path_annotation, dtype=int, delimiter=',')
            except FileNotFoundError:
                print(f'annotation not found for {path_annotation}')
                continue
            # annotation = annotation[:, [1, 0]]
            mask_actual = points_to_prob(annotation, shape=inp.shape, sigma=1)
            mask_preds, points_preds = model.predict(inp, prob_thresh=.5)
            res = points_matching(annotation[:, [1, 0]], points_preds)
            performances.append({'dataset': dataset,
                                 'fov': fov_name,
                                 'channel': chid,
                                 'foci_actual_n': len(annotation),
                                 'foci_preds_n': len(points_preds),
                                 'f1': res.f1,
                                 'precision': res.precision,
                                 'recall': res.recall})
            # Visualisation
            fig = show_images(inp, mask_actual)
            fig.savefig(path_visualisation / f"{fov_name}_max_C{chid}.png")

    performances_df = pd.DataFrame(performances).round(3)
    performances_df = performances_df.set_index('fov')
    performances_df.to_csv('/Users/buergy/Dropbox/epfl/projects/centrack/out/performances_base_model.csv')
