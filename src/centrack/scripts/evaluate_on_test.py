from pathlib import Path

import numpy as np
import tifffile

from centrack.commands.score import get_model
from centrack.utils.constants import datasets, PREFIX
from centrack.utils.visualisation import show_images
from spotipy.spotipy.utils import normalize_fast2d, points_to_label

if __name__ == '__main__':
    model = get_model(
        '/Users/buergy/Dropbox/epfl/projects/centrack/models/leo3_multiscale_True_mae_aug_1_sigma_1.5_split_2_batch_2_n_300')
    for dataset in datasets:
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
            annotation = annotation[:, [1, 0]]
            points, mask = model.predict(inp, prob_thresh=.5)
            res = model.evaluate(inp, mask)
            fig = show_images(inp, points)
            path_visualisation = PREFIX / dataset / 'visualisation'
            path_visualisation.mkdir(exist_ok=True)
            fig.savefig(path_visualisation / f"{fov_name}_max_C{chid}.png")
            print(res)
