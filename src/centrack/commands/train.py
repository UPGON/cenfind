from pathlib import Path
import json
import numpy as np
import tifffile as tif

from spotipy.spotipy.utils import points_to_prob, normalize_fast2d
from spotipy.spotipy.model import SpotNet, Config


def load_pairs(path, split='train'):
    """
    Load two arrays, the images and the foci masks
    path: the path to the dataset
    split: either train or test
    """
    path_projections = path / 'projections'
    path_centrioles = path / 'annotations/centrioles'

    with open(path / f'{split}_channels.txt', 'r') as fh:
        fovs = [i.strip().split(',') for i in fh.readlines()]

    images = []
    positions = []
    for fov, chid in fovs:
        chid = int(chid)
        image_path = str(path_projections / f"{fov}_max.tif")
        image = tif.imread(image_path)
        image = image[chid, :, :]
        image = normalize_fast2d(image)
        images.append(image)

        foci_path = str(path_centrioles / f"{fov}_max_C{chid}.txt")
        foci = np.loadtxt(foci_path, dtype=int, delimiter=',')  # in format x, y; origin at top left

        foci_mask = points_to_prob(foci, shape=image.shape, sigma=1)
        positions.append(foci_mask)

    return np.stack(images), np.stack(positions)


if __name__ == '__main__':
    path_dataset = Path('/Users/buergy/Dropbox/epfl/datasets/RPE1wt_CEP63+CETN2+PCNT_1')
    train_x, train_y = load_pairs(path_dataset, split='train')
    test_x, test_y = load_pairs(path_dataset, split='test')

    with open('../../../models/config.json',
              'r') as config:
        config_dict = json.load(config)
    config = Config(**config_dict)

    model = SpotNet(config, name=None, basedir='/Users/buergy/Dropbox/epfl/projects/centrack/models/test_model')
    model.train(train_x, train_y, validation_data=(test_x, test_y))
