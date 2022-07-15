from pathlib import Path
import json
import numpy as np
import tifffile as tif

from spotipy.spotipy.utils import points_to_prob
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
        image_path = str(path_projections / f"{fov}_max.tif")
        image = tif.imread(image_path)
        images.append(image[chid, :, :])

        foci_path = str(path_centrioles / f"{fov}_max_C{chid}.txt")
        foci = np.loadtxt(foci_path, dtype=int, delimiter=',')
        foci_mask = points_to_prob(foci, shape=fov_shape, sigma=1)
        positions.append(foci_mask)

    images = np.stack(images)
    positions = np.stack(positions)

    return images, positions


if __name__ == '__main__':
    # Load the data...
    path_dataset = Path('/Users/buergy/Dropbox/epfl/datasets/RPE1wt_CEP63+CETN2+PCNT_1')

    fov_shape = (2048, 2048)

    train_x, train_y = load_pairs(path_dataset, split='train')
    test_x, test_y = load_pairs(path_dataset, split='test')

    # Load the config
    with open('../../../models/leo3_multiscale_True_mae_aug_1_sigma_1.5_split_2_batch_2_n_300/config.json',
              'r') as config:
        config_dict = json.load(config)
    config = Config(**config_dict)

    # Create a model instance
    model = SpotNet(config, name=None, basedir='/Users/buergy/Dropbox/epfl/projects/centrack/models/test_model')

    # Train loop
    model.train(train_x, train_y, validation_data=(test_x, test_y))
