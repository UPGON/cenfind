from pathlib import Path
import json
import numpy as np
import tifffile as tf

from spotipy.spotipy.utils import points_to_prob
from spotipy.spotipy.model import Config, SpotNet, SpotNetData


def load_pairs(path, split='train'):
    """
    Load two arrays, the images and the foci masks
    path: the path to the dataset
    split: either train or test
    """
    path_projections = path / 'projections'
    path_centrioles = path / 'annotations/centrioles'
    with open(path / f'{split}.txt', 'r') as fh:
        fovs = [i.strip() for i in fh.readlines()]
    images = []
    positions = []
    for fov in fovs:
        image_path = str(path_projections / f"{fov}_max_C2.tif")
        image = tf.imread(image_path)
        images.append(image)

        foci_path = str(path_centrioles / f"{fov}_max_C2.txt")
        foci = np.loadtxt(foci_path, dtype=int, delimiter=',')
        foci_mask = points_to_prob(foci, shape=fov_shape, sigma=1)
        positions.append(foci_mask)

    images = np.stack(images)
    positions = np.stack(positions)

    return images, positions


if __name__ == '__main__':
    # Load the data...
    path_dataset = Path('/data1/centrioles/rpe/RPE1p53_Cnone_CEP63+CETN2+PCNT_1')

    fov_shape = (2048, 2048)

    train_x, train_y = load_pairs(path_dataset, split='train')
    test_x, test_y = load_pairs(path_dataset, split='test')

    # Load the config
    with open('../../../models/leo3_multiscale_True_mae_aug_1_sigma_1.5_split_2_batch_2_n_300/config.json',
              'r') as config:
        config_dict = json.load(config)
    config = Config(**config_dict)

    # Create a model instance
    model = SpotNet(config, name=None, basedir=None)

    # Train loop
    model.train(train_x, train_y, validation_data=[train_x, train_y], steps_per_epoch=2)
