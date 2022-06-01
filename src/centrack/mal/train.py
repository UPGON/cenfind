import numpy as np
from spotipy.spotipy.utils import points_to_prob
import torch
import json
from spotipy.spotipy.model import Config, SpotNet

if __name__ == '__main__':
    P = np.random.randint(10, 128 - 10, (5, 30, 2))
    Y = np.stack(tuple(points_to_prob(p, shape=(128, 128), sigma=1) for p in P))
    X = Y + .3 * np.random.normal(0, 1, Y.shape)
    with open('../../../models/leo3_multiscale_True_mae_aug_1_sigma_1.5_split_2_batch_2_n_300/config.json',
              'r') as config:
        config_dict = json.load(config)

    config = Config(**config_dict)

    model = SpotNet(config, name=None, basedir=None)
    #
    model.train(X, Y, validation_data=[X, Y], steps_per_epoch=100)
