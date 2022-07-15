import json
import numpy as np
from spotipy.spotipy.model import Config, SpotNetData, SpotNet
from spotipy.spotipy.utils import points_to_prob

if __name__ == '__main__':
    # config = Config(axes="YXC", n_channel_in=1, train_patch_size=(64, 64),
    #                 activation='elu', last_activation="sigmoid", backbone='unet')
    # Load the config
    with open('../../../models/leo3_multiscale_True_mae_aug_1_sigma_1.5_split_2_batch_2_n_300/config.json',
              'r') as config:
        config_dict = json.load(config)
    config = Config(**config_dict)

    fov_shape = 512

    P = np.random.randint(10, fov_shape - 10, (5, 30, 2))
    Y = np.stack(tuple(points_to_prob(p, shape=(fov_shape, fov_shape), sigma=1) for p in P))
    X = Y  # + .3 * np.random.normal(0, 1, Y.shape)

    model = SpotNet(config, name=None, basedir='/Users/buergy/Dropbox/epfl/projects/centrack/models/test_model')
    model.train(X, Y, validation_data=[X, Y], steps_per_epoch=10)
