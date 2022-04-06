import tensorflow as tf
import sys
import os
from spotipy.spotipy.model import SpotNet, Config
import numpy as np
from spotipy.spotipy.utils import points_to_prob
from pathlib import Path
from centrack.commands.score import get_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    print("Tensorflow version: ", tf.__version__)

    # current_path = Path(__file__).parent.parent.parent.parent
    # path_to_model = current_path / 'models/leo3_multiscale_True_mae_aug_1_sigma_1.5_split_2_batch_2_n_300'
    # model = get_model(
    #     model=path_to_model)

    config = Config(axes="YXC", n_channel_in=1, train_patch_size=(64, 64),
                    activation='elu', last_activation="sigmoid", backbone='unet')

    model = SpotNet(config, name=None, basedir=None)

    P = np.random.randint(10, 512 - 10, (5, 30, 2))

    Y = np.stack(tuple(points_to_prob(p, shape=(512, 512), sigma=1) for p in P))

    X = Y + .3 * np.random.normal(0, 1, Y.shape)

    model.train(X, Y, validation_data=[X, Y], steps_per_epoch=100)


if __name__ == '__main__':
    sys.exit(main())
