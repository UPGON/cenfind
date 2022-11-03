from datetime import datetime

import tensorflow as tf
from spotipy.model import SpotNet

from cenfind.core.data import Dataset
from cenfind.experiments.constants import datasets, PREFIX_REMOTE
from cenfind.experiments.train_spotnet import fetch_all_fields, config_unet

shuffle = True

# GLOBAL SEED
tf.random.set_seed(3)


def main():
    path_datasets = [PREFIX_REMOTE / ds for ds in datasets]
    dss = [Dataset(path) for path in path_datasets]

    all_train_x, all_train_y, all_test_x, all_test_y = fetch_all_fields(dss)

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_unet = SpotNet(config_unet, name=time_stamp, basedir='models/dev/unet')
    model_unet.train(all_train_x, all_train_y, validation_data=(all_test_x, all_test_y), epochs=100)

    return 0


if __name__ == '__main__':
    main()
