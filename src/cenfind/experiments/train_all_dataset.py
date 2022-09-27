from datetime import datetime

from tqdm import tqdm
import numpy as np
from spotipy.model import SpotNet

from cenfind.core.data import Dataset
from cenfind.experiments.train_spotnet import load_pairs, config, transforms
from cenfind.experiments.constants import datasets, PREFIX_REMOTE

shuffle = True


def main():

    path_datasets = [PREFIX_REMOTE / ds for ds in datasets]
    dss = [Dataset(path) for path in path_datasets]

    all_train_x = []
    all_train_y = []

    all_test_x = []
    all_test_y = []

    for ds in tqdm(dss):
        train_x, train_y = load_pairs(ds, split='train', transform=transforms)
        test_x, test_y = load_pairs(ds, split='test')
        all_train_x.append(train_x)
        all_train_y.append(train_y)
        all_test_x.append(test_x)
        all_test_y.append(test_y)

    all_train_x = np.concatenate(all_train_x, axis=0)
    all_train_y = np.concatenate(all_train_y, axis=0)

    all_test_x = np.concatenate(all_test_x, axis=0)
    all_test_y = np.concatenate(all_test_y, axis=0)

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model = SpotNet(config, name=time_stamp+"unet", basedir='models/dev')
    model.train(all_train_x, all_train_y, validation_data=(all_test_x, all_test_y), epochs=100)

    return 0


if __name__ == '__main__':
    main()
