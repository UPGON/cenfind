import contextlib
import os
from typing import Tuple

import albumentations as alb
import numpy as np
import tifffile as tf
from spotipy.utils import points_to_prob, normalize_fast2d
from tqdm import tqdm

from cenfind.core.data import Dataset


def load_foci(path) -> np.ndarray:
    """
    Load annotation file from text file given channel
    loaded as row col, row major.
    ! the text format is x, y; origin at top left;
    :param channel:
    :return:
    """

    return np.loadtxt(str(path), dtype=int, delimiter=",")


def load_pairs(
        ds: Dataset, split: str, sigma: float = 1.5, suffix=None, transforms: alb.Compose = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load two arrays, the images and the foci masks
    path: the path to the ds
    split: either train or test
    """

    channels = []
    masks = []

    with open(ds.path / f"{split}.txt", "r") as f:
        pairs = [l.strip("\n").split(",") for l in f.readlines()]

    for field, channel in pairs:
        data = tf.imread(ds.projections / f"{field}{suffix}.tif")[int(channel), :, :]
        foci = load_foci(ds.annotations / "centrioles" / f"{field}{suffix}_C{channel}.txt")

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            image = normalize_fast2d(data, clip=True)

        if len(foci) == 0:
            mask = np.zeros(image.shape, dtype="uint16")
        else:
            mask = points_to_prob(foci, shape=image.shape, sigma=sigma)
        # from matplotlib import pyplot as plt
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(mask)
        # plt.show()
        if transforms is not None:
            transformed = transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        channels.append(image)
        masks.append(mask)

    return np.stack(channels), np.stack(masks)


def fetch_all_fields(datasets: list[Dataset], transforms: alb.Compose = None):
    all_train_x = []
    all_train_y = []

    all_test_x = []
    all_test_y = []

    for ds in tqdm(datasets):
        train_x, train_y = load_pairs(ds, split="train", suffix="_max", transforms=transforms)
        test_x, test_y = load_pairs(ds, suffix="_max", split="test")
        all_train_x.append(train_x)
        all_train_y.append(train_y)
        all_test_x.append(test_x)
        all_test_y.append(test_y)

    all_train_x = np.concatenate(all_train_x, axis=0)
    all_train_y = np.concatenate(all_train_y, axis=0)

    all_test_x = np.concatenate(all_test_x, axis=0)
    all_test_y = np.concatenate(all_test_y, axis=0)

    return all_train_x, all_train_y, all_test_x, all_test_y
