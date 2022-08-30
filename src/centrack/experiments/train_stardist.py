import numpy as np
from stardist.models import Config2D, StarDist2D, StarDistData2D
from stardist import gputools_available
from csbdeep.utils import normalize
import tifffile as tf
from tqdm import tqdm
from pathlib import Path

from typing import List


def load_set(path: Path) -> List:
    """
    Load the train or test set from the file.
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        return [line.strip() for line in f]


def main():
    n_channel = 1

    path_dataset = Path('/data1/centrioles/rpe')
    path_images = path_dataset / 'projections'
    path_masks_cells = path_dataset / 'annotations/cells'

    train_set = load_set(path_dataset / 'train.txt')

    images = [str(path_images / f"{fov}_max_C2.tif") for fov in train_set]
    masks = [str(path_masks_cells / f"{fov}_max_C2.tif") for fov in train_set]

    images = list(map(tf.imread, images))
    masks = list(map(tf.imread, masks))

    images = [normalize(x, 1, 99.8) for x in tqdm(images)]

    rng = np.random.RandomState(42)
    ind = rng.permutation(len(images))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    images_val, masks_val = [images[i] for i in ind_val], [masks[i] for i in ind_val]
    images_trn, masks_trn = [images[i] for i in ind_train], [masks[i] for i in ind_train]

    # 32 is a good default choice (see 1_data.ipynb)
    n_rays = 32

    # Use OpenCL-based computations for dataset_test generator during training (requires 'gputools')
    use_gpu = False and gputools_available()

    # Predict on sub-sampled grid for increased efficiency and larger field of view
    grid = (2, 2)

    conf = Config2D(
        n_rays=n_rays,
        grid=grid,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
    )

    model = StarDist2D(conf, name='stardist_centrin', basedir='/home/buergy/projects/centrack/models')

    model.train(images_trn, masks_trn, validation_data=(images_val, masks_val),
                # epochs=2,
                # steps_per_epoch=10
                )


if __name__ == '__main__':
    main()
