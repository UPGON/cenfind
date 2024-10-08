from pathlib import Path

from numpy.random import seed

seed(1)

import tensorflow as tf

tf.random.set_seed(2)
from datetime import datetime
from spotipy.model import SpotNet
from spotipy.model import Config
from cenfind.core.data import Dataset
from cenfind.core.loading import load_pairs
from cenfind.training.config import transforms

PREFIX_REMOTE = Path("/data1/centrioles/canonical/")
config = Config(
    n_channel_in=1,
    backbone="unet",
    mode="mae",
    unet_n_depth=3,
    unet_pool=4,
    unet_n_filter_base=64,
    spot_weight=40,
    multiscale=True,
    train_learning_rate=3e-4,
    train_foreground_prob=1,
    train_batch_norm=False,
    train_multiscale_loss_decay_exponent=1,
    train_patch_size=(512, 512),
    spot_weight_decay=0.5,
    train_batch_size=2,
)


def main():
    ds = Dataset(PREFIX_REMOTE / 'centrioles')

    train_x, train_y = load_pairs(ds, split="train", transforms=transforms, sigma=1.5, suffix='_max')
    test_x, test_y = load_pairs(ds, split="test", sigma=1.5, suffix='_max')

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model = SpotNet(config, name=f"multi_depth3_bce_sigma5_{time_stamp}", basedir='models/dev/multiscale')
    model.train(train_x, train_y, validation_data=(test_x, test_y), epochs=50)

    return 0


if __name__ == '__main__':
    main()
