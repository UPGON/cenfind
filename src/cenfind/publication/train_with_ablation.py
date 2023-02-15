from datetime import datetime
from spotipy.model import SpotNet
from spotipy.model import Config
from cenfind.core.constants import PREFIX_REMOTE
from cenfind.core.data import Dataset
from cenfind.cli.train import load_pairs, transforms

config = Config(
    n_channel_in=1,
    backbone="unet",
    mode="mae",
    unet_n_depth=2,
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
    ds = Dataset(PREFIX_REMOTE / 'centrioles', projection_suffix='_max')

    train_x, train_y = load_pairs(ds, split="train", transform=transforms)
    test_x, test_y = load_pairs(ds, split="test")
    # validation_x, validation_y = load_pairs(ds, split="validation")

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model = SpotNet(config, name=f"multi_depth2_mae_{time_stamp}", basedir='models/dev/ablation')
    model.train(train_x, train_y, validation_data=(test_x, test_y), epochs=100)

    return 0


if __name__ == '__main__':
    main()