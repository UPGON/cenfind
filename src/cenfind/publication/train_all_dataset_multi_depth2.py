from datetime import datetime
from spotipy.model import SpotNet
from spotipy.model import Config
from cenfind.cli.train import fetch_all_fields
from cenfind.core.constants import PREFIX_REMOTE, datasets
from cenfind.core.data import Dataset

config = Config(
    n_channel_in=1,
    backbone="unet",
    mode="mae",
    unet_n_depth=2,
    unet_pool=8,
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
    path_datasets = [PREFIX_REMOTE / ds for ds in datasets]
    dss = [Dataset(path) for path in path_datasets]

    all_train_x, all_train_y, all_test_x, all_test_y = fetch_all_fields(dss)

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model = SpotNet(config, name=f"multi_depth2_{time_stamp}", basedir='models/dev/ablation')
    model.train(all_train_x, all_train_y, validation_data=(all_test_x, all_test_y), epochs=100)

    return 0


if __name__ == '__main__':
    main()
