import os
import contextlib
from spotipy.model import Config
import albumentations as alb

with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    config_unet = Config(
        n_channel_in=1,
        backbone="unet",
        mode="bce",
        unet_n_depth=3,
        unet_pool=4,
        unet_n_filter_base=64,
        spot_weight=40,
        multiscale=False,
        train_learning_rate=3e-4,
        train_foreground_prob=1,
        train_batch_norm=False,
        train_multiscale_loss_decay_exponent=1,
        train_patch_size=(512, 512),
        spot_weight_decay=0.5,
        train_batch_size=2,
    )
    config_multiscale = Config(
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

transforms = alb.Compose(
    [
        alb.ShiftScaleRotate(scale_limit=0.0),
        alb.Flip(),
        alb.RandomBrightnessContrast(always_apply=True),
        alb.RandomGamma(),
    ]
)