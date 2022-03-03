import tensorflow as tf

from src.centrack import SpotNet, Config


def main():
    config = Config(axes="YXC", n_channel_in=1, train_patch_size=(64, 64),
                    activation='elu', last_activation="sigmoid",
                    backbone='unet')

    model = SpotNet(config, name=None, basedir=None)

    tf.keras.utils.plot_model(model.keras_model, to_file='../../../data/models/spotnet.png',
                              show_shapes=True)


if __name__ == '__main__':
    main()
