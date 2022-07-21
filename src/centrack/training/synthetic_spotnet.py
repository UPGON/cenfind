import numpy as np
from spotipy.spotipy.model import Config, SpotNetData, SpotNet
from spotipy.spotipy.utils import points_to_prob


def generate_data(points, fov_shape):
    Y = np.stack(tuple(points_to_prob(p, shape=(fov_shape, fov_shape), sigma=1) for p in points))
    X = Y + .3 * np.random.normal(0, 1, Y.shape)

    return X, Y


if __name__ == '__main__':
    fov_shape = 2048
    config = Config(axes="YXC", n_channel_in=1, train_patch_size=(512, 512),
                    activation='elu', last_activation="sigmoid", backbone='unet',
                    train_tensorboard=True)

    P_train = np.random.randint(10, fov_shape - 10, (22, 30, 2))
    X_train, Y_train = generate_data(P_train, fov_shape)

    P_test = np.random.randint(10, fov_shape - 10, (3, 30, 2))
    X_test, Y_test = generate_data(P_test, fov_shape)

    model = SpotNet(config, name=None, basedir='/Users/buergy/Dropbox/epfl/projects/centrack/models/dev')

    model.train(X_train, Y_train, validation_data=[X_test, Y_test],
                steps_per_epoch=100)
