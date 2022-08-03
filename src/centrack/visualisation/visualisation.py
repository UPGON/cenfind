from matplotlib import pyplot as plt
import numpy as np
from centrack.training.synthetic_spotnet import generate_data


def show_images(image, mask):
    fig, ax = plt.subplots(ncols=3, figsize=(10, 3))
    ax[0].imshow(image, vmin=0, vmax=1)
    ax[0].set_title('Image')
    ax[1].imshow(mask, vmin=0, vmax=.1)
    ax[1].set_title('Mask')
    ax[2].imshow(mask * image, vmin=0, vmax=1)
    ax[2].set_title('Cross')
    return fig


if __name__ == '__main__':
    fov_shape = 512
    image = np.random.randint(0, 255, (fov_shape, fov_shape), dtype='uint8')
    mask = np.zeros_like(image)
    P_train = np.random.randint(10, fov_shape - 10, (1, 30, 2))
    X_train, Y_train = generate_data(P_train, fov_shape)
    fig = show_images(image, Y_train.squeeze())
    fig.savefig('/Users/buergy/Desktop/test.png')
