import cv2
import tifffile as tf
import numpy as np


def main():
    blue = cv2.imread('../data/blue.tif', cv2.IMREAD_GRAYSCALE)
    green = cv2.imread('../data/green.tif', cv2.IMREAD_GRAYSCALE)
    red = cv2.imread('../data/red.tif', cv2.IMREAD_GRAYSCALE)

    combined = np.zeros((512, 512, 3), np.uint8)

    combined[:, :, 0] = blue
    combined[:, :, 1] = green
    combined[:, :, 2] = red

    with tf.TiffWriter('../data/combined_tif_bgr.tif') as tif:
        tif.write(blue, photometric='minisblack')
        tif.write(green, photometric='minisblack')
        tif.write(red, photometric='minisblack')

    cv2.imwrite('../data/combined_cv2_bgr.tif', combined)


if __name__ == '__main__':
    main()