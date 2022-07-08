import numpy as np
from spotipy.spotipy.utils import points_to_prob
import cv2

if __name__ == '__main__':
    points = np.random.randint(10, 128 - 10, (30, 2))
    mask = points_to_prob(points, shape=(128, 128), sigma=1)
    mask_image = (255 * mask).astype('uint8')
    cv2.imwrite('/home/buergy/projects/centrack/out/test.png', mask_image)
