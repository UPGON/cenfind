import cv2
import numpy as np
import pandas as pd
from skimage.draw import disk
from skimage.exposure import rescale_intensity
from skimage.feature import blob_log
from spotipy.utils import points_matching

from centrack.data.base import Dataset, Projection, Channel, Field
from centrack.experiments.constants import datasets, PREFIX_REMOTE


def prob2img(data):
    return (((2 ** 16) - 1) * data).astype('uint16')


def blob2point(keypoint: cv2.KeyPoint) -> tuple[int, ...]:
    res = tuple(int(c) for c in keypoint.pt)
    return res


def detect_centrioles_skimage_blob_log(data: Channel) -> list:
    data = data.data
    data = rescale_intensity(data, out_range=(0, 1))
    foci = blob_log(data, min_sigma=.5, max_sigma=5, num_sigma=10, threshold=.1)
    res = [(int(c), int(r)) for r, c, _ in foci]

    return res


def detect_centrioles_cv2_simple_blob_detector(data: Channel) -> list:
    foci = rescale_intensity(data.data, out_range='uint8')
    params = cv2.SimpleBlobDetector_Params()

    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 100
    params.minDistBetweenBlobs = 1
    params.minThreshold = 0
    params.maxThreshold = 255

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(foci)

    res = [blob2point(kp) for kp in keypoints]

    return res


def main():
    perfs = []
    for ds_name in datasets:
        ds = Dataset(PREFIX_REMOTE / ds_name)
        test_fields = ds.splits_for('test')
        for field, channel in test_fields:
            field = Field(field)
            proj = Projection(ds, field)
            data = Channel(proj, channel)
            annot = data.annotation()
            foci = detect_centrioles_cv2_simple_blob_detector(data)
            res = points_matching(annot, foci, cutoff_distance=3)
            f1 = np.round(res.f1, 3)
            perf = {'field': field.name,
                    'channel': channel,
                    'n': len(foci),
                    'f1': f1}
            print(perf)
            perfs.append(perf)

            mask = np.zeros(data.data.shape, dtype='uint8')
            for r, c in foci:
                print(r, c)
                rr, cc = disk((r, c), 4)
                try:
                    mask[rr, cc] = 250
                except IndexError as e:
                    continue
            cv2.imwrite(f'out/images/{field.channel_name(channel)}_preds.png', mask)

    pd.DataFrame(perfs).to_csv('out/perfs_blobdetector_blobcv2.csv')


if __name__ == '__main__':
    main()
