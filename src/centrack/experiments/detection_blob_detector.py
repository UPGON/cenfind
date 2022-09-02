import cv2
import numpy as np
import pandas as pd
from skimage.draw import disk
from skimage.exposure import rescale_intensity
from skimage.feature import blob_log
from spotipy.utils import points_matching, normalize_fast2d
from centrack.cli.score import get_model
from centrack.data.base import Dataset, Projection, Channel, Field
from centrack.experiments.constants import datasets, PREFIX_REMOTE
from centrosome_analysis import centrosome_analysis_backend


def prob2img(data):
    return (((2 ** 16) - 1) * data).astype('uint16')


def blob2point(keypoint: cv2.KeyPoint) -> tuple[int, ...]:
    res = tuple(int(c) for c in keypoint.pt)
    return res


def log_skimage(data: Projection, channel: int, **kwargs) -> list:
    data = Channel(data, channel)
    data = data.data
    data = rescale_intensity(data, out_range=(0, 1))
    foci = blob_log(data, min_sigma=.5, max_sigma=5, num_sigma=10, threshold=.1)
    res = [(int(c), int(r)) for r, c, _ in foci]

    return res


def simpleblob_cv2(data: Projection, channel: int, **kwargs) -> list:
    data = Channel(data, channel)
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


def spotnet(data: Projection, foci_model_file, channel):
    data = Channel(data, channel)
    data = data.data
    data = normalize_fast2d(data)
    model = get_model(foci_model_file)
    mask_preds, points_preds = model.predict(data,
                                             prob_thresh=.5,
                                             min_distance=2)
    return points_preds[:, [1, 0]]


def sankaran(data: Projection, foci_model_file, **kwargs):
    data = data.data[1:, :, :]
    foci_model = centrosome_analysis_backend.load_foci_model(foci_model_file=foci_model_file)
    foci, foci_scores = centrosome_analysis_backend.run_detection_model(data, foci_model)
    detections = foci[foci_scores > .99, :]
    detections = np.round(detections).astype(int)
    return detections


def run_detection(method, data: Projection, tolerance, channel=None, model_path=None):
    channel_data = Channel(data, channel)
    annot = channel_data.annotation()
    foci = method(data, foci_model_file=model_path, channel=channel)
    res = points_matching(annot, foci, cutoff_distance=tolerance)
    f1 = np.round(res.f1, 3)
    return foci, f1


def draw_foci(data, foci):
    mask = np.zeros(data.data.shape, dtype='uint8')
    for r, c in foci:
        rr, cc = disk((r, c), 4)
        try:
            mask[rr, cc] = 250
        except IndexError:
            continue
    return mask


def main():
    methods = [spotnet, sankaran, log_skimage, simpleblob_cv2]
    model_paths = {
        'sankaran': 'src/centrosome_analysis/foci_model.pt',
        'spotnet': 'models/dev/2022-09-01_15:20:19',
        'log_skimage': None,
        'simpleblob_cv2': None,
    }
    perfs = []
    for ds_name in datasets:
        ds = Dataset(PREFIX_REMOTE / ds_name)
        test_fields = ds.splits_for('test')
        for field_name, channel in test_fields:
            field = Field(field_name)
            proj = Projection(ds, field)
            vis = proj.data[channel, :, :]

            for method in methods:
                model_path = model_paths[method.__name__]
                foci, f1 = run_detection(method, proj, 3, channel=channel, model_path=model_path)
                print(f"{field_name} using {method.__name__}: F1={f1}")
                perf = {'field': field.name, 'channel': channel, 'method': method.__name__, 'f1': f1}
                perfs.append(perf)
                mask = draw_foci(vis, foci)
                cv2.imwrite(f'out/images/{field.channel_name(channel)}_preds_{method.__name__}.png', mask)

    pd.DataFrame(perfs).to_csv(f'out/perfs_blobdetectors.csv')


if __name__ == '__main__':
    main()
