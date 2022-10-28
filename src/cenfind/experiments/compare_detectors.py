import cv2
import pandas as pd

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_foci
from cenfind.core.outline import draw_foci
from cenfind.experiments.constants import datasets, PREFIX_REMOTE
from cenfind.experiments.detectors_other import run_detection, log_skimage, simpleblob_cv2


def main():
    methods = [extract_foci, log_skimage, simpleblob_cv2]
    model_paths = {
        'spotnet': 'model/master',
        # 'sankaran': 'models/sankaran/dev/20220921_191227.pt',
        'log_skimage': None,
        'simpleblob_cv2': None,
    }
    perfs = []
    for ds_name in datasets:
        ds = Dataset(PREFIX_REMOTE / ds_name)
        for field, channel in ds.pairs(split='test'):
            vis = field.channel(channel)
            annotation = field.annotation(channel)

            for method in methods:
                model_path = model_paths[method.__name__]
                foci, f1 = run_detection(method, field, annotation=annotation, channel=channel,
                                         model_path=model_path, tolerance=3)
                print(f"{field.name} using {method.__name__}: F1={f1}")

                perf = {'field': field.name,
                        'channel': channel,
                        'method': method.__name__,
                        'f1': f1}
                perfs.append(perf)

                mask = draw_foci(vis, foci)
                cv2.imwrite(f'out/images/{field.name}_max_C{channel}_preds_{method.__name__}.png', mask)

    pd.DataFrame(perfs).to_csv(f'out/perfs_blobdetectors.csv')


if __name__ == '__main__':
    main()
