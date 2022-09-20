import cv2
import pandas as pd

from cenfind.core.data import Field, Dataset
from cenfind.core.measure import run_detection, detect_centrioles, sankaran, log_skimage, simpleblob_cv2
from cenfind.core.outline import draw_foci
from cenfind.experiments.constants import datasets, PREFIX_REMOTE


def main():
    methods = [detect_centrioles, sankaran, log_skimage, simpleblob_cv2]
    model_paths = {
        'sankaran': 'models/sankaran/dev/2022-09-05_09:23:45',
        'spotnet': 'models/dev/2022-09-02_14:31:28',
        'log_skimage': None,
        'simpleblob_cv2': None,
    }
    perfs = []
    for ds_name in datasets:
        ds = Dataset(PREFIX_REMOTE / ds_name)
        test_fields = ds.splits_for('test')
        for field_name, channel in test_fields:
            field = Field(field_name, ds)
            vis = field.channel(channel)
            annotation = field.annotation(channel)

            for method in methods:
                model_path = model_paths[method.__name__]
                foci, f1 = run_detection(method, field, annotation=annotation, channel=channel,
                                         model_path=model_path, tolerance=3)
                print(f"{field_name} using {method.__name__}: F1={f1}")

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
