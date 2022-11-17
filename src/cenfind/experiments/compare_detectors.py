import cv2
import pandas as pd

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_foci
from cenfind.core.outline import draw_foci
from cenfind.experiments.constants import datasets, PREFIX_REMOTE
from cenfind.experiments.detectors_other import run_detection, log_skimage, simpleblob_cv2


def main():
    methods = [
        ['unet', extract_foci, 'models/dev/unet/20221116_193624'],
        ['multiscale', extract_foci, 'models/dev/multiscale/20221116_160118'],
        ['log_skimage', log_skimage, None],
        ['simpleblob_cv2', simpleblob_cv2, None],
    ]
    perfs = []
    for ds_name in datasets:
        ds = Dataset(PREFIX_REMOTE / ds_name)
        for field, channel in ds.pairs(split='test'):
            vis = field.channel(channel)
            annotation = field.annotation(channel)

            for name, method, model_path in methods:
                foci, f1 = run_detection(method, field, annotation=annotation, channel=channel,
                                         model_path=model_path, tolerance=3)
                print(f"{field.name} using {name}: F1={f1} (foci detected: {len(foci)})")

                perf = {'dataset': field.dataset.path.name,
                        'field': field.name,
                        'channel': channel,
                        'method': name,
                        'f1': round(f1, 3)}
                perfs.append(perf)

                mask = draw_foci(vis, foci)
                cv2.imwrite(f'out/images/{field.name}_max_C{channel}_preds_{name}.png', mask)

    perfs_df = pd.DataFrame(perfs)
    perfs_df.to_csv(f'out/perfs_blobdetectors.csv')

    summary = perfs_df.groupby('method')['f1'].agg(['mean', 'std']).round(3)
    summary = summary.reset_index()
    summary['F1 (mean ± st. dev.)'] = summary['mean'].astype(str) + '±' + summary['std'].astype(str)
    summary.columns = ['Method', 'F1', 'Standard deviation', 'F1 (mean ± st. dev.)']
    summary.to_csv(f'out/perfs_blobdetectors_summary.csv')


if __name__ == '__main__':
    main()
