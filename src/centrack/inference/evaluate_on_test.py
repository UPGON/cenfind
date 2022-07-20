import pandas as pd

from centrack.inference.score import get_model
from centrack.layout.constants import datasets, PREFIX
from centrack.layout.dataset import DataSet, FieldOfView
from centrack.visualisation.visualisation import show_images
from spotipy.spotipy.utils import normalize_fast2d, points_matching


def main():
    model = get_model('models/master')
    performances = []
    for path_ds in datasets:
        print(path_ds)
        ds = DataSet(PREFIX / path_ds)
        ds.visualisation.mkdir(exist_ok=True)
        test_files = ds.split_images_channel('test')
        print(test_files)
        for fov_name, chid in test_files:
            print(fov_name)
            projection = ds.projections / f"{fov_name}_max.tif"
            fov = FieldOfView(projection)
            channel = fov[chid]
            inp = normalize_fast2d(channel)

            annotation = fov.fetch_annotation(chid)
            mask_preds, points_preds = model.predict(inp, prob_thresh=.5)
            res = points_matching(annotation[:, [1, 0]], points_preds)
            performances.append({'ds': ds.path.name,
                                 'fov': fov.name,
                                 'channel': chid,
                                 'foci_actual_n': len(annotation),
                                 'foci_preds_n': len(points_preds),
                                 'f1': res.f1,
                                 'precision': res.precision,
                                 'recall': res.recall})
            fig = show_images(inp, mask_preds)
            fig.savefig(ds.visualisation / f"{fov_name}_max_C{chid}.png")

    performances_df = pd.DataFrame(performances).round(3)
    performances_df = performances_df.set_index('fov')
    performances_df.to_csv('out/performances_base_model.csv')


if __name__ == '__main__':
    main()
