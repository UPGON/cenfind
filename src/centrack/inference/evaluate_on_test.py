import pandas as pd

from centrack.inference.score import get_model
from centrack.utils.constants import datasets, PREFIX_LOCAL
from centrack.layout.dataset import DataSet, FieldOfView
from spotipy.utils import normalize_fast2d, points_matching


def write_performances(dst, performances):
    performances_df = pd.DataFrame(performances).round(3)
    performances_df = performances_df.set_index('fov')
    performances_df.to_csv(dst)


def run_evaluation(path, model, cutoff):
    ds = DataSet(path)
    ds.visualisation.mkdir(exist_ok=True)
    test_files = ds.split_images_channel('test')
    perfs = []

    for fov_name, channel_id in test_files:
        print(fov_name)
        projection = f"{fov_name}_max.tif"
        fov = FieldOfView(ds, projection)
        channel = fov.load_channel(channel_id)
        inp = normalize_fast2d(channel)

        annotation = fov.load_annotation(channel_id)
        mask_preds, points_preds = model.predict(inp,
                                                 prob_thresh=.5,
                                                 min_distance=2)
        res = points_matching(annotation[:, [1, 0]],
                              points_preds,
                              cutoff_distance=cutoff)

        perfs.append({'ds': ds.path.name,
                      'fov': fov.name,
                      'channel': channel_id,
                      'foci_actual_n': len(annotation),
                      'foci_preds_n': len(points_preds),
                      'cutoff': cutoff,
                      'f1': res.f1,
                      'precision': res.precision,
                      'recall': res.recall}
                     )
    return perfs


def main():
    model_name = 'master'
    cutoffs = [1, 2, 5, 10]
    model = get_model(f'models/{model_name}')
    dst = f'out/performances_{model_name}.csv'
    performances = []

    for ds_name in datasets:
        path_dataset = PREFIX_LOCAL / ds_name
        for cutoff in cutoffs:
            performance = run_evaluation(path_dataset, model, cutoff)
            performances.append(performance)

    perfs_flat = [s
                  for p in performances
                  for s in p]
    write_performances(dst, perfs_flat)


if __name__ == '__main__':
    main()
