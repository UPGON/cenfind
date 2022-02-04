from pathlib import Path

import labelbox
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

with open('../configs/labelbox_api_key.txt', 'r') as apikey:
    LB_API_KEY = apikey.readline()
    LB_API_KEY = LB_API_KEY.rstrip('\n')

projects = {
    'RPE1wt_CEP63+CETN2+PCNT_1': 'ckusghnn68fir0zehargo6jcd',
    'RPE1wt_CEP152+GTU88+PCNT_1': 'ckuwn9zv52zi20zb2esdv14xp',
    'RPE1wt_CP110+GTU88+PCNT_2': 'ckuph8qhf3evx0zagg5g5cwdb',
}

dataset_name = 'RPE1wt_CEP152+GTU88+PCNT_1'
path_root = Path('/Volumes/work/epfl/datasets')
path_dataset = path_root / dataset_name


def main():
    lb = labelbox.Client(api_key=LB_API_KEY)
    project_uid = projects[dataset_name]
    project = lb.get_project(project_uid)
    labels = project.export_labels(download=True)

    data_export = []
    channels_id = [1, 2, 3]

    for label_id, label in enumerate(labels):
        for c_id, channel_id in enumerate(channels_id):
            image_name = label['External ID']
            image_name = image_name.split('/')[-1]

            logging.info(f"{image_name}: {label_id}, {c_id}")

            if image_name.startswith('.'):
                logging.warning(f"{image_name} skipped")
                continue

            image_name = image_name.split('.')[0]

            condition, markers, replicate, ds_row, ds_col, _, channel_code = image_name.split('_')
            markers = markers.split('+')
            markers.insert(0, 'DAPI')
            marker = markers[channel_id]

            if f'C{channel_id}' not in image_name:
                continue
            if 'objects' not in label['Label'].keys():
                continue

            foci = label['Label']['objects']

            for focus_id, focus in enumerate(foci):
                point = focus['point']
                x = int(point['x'])
                y = int(point['y'])

                focus_dict = {'image_name': image_name,
                              'ds_row': int(ds_row),
                              'ds_col': int(ds_col),
                              'channel_id': channel_id,
                              'marker': marker,
                              'focus_id': focus_id,
                              'x': x,
                              'y': y}
                data_export.append(focus_dict)

    data_export_df = pd.DataFrame(data_export)
    data_export_df.to_csv(path_dataset / f'{dataset_name}_annotation.csv', index=False)
    logging.info(f"Writing {len(data_export_df)} objects to {path_dataset / f'{dataset_name}_annotation.csv'}")


if __name__ == '__main__':
    main()
