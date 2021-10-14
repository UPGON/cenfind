from pathlib import Path

import labelbox

if __name__ == '__main__':
    path_root = Path('/Volumes/work/epfl/datasets')

    dataset_name = 'RPE1wt_CP110+GTU88+PCNT_2'

    path_dataset = path_root / dataset_name
    path_projections = path_dataset / 'projections'
    path_projections_channel = path_dataset / 'projections_channel'

    with open('configs/labelbox_api_key.txt', 'r') as apikey:
        LB_API_KEY = apikey.readline()
        LB_API_KEY = LB_API_KEY.rstrip('\n')

    client = labelbox.Client(api_key=LB_API_KEY)

    dataset = client.create_dataset(iam_integration=None, name=dataset_name)

    datarows = [str(path) for path in path_projections_channel.rglob('**/*.png')]

    task1 = dataset.create_data_rows(datarows)
    task1.wait_till_done()
