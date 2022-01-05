from pathlib import Path
import logging
import labelbox

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    path_root = Path('/Volumes/work/epfl/datasets')

    dataset_name = input(f'Enter the dataset name (under {path_root}): ')

    path_dataset = path_root / dataset_name
    path_projections = path_dataset / 'projections'
    path_projections_channel = path_dataset / 'projections_channel'

    with open('../configs/labelbox_api_key.txt', 'r') as apikey:
        LB_API_KEY = apikey.readline()
        LB_API_KEY = LB_API_KEY.rstrip('\n')

    client = labelbox.Client(api_key=LB_API_KEY)
    logging.info('Connection established')

    dataset = client.create_dataset(iam_integration=None, name=dataset_name)
    logging.info(f'Dataset created ({dataset_name})')

    datarows = sorted([str(path) for path in path_projections_channel.rglob('**/*.png')])

    task = dataset.create_data_rows(datarows)
    task.wait_till_done()
