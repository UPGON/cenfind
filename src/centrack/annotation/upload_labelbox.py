import argparse
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
from labelbox import Client

from centrack.layout.dataset import DataSet

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser('Upload dataset to Labelbox')
    parser.add_argument('path', type=Path, help='The path to the dataset')
    return parser.parse_args()


def cli(args):
    client = Client(api_key=os.environ['LABELBOX_API_KEY'])
    path_dataset = args.path
    channel = args.channel
    dataset = DataSet(path_dataset)

    fields = tuple(
        f for f in dataset.vignettes.glob(f'*C{channel}.png') if
        not f.name.startswith('.'))

    data_rows = []
    for field in sorted(fields):
        external_id = field.name
        data_rows.append({'row_data': field,
                          'external_id': external_id})

    dataset_lb = client.create_dataset(name=f"{dataset.name}_C{channel}",
                                       iam_integration=None)
    task = dataset_lb.create_data_rows(data_rows)

    task.wait_till_done()
    print(task.status)


if __name__ == '__main__':
    load_dotenv('.env')
    args = get_args()
    cli(args)
