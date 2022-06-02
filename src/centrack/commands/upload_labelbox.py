#! /home/buergy/.cache/pypoetry/virtualenvs/centrack-7dpZ9I7w-py3.9/bin/python
"""
Take the path to the projection dataset
Select the channel
Construct the dataset on labelbox
Append it if necessary to the project on labelbox
Upload
"""
import argparse
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

import tifffile as tf
from cv2 import cv2
from labelbox import Client

from centrack.commands.status import DataSet

from centrack.mal.labelbox_api import (
    project_create,
    dataset_create,
    ontology_setup,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        type=Path)
    parser.add_argument('channel',
                        type=str,
                        help='Index of the channel')

    return parser.parse_args()


def cli():
    parsed_args = args_parse()
    load_dotenv('/home/buergy/projects/centrack/.env')
    client = Client(api_key=os.environ['LABELBOX_API_KEY'])

    path_dataset = Path(parsed_args.path)
    channel_index = parsed_args.channel
    project_name_lb = f"{path_dataset.name}_C{channel_index}"
    project = project_create(client, project_name_lb)

    logger.debug('Get the ontology.')
    ontology_setup(client, project, ontology_id='cl3k8y38t11xc07807tar8hg6')

    dataset_name_lb = project_name_lb
    dataset_lb = dataset_create(client, dataset_name_lb)

    project.datasets.connect(dataset_lb)
    logger.debug('Attach the dataset to the project.')

    dataset = DataSet(path_dataset)
    fields = tuple(
        f for f in dataset.vignettes.glob(f'*C{channel_index}.png') if
        not f.name.startswith('.'))

    data_rows = []
    for field in fields:
        external_id = field.name
        data_rows.append({'row_data': field,
                          'external_id': external_id})

    # Bulk add data rows to the dataset
    task = dataset_lb.create_data_rows(data_rows)

    task.wait_till_done()
    print(task.status)


if __name__ == '__main__':
    cli()
