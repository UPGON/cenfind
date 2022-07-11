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

from labelbox import Client

from centrack.utils.status import DataSet

from centrack.utils.labelbox_api import (
    project_create,
    dataset_create,
    ontology_setup,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        type=Path)
    args = parser.parse_args()
    client = Client(api_key=os.environ['LABELBOX_API_KEY'])

    path_dataset = Path(args.path)
    dataset_name_lb = path_dataset.name
    project = project_create(client, 'centrioles')

    logger.debug('Get the ontology.')
    ontology_setup(client, project, ontology_id='cl3k8y38t11xc07807tar8hg6')

    dataset_lb = dataset_create(client, dataset_name_lb)

    project.datasets.connect(dataset_lb)
    logger.debug('Attach the dataset to the project.')

    dataset = DataSet(path_dataset)
    fields = tuple(
        f for f in dataset.vignettes.glob(f'*.png') if
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
    load_dotenv('/Users/buergy/Dropbox/epfl/projects/centrack/.env')
    cli()
