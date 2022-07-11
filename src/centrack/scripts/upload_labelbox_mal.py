import argparse
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

import tifffile as tf
from labelbox import Client

from centrack.utils.status import DataSet
from centrack.utils.outline import to_8bit
from centrack.commands.score import extract_centrioles

from centrack.utils.labelbox_api import (
    project_create,
    dataset_create,
    ontology_setup,
    label_create,
    labels_list_create,
    task_prepare
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

    logger.debug('Enable MAL.')
    project.enable_model_assisted_labeling()

    logger.debug('Get the ontology.')
    ontology_setup(client, project, ontology_id='cl3k8y38t11xc07807tar8hg6')

    dataset_name_lb = project_name_lb
    dataset_lb = dataset_create(client, dataset_name_lb)

    project.datasets.connect(dataset_lb)
    logger.debug('Attach the dataset to the project.')

    dataset = DataSet(path_dataset)
    fields = tuple(
        f for f in dataset.projections.glob(f'*C{channel_index}.tif') if
        not f.name.startswith('.'))
    labels = []
    for field in fields:
        external_id = field.name
        data = tf.imread(field)
        if data.ndim == 2:
            foci = data
        else:
            foci = data[channel_index, :, :]
        predictions = extract_centrioles(data, -1)
        predictions_np = [pred.position for pred in predictions]
        image = to_8bit(foci)
        labels.append(label_create(image, predictions_np, external_id))

    labels_list = labels_list_create(labels)

    task = task_prepare(client, project, dataset_lb, labels_list)
    task.wait_until_done()


if __name__ == '__main__':
    cli()
