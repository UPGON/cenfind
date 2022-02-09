import logging
from pathlib import Path

import numpy as np
import tifffile as tf
from labelbox import Client

from centrack.fetch import DataSet
from mal.labelbox_api import (
    project_create,
    dataset_create,
    ontology_setup,
    image_generate,
    label_create,
    labels_list_create,
    task_prepare
    )
from centrack.outline import contrast
from centrack.score import extract_centrioles

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

synthetic = False
dataset_name = '20210727_HA-FL-SAS6_Clones'


def main():
    with open('../configs/labelbox_api_key.txt', 'r') as apikey:
        lb_api_key = apikey.readline().rstrip('\n')

    client = Client(api_key=lb_api_key)

    project_name_lb = dataset_name
    project = project_create(client, project_name_lb)

    logger.debug('Enable MAL.')
    project.enable_model_assisted_labeling()

    logger.debug('Get the ontology.')
    ontology_setup(client, project, ontology_id='ckywqubua5nkp0zb2h9lm3vn7')

    dataset_name_lb = dataset_name
    dataset_lb = dataset_create(client, dataset_name_lb)

    project.datasets.connect(dataset_lb)
    logger.debug('Attach the dataset to the project.')

    if synthetic:
        shape = (2048, 2048)
        number_foci = 10

        labels = []
        for i in range(10):
            logger.debug('Create a label')
            predictions = np.random.randint(0, min(shape), (number_foci, 2))
            canvas = np.zeros(shape, 'uint8')
            image = image_generate(canvas, predictions)
            labels.append(label_create(image, predictions))
    else:
        dataset = DataSet(Path('/Volumes/work/epfl/datasets') / dataset_name)
        fields = tuple(f for f in dataset.projections.glob('*.tif') if
                       not f.name.startswith('.'))
        labels = []
        for field in fields:
            data = tf.imread(field)
            foci = data[1, :, :]
            predictions = extract_centrioles(foci)
            predictions_np = [pred.position for pred in predictions]
            image = contrast(foci)
            labels.append(label_create(image, predictions_np))

    labels_list = labels_list_create(labels)

    task = task_prepare(client, project, dataset_lb, labels_list)
    task.wait_until_done()


if __name__ == '__main__':
    main()
