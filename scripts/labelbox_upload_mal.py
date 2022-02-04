import logging
from pathlib import Path

import numpy as np
import tifffile as tf
from labelbox import Client

from centrack.data import DataSet
from centrack.labelbox_api import (
    project_create,
    dataset_create,
    ontology_setup,
    generate_image,
    create_label,
    labels_list_create,
    prepare_upload_task
    )
from centrack.utils import contrast, extract_centriole

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
            image = generate_image(canvas, predictions)
            labels.append(create_label(image, predictions))
    else:
        dataset = DataSet(Path('/Volumes/work/epfl/datasets') / dataset_name)
        fields = tuple(f for f in dataset.projections.glob('*.tif') if
                       not f.name.startswith('.'))
        labels = []
        for field in fields:
            data = tf.imread(field)
            foci = data[1, :, :]
            predictions = extract_centriole(foci)
            predictions_np = [pred.position for pred in predictions]
            image = contrast(foci)
            labels.append(create_label(image, predictions_np))

    labels_list = labels_list_create(labels)

    task = prepare_upload_task(client, project, dataset_lb, labels_list)
    task.wait_until_done()


if __name__ == '__main__':
    main()
