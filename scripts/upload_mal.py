import contextlib
import logging
import os
import uuid
from pathlib import Path
from random import randint

import numpy as np
import tifffile as tf
from cv2 import cv2
from labelbox import OntologyBuilder, LabelingFrontend, Client
from labelbox.data.annotation_types import (
    ImageData,
    ObjectAnnotation,
    Label,
    LabelList,
    Point,
    )
from labelbox.data.serialization import NDJsonConverter

from centrack.labelbox_api import get_dataset_uid, get_project_uid, \
    get_lb_api_key
from centrack.utils import extract_centriole

logging.basicConfig(level=logging.INFO)


def create_image(foci_number=30, brightness=255, width=2048, height=2048):
    canvas = np.zeros((1024, 1024), 'uint8')
    for _ in range(foci_number):
        x, y = randint(0, width - 1), randint(0, height - 1)
        cv2.circle(canvas, (x, y), 2, brightness, cv2.FILLED)
    return canvas


def format_annotations(centres_list):
    annotations = []
    for centre in centres_list:
        name = 'Centriole'
        x, y = centre.position
        annotations.append(ObjectAnnotation(
            name=name,
            value=Point(x=x, y=y)
            ))

    return annotations


def main():
    dataset = [create_image() for _ in range(10)]
    image_path = Path('../out/mal_upload')
    image_path.mkdir(exist_ok=True)
    for i, image in enumerate(dataset):
        cv2.imwrite(str(image_path / f'image{i:02}.tif'), image)

    paths = tuple(f for f in image_path.iterdir()
                  if f.name.endswith('.tif')
                  and not f.name.startswith('.'))
    labels_list = LabelList()
    for p in paths:
        foci = tf.imread(p)
        foci_bgr = np.expand_dims(foci, -1)
        image_data = ImageData(arr=foci_bgr)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            foci_detected = extract_centriole(foci)
        annotations = format_annotations(foci_detected)
        labels_list.append(Label(
            data=image_data,
            annotations=annotations
            ))

    # SETUP
    lb_api_key = get_lb_api_key('../configs/labelbox_api_key.txt')
    client = Client(api_key=lb_api_key)
    logging.info('Connection established')

    project_name = 'Test Project'
    dataset_name = 'Test Dataset'

    # PROJECT SETUP
    project_uid = get_project_uid(client, project_name)

    if project_uid:
        project = client.get_project(project_uid)
        logging.info('Project (%s) already exists.', project_name)
    else:
        project = client.create_project(name=project_name,
                                        description="")
        logging.info('Project (%s) created.', project_name)

    # DATASET SETUP
    dataset_uid = get_dataset_uid(client, dataset_name)

    if dataset_uid:
        dataset = client.get_dataset(dataset_uid)
        logging.info(f'Dataset (%s) already exists.', dataset_name)
    else:
        dataset = client.create_dataset(name=dataset_name, iam_integration=None)
        logging.info(f'Dataset (%s) created.', dataset_name)

    ontology = client.get_ontology('ckywqubua5nkp0zb2h9lm3vn7')
    ontology_builder = OntologyBuilder.from_ontology(ontology)
    editor = next(
        client.get_labeling_frontends(where=LabelingFrontend.name == 'editor'))
    project.setup(editor, ontology_builder.asdict())

    project.datasets.connect(dataset)
    project.enable_model_assisted_labeling()

    signer = lambda _bytes: client.upload_data(content=_bytes, sign=True)
    (labels_list.assign_feature_schema_ids(
        OntologyBuilder.from_project(project))
     .add_to_dataset(dataset, signer))

    ndjsons = list(NDJsonConverter.serialize(labels_list))
    upload_task = project.upload_annotations(
        name=f"upload-job-{uuid.uuid4()}",
        annotations=ndjsons,
        validate=True)

    upload_task.wait_until_done()


if __name__ == '__main__':
    main()
