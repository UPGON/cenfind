import logging
import logging
import shutil
import uuid
from pathlib import Path
from random import randint

import labelbox
import numpy as np
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

from centrack.annotation import Centre
from centrack.labelbox_api import get_dataset_uid, get_project_uid, \
    get_lb_api_key

logging.basicConfig(level=logging.DEBUG)


def create_image(foci_number=1, brightness=255, width=2048, height=2048):
    canvas = np.zeros((width, height), 'uint8')
    annotation = []
    for _ in range(foci_number):
        x, y = randint(0, width - 1), randint(0, height - 1)
        annotation.append((x, y))
        logging.debug('Generate a point at %s', (x, y))
        cv2.circle(canvas, (x, y), 2, brightness, cv2.FILLED)
    return annotation, canvas


def format_annotations(centres_list):
    annotations = []
    for centre in centres_list:
        name = 'Centriole'
        r, c = centre.position
        logging.debug('Add the point %s', (r, c))
        annotations.append(ObjectAnnotation(
            name=name,
            value=Point(x=r, y=c)
            ))

    return annotations


def create_project(project_name, client):
    project_uid = get_project_uid(client, project_name)

    if project_uid:
        project = client.get_project(project_uid)
        logging.info('Project (%s) already exists.', project_name)
    else:
        project = client.create_project(name=project_name,
                                        description="")
        project.enable_model_assisted_labeling()
        add_ontology(ontology_id='ckywqubua5nkp0zb2h9lm3vn7',
                     project=project,
                     client=client)
        logging.info('Project (%s) created.', project_name)

    return project


def create_dataset(dataset_name, client):
    dataset_uid = get_dataset_uid(client, dataset_name)

    if dataset_uid:
        dataset = client.get_dataset(dataset_uid)
        logging.info(f'Dataset (%s) already exists. It will be deleted.',
                     dataset_name)
        dataset.delete()
    dataset = client.create_dataset(name=dataset_name,
                                    iam_integration=None)
    logging.info(f'Dataset (%s) created.', dataset_name)

    return dataset


def add_ontology(ontology_id, project, client):
    ontology = client.get_ontology(ontology_id)
    ontology_builder = OntologyBuilder.from_ontology(ontology)
    editor = next(
        client.get_labeling_frontends(
            where=LabelingFrontend.name == 'editor'))

    project.setup(editor, ontology_builder.asdict())


def annotation_data_upload(client, project, dataset, labels_list):
    ontology_builder = OntologyBuilder.from_project(project)
    signer = lambda _bytes: client.upload_data(content=_bytes, sign=True)

    (labels_list.assign_feature_schema_ids(ontology_builder)
     .add_to_dataset(dataset, signer))

    ndjsons = list(NDJsonConverter.serialize(labels_list))
    upload_task = project.upload_annotations(
        name=f"upload-job-{uuid.uuid4()}",
        annotations=ndjsons,
        validate=True)

    return upload_task


def label_list_from(mapping_dataset):
    labels_list = LabelList()
    for path, (annotation, foci) in mapping_dataset.items():
        foci = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image_data = ImageData.from_2D_arr(foci)
        annotations_gt = [Centre(position) for position in annotation]
        annotations = format_annotations(annotations_gt)
        labels_list.append(Label(
            data=image_data,
            annotations=annotations
            ))
    return labels_list


def main():
    image_path = Path('../out/mal_upload')

    if image_path.exists():
        shutil.rmtree(image_path)
        logging.warning('Delete the existing folder of synthetic images.')

    image_path.mkdir()

    logging.debug('Create synthetic images')
    mapping_dataset = dict()
    for i in range(3):
        path_dst = str(image_path / f'image{i:02}.png')
        mapping_dataset[path_dst] = create_image(foci_number=10)

    logging.debug('Save synthetic images')
    for path, (annotation, image) in mapping_dataset.items():
        cv2.imwrite(path, image)

    lb_api_key = get_lb_api_key('../configs/labelbox_api_key.txt')
    client = Client(api_key=lb_api_key)
    logging.info('Connection established')

    project_name = 'Test Project 4'
    project = create_project(project_name, client)

    dataset_name = 'Test Dataset'
    dataset = create_dataset(dataset_name,
                             client)  # Error upon second run when the project already exists

    # MAL
    # labels_list = label_list_from(mapping_dataset)
    # upload_task = annotation_data_upload(client, project, dataset, labels_list)
    # upload_task.wait_until_done()

    datarows = [p for p in mapping_dataset.keys()]
    uploads = []
    for path in datarows:
        path_name = Path(path).name
        item = {labelbox.DataRow.row_data: path,
                labelbox.DataRow.external_id: path_name}
        uploads.append(item)

    task = dataset.create_data_rows(uploads)
    task.wait_till_done()

    project.datasets.connect(dataset)

    logging.info('Dataset has been attached to the project')


if __name__ == '__main__':
    main()
