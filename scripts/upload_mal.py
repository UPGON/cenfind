import logging
import uuid

import numpy as np
from cv2 import cv2
from labelbox import (
    Client, OntologyBuilder, LabelingFrontend, Dataset, Project
    )
from labelbox.data.annotation_types import (
    ObjectAnnotation,
    Point,
    LabelList,
    Label,
    ImageData,
    )
from labelbox.data.serialization import NDJsonConverter

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


def project_create(client, project_name):
    projects_test = client.get_projects(where=Project.name == project_name)

    try:
        project = next(projects_test)
        logger.debug('Project test found (%s)', project)
        project.delete()
    except StopIteration:
        logger.debug('No such test project; creating...')
    finally:
        project = client.create_project(name=project_name, description='')
    return project


def ontology_setup(client, project, ontology_id):
    """
    Fetch the ontology using the ID and attach it to the project.
    This is in place.
    :param client:
    :param project:
    :param ontology_id:
    :return:
    """
    ontology = client.get_ontology(ontology_id=ontology_id)
    ontology_builder = OntologyBuilder.from_ontology(ontology)
    editor = next(
        client.get_labeling_frontends(where=LabelingFrontend.name == 'editor'))
    project.setup(editor, ontology_builder.asdict())


def dataset_create(client, dataset_name):
    """
    Create a dataset object and delete any dataset with same name.
    :param client:
    :param dataset_name:
    :return:
    """
    datasets_test = client.get_datasets(where=Dataset.name == dataset_name)
    try:
        dataset = next(datasets_test)
        logger.debug('Found a test dataset (%s)', dataset.name)
        dataset.delete()
    except StopIteration:
        logger.debug('No such test dataset; creating one...')
    finally:
        dataset = client.create_dataset(name=dataset_name, iam_integration=None)
    return dataset


def generate_image(canvas, predictions):
    """
    Draw foci on a black canvas at positions of predictions.
    :param canvas:
    :param predictions:
    :return:
    """
    for r, c in predictions:
        cv2.circle(canvas, (r, c), 20, 255, thickness=cv2.FILLED)
    return canvas


def to_labelbox_format(predictions):
    """
    Convert numpy coordinates into Labelbox format.
    :param predictions:
    :return:
    """
    annotations = []
    for r, c in predictions:
        annot = ObjectAnnotation(name='Centriole', value=Point(x=r, y=c))
        annotations.append(annot)
    return annotations


def create_label(image, predictions):
    """
    Combine an image and its annotation into a Label.
    :param predictions:
    :param image:
    :return:
    """
    return Label(data=ImageData.from_2D_arr(image),
                 annotations=to_labelbox_format(predictions))


def labels_list_create(labels):
    """
    Combine all labels into a LabelList object.
    :return:
    """
    labels_list = LabelList()
    for label in labels:
        labels_list.append(label)
    return labels_list


def prepare_upload_task(client, project, dataset, labels_list):
    signer = lambda _bytes: client.upload_data(content=_bytes, sign=True)

    labels_list.assign_feature_schema_ids(OntologyBuilder.from_project(project))
    labels_list.add_to_dataset(dataset, signer)

    ndjsons = list(NDJsonConverter.serialize(labels_list))
    task = project.upload_annotations(
        name=f'upload-job-{uuid.uuid4()}',
        annotations=ndjsons,
        validate=True
        )
    return task


def main():
    with open('../configs/labelbox_api_key.txt', 'r') as apikey:
        lb_api_key = apikey.readline().rstrip('\n')

    client = Client(api_key=lb_api_key)

    project_name = 'Test project'
    project = project_create(client, project_name)

    logger.debug('Enable MAL.')
    project.enable_model_assisted_labeling()

    logger.debug('Get the ontology.')
    ontology_setup(client, project, ontology_id='ckywqubua5nkp0zb2h9lm3vn7')

    dataset_name = 'Test dataset'
    dataset = dataset_create(client, dataset_name)

    project.datasets.connect(dataset)
    logger.debug('Attach the dataset to the project.')

    shape = (2048, 2048)
    number_foci = 10

    labels = []
    for i in range(10):
        logger.debug('Create a label')
        predictions = np.random.randint(0, min(shape), (number_foci, 2))
        canvas = np.zeros(shape, 'uint8')
        image = generate_image(canvas, predictions)
        labels.append(create_label(image, predictions))

    labels_list = labels_list_create(labels)

    task = prepare_upload_task(client, project, dataset, labels_list)
    task.wait_until_done()


if __name__ == '__main__':
    main()
