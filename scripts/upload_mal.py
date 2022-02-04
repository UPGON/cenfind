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


def labels_list_create(shape, number_foci):
    """
    Generate a synthetic annotated dataset and convert it as a LabelList object.
    :return:
    """
    canvas = np.zeros(shape, 'uint8')
    annotations = []
    random_2dpoints = np.random.randint(0, min(shape), (number_foci, 2))
    for r, c in random_2dpoints:
        logger.debug('Generate object at position %s', (r, c))
        cv2.circle(canvas, (r, c), 20, 255, thickness=cv2.FILLED)
        annot = ObjectAnnotation(name='Centriole', value=Point(x=r, y=c))
        annotations.append(annot)

    image_data = ImageData.from_2D_arr(canvas)
    labels_list = LabelList()
    labels_list.append(Label(data=image_data, annotations=annotations))

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

    project_name = 'Test project.'
    project = project_create(client, project_name)

    logger.debug('Enable MAL.')
    project.enable_model_assisted_labeling()

    logger.debug('Get the ontology.')
    ontology_setup(client, project, ontology_id='ckywqubua5nkp0zb2h9lm3vn7')

    dataset_name = 'Test dataset.'
    dataset = dataset_create(client, dataset_name)

    project.datasets.connect(dataset)
    logger.debug('Attach the dataset to the project.')

    labels_list = labels_list_create((2048, 2048), 10)

    task = prepare_upload_task(client, project, dataset, labels_list)
    task.wait_until_done()


if __name__ == '__main__':
    main()
