import logging
import uuid
from cv2 import cv2
import numpy as np
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


def main():
    with open('labelbox_api_key.txt', 'r') as apikey:
        lb_api_key = apikey.readline().rstrip('\n')

    client = Client(api_key=lb_api_key)

    project_name = 'Test project'
    projects_test = client.get_projects(where=Project.name == project_name)

    try:
        project = next(projects_test)
        logger.debug('Project test found (%s)', project)
        project.delete()
    except StopIteration:
        logger.debug('No such test project; creating...')
    finally:
        project = client.create_project(name=project_name, description='')

    project.enable_model_assisted_labeling()
    logger.debug('MAL enabled.')

    logger.debug('Getting the ontology')
    ontology = client.get_ontology(ontology_id='ckywqubua5nkp0zb2h9lm3vn7')
    ontology_builder = OntologyBuilder.from_ontology(ontology)
    editor = next(
        client.get_labeling_frontends(where=LabelingFrontend.name == 'editor'))
    project.setup(editor, ontology_builder.asdict())

    dataset_name = 'Test dataset'
    datasets_test = client.get_datasets(where=Dataset.name == dataset_name)
    try:
        dataset = next(datasets_test)
        logger.debug('Found a test dataset (%s)', dataset)
        dataset.delete()
    except StopIteration:
        logger.debug('No such test dataset; creating one...')
    finally:
        dataset = client.create_dataset(name=dataset_name, iam_integration=None)

    project.datasets.connect(dataset)
    logger.debug('Attaching the dataset to the project.')

    canvas = np.zeros((2048, 2048), 'uint8')
    annotations = []
    random_2dpoints = np.random.randint(0, 2048, (10, 2))
    for r, c in random_2dpoints:
        logger.debug('adding annot')
        cv2.circle(canvas, (r, c), 20, 255, thickness=cv2.FILLED)
        annot = ObjectAnnotation(name='Centriole', value=Point(x=r, y=c))
        annotations.append(annot)

    image_data = ImageData.from_2D_arr(canvas)
    labels_list = LabelList()
    labels_list.append(Label(data=image_data, annotations=annotations))

    signer = lambda _bytes: client.upload_data(content=_bytes, sign=True)

    labels_list.assign_feature_schema_ids(OntologyBuilder.from_project(project))
    labels_list.add_to_dataset(dataset, signer)

    ndjsons = list(NDJsonConverter.serialize(labels_list))
    task = project.upload_annotations(
        name=f'upload-job-{uuid.uuid4()}',
        annotations=ndjsons,
        validate=True
        )
    task.wait_until_done()


if __name__ == '__main__':
    main()
