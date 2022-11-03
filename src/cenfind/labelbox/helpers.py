import logging
import uuid

from labelbox import (
    OntologyBuilder,
    LabelingFrontend,
    Dataset,
    Project,
    MALPredictionImport
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
logger.setLevel(logging.DEBUG)


def to_labelbox_format(predictions):
    """
    Convert numpy coordinates into Labelbox format.
    :param predictions:
    :return:
    """
    annotations = []
    for r, c in predictions:
        annot = ObjectAnnotation(name='Centriole', value=Point(x=c, y=r))
        annotations.append(annot)
    return annotations


def label_create(image, predictions, external_id):
    """
    Combine an image and its annotation into a Label.
    :param predictions:
    :param image:
    :return:
    """
    label = Label(data=ImageData(arr=image),
                  annotations=to_labelbox_format(predictions))
    label.data.external_id = external_id
    return label


def labels_list_create(labels):
    """
    Combine all labels into a LabelList object.
    :return:
    """
    labels_list = LabelList()
    for label in labels:
        labels_list.append(label)
    return labels_list


def task_prepare(client, project, dataset, labels_list):
    signer = lambda _bytes: client.upload_data(content=_bytes, sign=True)

    # labels_list.assign_feature_schema_ids(OntologyBuilder.from_project(project))
    labels_list.add_to_dataset(dataset, signer)

    mal_ndjson = list(NDJsonConverter.serialize(labels_list))
    upload_job = MALPredictionImport.create_from_objects(
        client=client,
        project_id=project.uid,
        name=f'upload-job-{uuid.uuid4()}',
        predictions=mal_ndjson)

    return upload_job


def project_create(client, project_name):
    """
    Create a project object or return any existing with `project name`.
    :param client:
    :param project_name:
    :return:
    """
    projects_test = client.get_projects(where=Project.name == project_name)

    try:
        project = next(projects_test)
        logger.debug('Project test found (%s)', project.uid)
        return project
    except StopIteration:
        logger.debug('No such project; creating...')
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
