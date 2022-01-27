import logging
from pathlib import Path

import labelbox

from utils import get_lb_api_key

logging.basicConfig(level=logging.INFO)


def get_dataset_uid(client, name):
    """
    Retrieves the uid of a possibly existing dataset.
    :param client:
    :param name:
    :return:
    """
    datasets = client.get_datasets()
    dataset_ids = {ds.name: ds.uid for ds in datasets}
    if name in dataset_ids.keys():
        return dataset_ids[name]
    else:
        return None


def get_project_uid(client, name):
    """
    Retrieves the uid of a possibly existing project.
    :param client:
    :param name:
    :return:
    """
    projects = client.get_projects()
    project_ids = {proj.name: proj.uid for proj in projects}
    if name in project_ids.keys():
        return project_ids[name]
    else:
        return None


def main():
    path_root = Path('/Volumes/work/epfl/datasets')

    # 20210727_HA-FL-SAS6_Clones
    dataset_name = input(f'Enter the dataset name (under {path_root}): ')

    path_dataset = path_root / dataset_name
    if not path_dataset.exists():
        raise FileNotFoundError

    path_projections = path_dataset / 'projections'
    path_projections_channel = path_dataset / 'projections_channel'

    lb_api_key = get_lb_api_key('../configs/labelbox_api_key.txt')

    client = labelbox.Client(api_key=lb_api_key)
    logging.info('Connection established')

    # PROJECT SETUP
    project_uid = get_project_uid(client, dataset_name)

    if project_uid:
        project = client.get_project(project_uid)
        logging.info('Project (%s) already exists.', dataset_name)
    else:
        project = client.create_project(name=dataset_name,
                                        description="")
        logging.info('Project (%s) created.', dataset_name)

    # DATASET SETUP
    dataset_uid = get_dataset_uid(client, dataset_name)

    if dataset_uid:
        dataset = client.get_dataset(dataset_uid)
        logging.info(f'Dataset (%s) already exists.', dataset_name)
    else:
        dataset = client.create_dataset(name=dataset_name, iam_integration=None)
        logging.info(f'Dataset (%s) created.', dataset_name)

    # DATA LOADING
    datarows = sorted([str(path) for path in
                       path_projections_channel.rglob('**/*.png')],
                      key=lambda x: int(x[-5]))
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
