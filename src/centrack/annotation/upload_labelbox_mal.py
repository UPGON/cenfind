import argparse
import logging
import os
from pathlib import Path

from labelbox import Client
from dotenv import dotenv_values

from spotipy.utils import normalize_fast2d

from centrack.inference.score import get_model
from centrack.layout.dataset import DataSet, FieldOfView
from centrack.visualisation.outline import to_8bit
from centrack.annotation.vignettes import create_vignettes
from centrack.annotation.labelbox_api import (
    label_create,
    labels_list_create,
    task_prepare
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def cli():
    config = dotenv_values('/home/buergy/projects/centrack/.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        type=Path)
    parser.add_argument('model', type=str)
    args = parser.parse_args()

    client = Client(api_key=config['LABELBOX_API_KEY'])

    path_dataset = Path(args.path)

    dataset = DataSet(path_dataset)

    project = client.get_project(project_id='cl64z5mv20gte07vfbsjqdtvb')
    dataset_lb = client.create_dataset(name=f"{dataset.name}_test",
                                       iam_integration=None)
    project.datasets.connect(dataset_lb)

    train_files = dataset.split_images_channel('train')
    test_files = dataset.split_images_channel('test')
    all_files = train_files + test_files

    model = get_model(args.model)

    labels = []
    for fov_name, channel_id in all_files:
        channel_id = int(channel_id)
        data = FieldOfView(dataset, fov_name)
        foci = data.load_channel(channel_id)
        foci = normalize_fast2d(foci)
        mask_preds, points_preds = model.predict(foci,
                                                 prob_thresh=.5,
                                                 min_distance=2)
        vignette = create_vignettes(data, channel_id, 0)
        labels.append(label_create(vignette, points_preds, fov_name))

    labels_list = labels_list_create(labels)

    task = task_prepare(client, project, dataset_lb, labels_list)
    task.wait_until_done()


if __name__ == '__main__':
    cli()
