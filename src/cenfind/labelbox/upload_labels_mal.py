import logging
from pathlib import Path

import cv2
import numpy as np
from dotenv import dotenv_values
from labelbox.exceptions import ResourceNotFoundError
from labelbox import (Client,
                      MediaType,
                      )

from cenfind.core.data import Dataset, Field
from cenfind.core.detectors import extract_foci
from cenfind.experiments.constants import datasets, PREFIX_REMOTE
from cenfind.labelbox.helpers import (ontology_setup,
                                      label_create,
                                      labels_list_create,
                                      task_prepare
                                      )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
foci_model = Path("models/dev/20221006_130126")


def main():
    config = dotenv_values('.env')
    client = Client(api_key=config['LABELBOX_API_KEY'])

    project_name = 'project_centrioles_c1'
    project_id = config['PROJECT_CENTRIOLES_C1']
    ontology_id = 'cl6gk46xv4wjc07yt4kyygxh1'
    channel_id = 3

    try:
        project = client.get_project(project_id)
    except ResourceNotFoundError:
        print('Project not existing, creating one...')
        project = client.create_project(name=project_name, media_type=MediaType.Image)
        ontology_setup(client, project, ontology_id=ontology_id)

    dataset_name = f'all_channel_{channel_id}'
    dataset = client.create_dataset(name=dataset_name, iam_integration=None)
    project.datasets.connect(dataset)

    labels = []
    for _ds in datasets:
        ds = Dataset(PREFIX_REMOTE / _ds)
        for field_name in ds.fields:
            field = Field(field_name, ds)
            predictions = extract_foci(field, foci_model, channel_id)
            vignette_path = ds.path / 'vignettes' / f"{field_name}_max_C{channel_id}.png"
            image = cv2.imread(str(vignette_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            labels.append(label_create(image, predictions, vignette_path.name))

    mal_label_list = labels_list_create(labels)
    task = task_prepare(client, project, dataset, mal_label_list)
    print(task.errors)


if __name__ == '__main__':
    main()
