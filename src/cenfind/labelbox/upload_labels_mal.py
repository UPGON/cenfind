import logging
from pathlib import Path
from tqdm import tqdm
import cv2
from dotenv import dotenv_values
from labelbox import Client, MediaType

from labelbox.schema.queue_mode import QueueMode
from labelbox.exceptions import ResourceNotFoundError

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_foci
from cenfind.experiments.constants import datasets, PREFIX_REMOTE
from cenfind.labelbox.helpers import (ontology_setup,
                                      label_create,
                                      labels_list_create,
                                      task_prepare
                                      )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
foci_model = Path("/data1/models/cenfind/master/")

path_dotenv = Path('.env')
if not path_dotenv.exists():
    FileNotFoundError(path_dotenv.resolve())

config = dotenv_values(path_dotenv)

def main():
    client = Client(api_key=config['LABELBOX_API_KEY'])

    project_name = 'centrioles_mal'
    project_id = config['PROJECT_CENTRIOLES_MAL']
    ontology_id = 'cl6gk46xv4wjc07yt4kyygxh1'

    try:
        project = client.get_project(project_id)
    except ResourceNotFoundError:
        print('Project not existing, creating one...')
        project = client.create_project(name=project_name,
                                        queue_mode=QueueMode.Dataset,
                                        # Quality Settings setup
                                        auto_audit_percentage=1,
                                        auto_audit_number_of_labels=1,
                                        media_type=MediaType.Image)
        ontology_setup(client, project, ontology_id=ontology_id)
    project.enable_model_assisted_labeling()

    for _ds in datasets:
        print(f'Processing {_ds}')
        ds = Dataset(PREFIX_REMOTE / _ds)
        dataset_name = ds.path.name

        dataset = client.create_dataset(name=dataset_name, iam_integration=None)
        project.datasets.connect(dataset)

        labels = []
        for field in tqdm(ds.fields):
            for channel_id in range(1, 4):
                mask, mal_label = extract_foci(field, foci_model, channel_id)
                vignette_path = ds.path / 'vignettes' / f"{field.name}_max_C{channel_id}.png"
                image = cv2.imread(str(vignette_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                labels.append(label_create(image, mal_label, vignette_path.name))
        mal_label_list = labels_list_create(labels)
        task = task_prepare(client, project, dataset, mal_label_list)
        print(task.errors)


if __name__ == '__main__':
    main()
