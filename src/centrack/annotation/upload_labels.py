import argparse
from pathlib import Path

from dotenv import dotenv_values
from labelbox import (Client,
                      OntologyBuilder,
                      LabelingFrontend, )

from centrack.layout.dataset import DataSet


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    config = dotenv_values('/home/buergy/projects/centrack/.env')

    client = Client(api_key=config['LABELBOX_API_KEY'])
    project = client.create_project(name='centrioles')
    project.enable_model_assisted_labeling()

    ontology_id = config['ONTOLOGY_CENTRIOLES']
    ontology = client.get_ontology(ontology_id)
    ontology_builder = OntologyBuilder.from_ontology(ontology)
    editor = next(client.get_labeling_frontends(where=LabelingFrontend.name == "Editor"))
    project.setup(editor, ontology_builder.asdict())

    ds = DataSet(args.path)

    dataset = client.create_dataset(name=f"{ds.name}", iam_integration=None)
    project.datasets.connect(dataset)

    asset = [{"row_data": path, "external_id": path.name} for path in ds.vignettes.iterdir()]
    dataset.create_data_rows(asset)


if __name__ == '__main__':
    cli()
