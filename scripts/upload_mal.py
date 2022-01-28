import logging

from labelbox import Client, LabelingFrontend
from labelbox.schema.ontology import OntologyBuilder

from centrack.utils import get_lb_api_key

logging.basicConfig(level=logging.INFO)


def main():
    # SETUP
    lb_api_key = get_lb_api_key('../configs/labelbox_api_key.txt')
    client = Client(api_key=lb_api_key)
    logging.info('Connection established')

    project = client.get_project('cktegl25f5d4v0yc47b5n4gtx')
    project.enable_model_assisted_labeling()

    ontology = client.get_ontology('ckywqubua5nkp0zb2h9lm3vn7')
    ontology_builder = OntologyBuilder.from_ontology(ontology)
    editor = next(
        client.get_labeling_frontends(where=LabelingFrontend.name == 'editor'))
    project.setup(editor, ontology_builder.asdict())

    dataset = client.create_dataset(name="Test", iam_integration=None)
    project.datasets.connect(dataset)


if __name__ == '__main__':
    main()
