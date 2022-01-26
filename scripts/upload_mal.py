import logging
import uuid

from labelbox import Client, LabelingFrontend
from labelbox.schema.ontology import OntologyBuilder, Tool

from centrack.utils import get_lb_api_key

logging.basicConfig(level=logging.INFO)


def main():
    lb_api_key = get_lb_api_key('../configs/labelbox_api_key.txt')
    client = Client(api_key=lb_api_key)
    logging.info('Connection established')

    # Only update this if you have an on-prem deployment
    ontology_builder = OntologyBuilder(tools=[
        Tool(tool=Tool.Type.BBOX, name="person"),
    ])
    project = client.create_project(name="image_mal_project")
    dataset = client.create_dataset(name="image_mal_dataset", iam_integration=None)
    test_img_url = "https://raw.githubusercontent.com/Labelbox/labelbox-python/develop/examples/assets/2560px-Kitano_Street_Kobe01s5s4110.jpg"
    data_row = dataset.create_data_row(row_data=test_img_url)
    editor = next(
        client.get_labeling_frontends(where=LabelingFrontend.name == 'editor'))
    project.setup(editor, ontology_builder.asdict())
    project.datasets.connect(dataset)
    project.enable_model_assisted_labeling()
    ontology = ontology_builder.from_project(project)
    feature_schema_id = ontology.tools[0].feature_schema_id

    # For more details see image_mal.ipynb or ner_mal.ipynb
    annotations = [{
        "uuid": str(uuid.uuid4()),
        "schemaId": feature_schema_id,
        "dataRow": {
            "id": data_row.uid,
        },
        "bbox": {
            "top": int(30),
            "left": int(30),
            "height": 200,
            "width": 200
        }
    }]

    upload_job = project.upload_annotations(name="upload_py_object_job",
                                            annotations=annotations)
    upload_job.wait_until_done()


if __name__ == '__main__':
    main()
