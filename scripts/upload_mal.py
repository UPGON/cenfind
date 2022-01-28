import logging
import uuid

from labelbox import Client, LabelingFrontend
from labelbox.schema.ontology import OntologyBuilder

from centrack.utils import get_lb_api_key

logging.basicConfig(level=logging.INFO)


def main():
    lb_api_key = get_lb_api_key('../configs/labelbox_api_key.txt')
    client = Client(api_key=lb_api_key)
    logging.info('Connection established')

    project = client.get_project('cktegl25f5d4v0yc47b5n4gtx')
    project.enable_model_assisted_labeling()

    ontology = client.get_ontology('ckywqubua5nkp0zb2h9lm3vn7')
    ontology_builder = OntologyBuilder.from_ontology(ontology)
    editor = next(client.get_labeling_frontends(where=LabelingFrontend.name == 'editor'))
    project.setup(editor, ontology_builder.asdict())

    dataset = client.create_dataset(name="Test", iam_integration=None)
    project.datasets.connect(dataset)

    test_img_path = '/Volumes/work/epfl/datasets/20210727_HA-FL-SAS6_Clones/projections_channel/CPAP/png/20210727_RPE1_HA-FL_S6_CloneE7_Ref_DAPI+rPOC5AF488+mHA568+gCPAP647_R1_1_MMStack_Default_max_C3.png'
    data_row = dataset.create_data_row(row_data=test_img_path)
    feature_schema_id = ontology.tools()[3].feature_schema_id

    annotations = [{
        "uuid": str(uuid.uuid4()),
        "schemaId": feature_schema_id,
        "dataRow": {
            "id": data_row.uid,
        },
        "point": {
            "x": 200,
            "y": 200
        }
    }]

    upload_job = project.upload_annotations(name="upload_predictions",
                                            annotations=annotations)
    upload_job.wait_until_done()


if __name__ == '__main__':
    main()
