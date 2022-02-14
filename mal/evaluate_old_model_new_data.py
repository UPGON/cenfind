import json
import logging
from pathlib import Path

from cv2 import cv2
import labelbox
import requests

from centrack.evaluate import process_one_image
from centrack.outline import Centre

logging.basicConfig(level=logging.INFO)

with open('../configs/labelbox_api_key.txt', 'r') as apikey:
    LB_API_KEY = apikey.readline()
    LB_API_KEY = LB_API_KEY.rstrip('\n')

project_name = '20210727_HA-FL-SAS6_Clones'
project_uid = 'ckz8af4ss60300zg53zyd92ue'
path_root = Path('/Volumes/work/epfl/datasets')
path_dataset = path_root / project_name
tmp_dir = path_dataset / 'tmp_lb_images'
tmp_dir.mkdir(exist_ok=True)


def main():
    lb = labelbox.Client(api_key=LB_API_KEY)
    project = lb.get_project(project_uid)
    labels = project.label_generator()
    labels_list = labels.as_list()
    performances = []
    for label in labels_list:
        label_uid = label.uid
        data = requests.get(label.data.url).content

        with open(tmp_dir / f"{label_uid}.png", 'wb') as f:
            f.write(data)
        image = cv2.imread(str(tmp_dir / f"{label_uid}.png"), cv2.IMREAD_GRAYSCALE)
        annotation = label.annotations

        annotation_list = []
        with open(tmp_dir / f"{label_uid}.csv", 'w') as f:
            for annot in annotation:
                x, y = int(annot.value.x), int(annot.value.y)
                f.write(f'{x}, {y}\n')
                annotation_list.append(Centre((x, y)))

        performances.append(process_one_image(image, annotation_list))
    with open(tmp_dir / f"precision_recall.txt", 'w') as f:
        json.dump(performances, f)


if __name__ == '__main__':
    main()
