import logging
from pathlib import Path

import labelbox
import pandas as pd
import requests
from cv2 import cv2

from src.centrack.commands import process_one_image, extract_centrioles
from src.centrack.commands import Centre

logging.basicConfig(level=logging.INFO)

with open('../../../data/labelbox_api_key.txt', 'r') as apikey:
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
        image = cv2.imread(str(tmp_dir / f"{label_uid}.png"),
                           cv2.IMREAD_GRAYSCALE)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        annotation = label.annotations

        annotation_list = []
        with open(tmp_dir / f"{label_uid}.csv", 'w') as f:
            for annot in annotation:
                x, y = int(annot.value.x), int(annot.value.y)
                f.write(f'{x}, {y}\n')
                annotation_list.append(Centre((x, y)))
        predictions_list = extract_centrioles(image)
        performances.append(
            process_one_image(image, annotation_list, predictions_list))
        for c in annotation_list:
            c.draw(image_bgr, marker_type=cv2.MARKER_SQUARE, marker_size=4,
                   annotation=False)
        for c in predictions_list:
            c.draw(image_bgr, marker_type=cv2.MARKER_DIAMOND, annotation=False)
        cv2.imwrite(str(tmp_dir / f"{label_uid}_annot.png"), image_bgr)
    results = pd.DataFrame(performances)
    results.to_csv(tmp_dir / 'precision_recall.csv')


if __name__ == '__main__':
    main()
