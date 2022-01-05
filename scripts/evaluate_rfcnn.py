from pathlib import Path
import logging

import cv2
import pandas as pd
import tifffile as tf
from centrack.data import contrast

logging.basicConfig(level=logging.INFO)

path_root = Path('/Users/leoburgy/Dropbox/epfl/centriole_detection')

path_dataset = path_root / 'test'
path_annotation = path_root / 'annotations.csv'
path_pred_annotation = path_root / 'predictions/annotations'
path_pred_vis = path_root / 'predictions/visualisation'
path_pred_vis.mkdir(exist_ok=True)


def main():
    annot_actual = pd.read_csv(path_annotation)
    for image in path_dataset.iterdir():
        name_file = image.stem
        logging.info(name_file)
        annot_image_actual = annot_actual.loc[annot_actual['image_name'] == name_file]
        annot_image_pred = pd.read_csv(path_pred_annotation / (name_file + '.tif.csv'))
        plane = tf.imread(path_dataset / (name_file + '.tif'))
        plane = contrast(plane)

        for f, (x, y) in annot_image_pred.iterrows():
            cv2.drawMarker(plane, (x, y), 255, markerType=cv2.MARKER_CROSS, markerSize=10)

        for f, row in annot_image_actual.iterrows():
            x, y = row.loc[['x', 'y']]
            logging.info(f"{x}, {y}")
            cv2.drawMarker(plane, (x, y), 255, markerType=cv2.MARKER_SQUARE, markerSize=10)

        tf.imwrite(path_pred_vis / (name_file + '_annot.png'), plane)


if __name__ == '__main__':
    main()
