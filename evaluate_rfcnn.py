from pathlib import Path

import cv2
import pandas as pd
from aicsimageio import AICSImage

from centrack.data import contrast

path_dataset = Path('/Users/leoburgy/epfl/training_data')
path_annotation = Path('out/positions_foci.csv')
path_annotation_pred = Path('/Users/leoburgy/epfl/leo_results')

name_file = 'RPE1wt_CEP63+CETN2+PCNT_1_000_000_max_C2'


def main():
    annot_actual = pd.read_csv(path_annotation)
    annot_actual = annot_actual.loc[annot_actual['image_name'] == (name_file + '.png')]
    annot_pred = pd.read_csv(path_annotation_pred / (name_file + '.ome.tif.csv'))
    image = AICSImage(path_dataset / (name_file + '.ome.tif'))
    plane = contrast(image.get_image_data().squeeze())

    for f, (x, y) in annot_pred.iterrows():
        print(x, y)
        cv2.drawMarker(plane, (y, x), (255), markerType=cv2.MARKER_CROSS, markerSize=10)

    for row in annot_actual.iterrows():
        x, y = row.loc[[ 'x', 'y']]
    print(0)


if __name__ == '__main__':
    main()
