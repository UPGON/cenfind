from pathlib import Path
import cv2
import numpy as np
import tifffile as tf
from utils import image_8bit_contrast, labelbox_annotation_load, label_coordinates
from centrack import centrioles_detect


def main():
    path_dataset = Path('/Volumes/work/datasets/RPE1wt_CEP63+CETN2+PCNT_1')
    image_name = 'RPE1wt_CEP63+CETN2+PCNT_1_000_000.tif'
    image_stem = image_name.split('.')[0]

    image_cetn2 = tf.imread(path_dataset / 'projected'/ image_name, key=2)
    image_cetn2 = cv2.medianBlur(image_cetn2, 5)
    original = image_8bit_contrast(image_cetn2)

    foci_coords = centrioles_detect(image_cetn2, 1500, 1)

    image_centrioles = image_8bit_contrast(image_cetn2)
    image_centrosomes = np.zeros(image_cetn2.shape, np.uint8)

    for i, (r, c) in enumerate(foci_coords):
        cv2.circle(image_centrosomes, (c, r), 20, 255, thickness=-1)
        cv2.drawMarker(image_centrioles, (c, r), 100, cv2.MARKER_CROSS, 10, 2)

    labels = labelbox_annotation_load(path_dataset / 'annotation.json', image_stem + '.png')

    for i, (r, c) in enumerate(foci_coords):
        cv2.drawMarker(image_centrosomes, (c, r), 255, cv2.MARKER_CROSS, 10, 2)

    for i, label in enumerate(labels):
        x, y = label_coordinates(label)
        x, y = int(x), int(y)
        cv2.circle(image_centrosomes, (x, y), 10, 100, 2)
        cv2.putText(img=image_centrosomes, text=str(i), org=(x + 20, y + 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.6, thickness=2, color=150)

    cv2.imwrite(str(path_dataset / 'out' / f'{image_stem}_vis.png'), original)
    cv2.imwrite(str(path_dataset / 'out' / f'{image_stem}_detected.png'), image_centrioles)
    cv2.imwrite(str(path_dataset / 'out' / f'{image_stem}_centrosome.png'), image_centrosomes)


if __name__ == '__main__':
    main()