from operator import itemgetter
from pathlib import Path

import cv2
import numpy as np
import tifffile as tf

from utils import (
    labelbox_annotation_load,
    label_coordinates,
    image_8bit_contrast,
    channel_extract,
    mask_create_from_contours,
    cnt_centre, image_tint,
)

from vision import (
    nuclei_segment,
    foci_process,
    centrosomes_box,
)


def color_scale(color):
    color = np.array(color, dtype=float)
    return color / 255


# Foci
RED_BGR = (44, 36, 187)
RED_BGR_SCALED = color_scale(RED_BGR)

# Foci label
VIOLET_BGR = (251, 112, 154)
VIOLET_BGR_SCALED = color_scale(VIOLET_BGR)

# Nuclei
BLUE_BGR = (251, 139, 94)
BLUE_BGR_SCALED = color_scale(BLUE_BGR)

# Annotation
WHITE = (255, 255, 255)

ANNOTATED = False
LEGEND = False
GROUND_TRUTH = True

ADD_FOCI = True
ADD_CENT = True
ADD_NUCLEI = True


def main():
    path_root = Path('/Volumes/work/datasets')

    # dataset_name = '20210709_RPE1_deltS6_Lentis_HA-DM4_B3_pCW571_48hDOX_rCep63_mHA_gCPAP_1'
    # fov_name = f'{dataset_name}_MMStack_Default_max.ome.tif'

    datasets_test = [
        'RPE1wt_CEP63+CETN2+PCNT_1',
        'U2OS_CEP63+SAS6+PCNT_1',
        'RPE1wt_CEP152+GTU88+PCNT_1',
    ]

    dataset_name = datasets_test[0]
    channel_id = 1
    x, y = 0, 0
    fov_name = f'{dataset_name}_{x:03}_{y:03}_max.ome.tif'
    path_projected = path_root / f'{dataset_name}' / 'projections' / fov_name
    path_out = path_root / dataset_name / 'out'
    path_out.mkdir(exist_ok=True)

    # Data loading
    projected = tf.imread(path_projected, key=range(4))
    c, w, h = projected.shape

    # SEMANTIC ANNOTATION
    # Segment nuclei
    nuclei_raw = channel_extract(projected, 0)
    nuclei_8bit = image_8bit_contrast(nuclei_raw)
    nuclei_contours = nuclei_segment(nuclei_8bit, dest=path_out, threshold=150)

    # Detect foci
    centrioles_raw = channel_extract(projected, channel_id=channel_id)
    foci_masked, foci_coords = foci_process(centrioles_raw, ks=3, dist_foci=2,
                                            factor=3, blur_type='gaussian')
    centrioles_8bit = image_8bit_contrast(centrioles_raw)

    # Infer centrosomes
    centrosomes_bboxes = centrosomes_box(foci_masked)

    # ONTOLOGY
    # Label the centrosomes
    centrosomes_mask = np.zeros((w, h), dtype=np.uint8)
    centrosomes_labels = mask_create_from_contours(centrosomes_mask, centrosomes_bboxes)

    # Group foci into centrosomes
    cent2foci = []
    for f_id, (r, c) in enumerate(foci_coords):
        cent2foci.append((int(centrosomes_labels[r, c]), f_id))

    # Assign centrosomes to nearest nucleus
    nuclei2cent = []
    for cent_id, cnt in enumerate(centrosomes_bboxes):
        r, c = cnt_centre(cnt)
        distances = [(n_id, cv2.pointPolygonTest(nucleus, (r, c), measureDist=True))
                     for n_id, nucleus in enumerate(nuclei_contours)]
        closest = max(distances, key=itemgetter(1))[0]
        nuclei2cent.append(closest)

    # VISUALISATION
    annotation = np.zeros((w, h, 3), dtype=np.uint8)
    composite = np.zeros((w, h, 3), dtype=np.uint8)
    legend = np.zeros((w, h, 3), dtype=np.uint8)

    alpha = 1
    beta = 1

    # Add nuclei
    nuclei = cv2.cvtColor(nuclei_8bit, cv2.COLOR_GRAY2BGR)
    composite = cv2.addWeighted(composite, alpha, image_tint(nuclei, BLUE_BGR_SCALED), beta, 0.0)

    if ANNOTATED and ADD_NUCLEI:
        # Draw nuclei contours
        for n_id, cnt in enumerate(nuclei_contours):
            cv2.drawContours(annotation, [cnt], 0, WHITE, thickness=2)
            if LEGEND:
                cv2.putText(legend, f'N{n_id}', org=cnt_centre(cnt), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=.8, thickness=2, color=WHITE)

    # Draw foci coords
    for f_id, (r, c) in enumerate(foci_coords):
        cv2.drawMarker(annotation, position=(r, c), color=WHITE,
                       markerType=cv2.MARKER_DIAMOND, markerSize=20)
    if ADD_FOCI:
        # Add foci
        foci = cv2.cvtColor(foci_masked, cv2.COLOR_GRAY2BGR)
        centrioles_contrast = cv2.cvtColor(centrioles_8bit, cv2.COLOR_GRAY2BGR)

        composite = cv2.addWeighted(composite, alpha, image_tint(foci, RED_BGR_SCALED), beta, 0.0)
        composite = cv2.addWeighted(composite, alpha, image_tint(centrioles_contrast, RED_BGR_SCALED), beta, 0.0)

        # Draw ground truth
        if GROUND_TRUTH:
            try:
                labels = labelbox_annotation_load(path_root / dataset_name / 'annotation.json',
                                                  f'{dataset_name}_{x:03}_{y:03}_max_C{channel_id}.png')
                for i, label in enumerate(labels):
                    x, y = label_coordinates(label)
                    x, y = int(x), int(y)
                    cv2.drawMarker(annotation, position=(x, y), color=WHITE,
                                   markerType=cv2.MARKER_SQUARE, markerSize=20)
            except IndexError:
                pass

    if ADD_CENT and ANNOTATED:
        # Draw centrosome boxes
        cv2.drawContours(annotation, centrosomes_bboxes, -1, WHITE)

    if LEGEND:
        for c_id, cnt in enumerate(centrosomes_bboxes):
            r, c = cnt_centre(cnt)
            cv2.putText(annotation, f'C{c_id}', org=(r + 10, c), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.8, thickness=2, color=WHITE)

    path_crop = path_root / dataset_name / 'crops'

    path_crop.mkdir(exist_ok=True)

    # foci_for_crops = cv2.cvtColor(image_8bit_contrast(centrioles_raw), cv2.COLOR_GRAY2BGR)

    for cent_id, cent_cnt in enumerate(centrosomes_bboxes):
        c, r = cnt_centre(cent_cnt)
        width = 32

        if r <= width:
            r = width
        if c <= width:
            c = width

        r_start = r - width
        r_stop = r + width
        c_start = c - width
        c_stop = c + width

        crop_composite = foci_masked[r_start:r_stop, c_start:c_stop]
        crop_composite = image_8bit_contrast(crop_composite)
        crop_composite = cv2.cvtColor(crop_composite, cv2.COLOR_GRAY2BGR)
        crop_annotation = annotation[r_start:r_stop, c_start:c_stop]

        crop = cv2.addWeighted(image_tint(crop_composite, RED_BGR_SCALED), 1,
                               image_tint(crop_annotation,BLUE_BGR_SCALED), 1, 0.0)
        print(cent_id, crop)
        cv2.imwrite(str(path_crop / f'{fov_name.split(".")[0]}_C{channel_id}_{cent_id:03}.png'), crop)

    cv2.imwrite(str(path_out / f'{fov_name.split(".")[0]}_C{channel_id}_composite.png'), composite)
    cv2.imwrite(str(path_out / f'{fov_name.split(".")[0]}_C{channel_id}_annotated.png'), annotation)

    print(len(foci_coords))
    print(0)


if __name__ == '__main__':
    main()
