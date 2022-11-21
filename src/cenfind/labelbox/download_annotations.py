import logging
import re

import PIL
import labelbox
import numpy as np
import tifffile as tf
from dotenv import dotenv_values
from tqdm import tqdm

from cenfind.experiments.constants import PREFIX_REMOTE

config = dotenv_values('.env')


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('logs/download.log')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    lb = labelbox.Client(api_key=config['LABELBOX_API_KEY'])
    project = lb.get_project(config['PROJECT_CENTRIOLES'])
    labels = project.label_generator()

    for label in tqdm(labels):
        ds = label.extra['Dataset Name']
        path_dataset = PREFIX_REMOTE / ds

        path_annotations = path_dataset / 'annotations'
        path_annotations_centrioles = path_annotations / 'centrioles'
        path_annotations_cells = path_annotations / 'cells'

        path_annotations.mkdir(parents=True, exist_ok=True)
        path_annotations_centrioles.mkdir(exist_ok=True)
        path_annotations_cells.mkdir(exist_ok=True)

        external_name = label.data.external_id
        channel_id = external_name.split('.')[0][-1]
        print('Processing %s / %s', ds, external_name)
        if external_name.endswith(".png"):
            annotation_name = re.sub('.png$', '.txt', external_name)
            mask_name = re.sub('C\d.png$', '_max_C0.tif', external_name)
        else:
            annotation_name = f"{external_name}_max_C{channel_id}.txt"
            mask_name = f"{external_name}_max_C0.tif"

        if not (path_annotations_centrioles / annotation_name).exists():
            print('annotation does not exist')
            foci_in_label = [lab for lab in label.annotations if lab.name == 'Centriole']
            with open(path_annotations_centrioles / annotation_name, 'w') as f:
                for lab in foci_in_label:
                    logger.info('Adding point')
                    # the coordinates in labelbox are (x, y) and start
                    # in the top left corner;
                    # thus, they correspond to (col, row).
                    x = int(lab.value.x)
                    y = int(lab.value.y)
                    f.write(f"{x},{y}\n")
        else:
            print('exists')

        if (path_annotations_cells / mask_name).exists():
            logger.info("%s already exists; skipping..." % mask_name)
            continue

        nuclei_in_label = [lab for lab in label.annotations if lab.name == 'Nucleus']
        if len(nuclei_in_label) == 0:
            logger.warning("No nucleus found in %s" % external_name)
            continue
        mask_shape = nuclei_in_label[0].value.mask.value.shape
        logger.debug('mask shape: %s' % str(mask_shape))
        res = np.zeros(mask_shape[:2], dtype='uint16')
        nucleus_id = 1
        for lab in nuclei_in_label:
            logger.info('Adding contour')
            try:
                cell_mask = lab.value.mask.value[:, :, 0]
                res += ((cell_mask / 255) * nucleus_id).astype('uint16')
                nucleus_id += 1
            except PIL.UnidentifiedImageError as e:
                logger.error('Problem with %s (%s)' % external_name, e)
                continue
        tf.imwrite(path_annotations_cells / mask_name, res)
        logger.info("%s: DONE" % external_name)
    logger.info('FINISHED')


if __name__ == '__main__':
    main()
