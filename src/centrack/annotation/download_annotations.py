import logging
import PIL
from tqdm import tqdm
import labelbox
import numpy as np
import tifffile as tf
from dotenv import dotenv_values

from centrack.utils.constants import PREFIX_REMOTE

config = dotenv_values('/home/buergy/projects/centrack/.env')


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    lb = labelbox.Client(api_key=config['LABELBOX_API_KEY'])
    project = lb.get_project(config['PROJECT_CENTRIOLES'])
    labels = project.label_generator()

    download_nuclei_mask = False

    for label in tqdm(labels):
        ds = label.extra['Dataset Name']
        external_name = label.data.external_id
        logger.debug('Processing %s / %s', ds, external_name)

        annotation_file = external_name.replace('.png', '.txt')
        mask_name = external_name.replace('.png', '.tif')

        path_dataset = PREFIX_REMOTE / ds

        logger.info('Setting up...')
        path_annotations = path_dataset / 'annotations'
        path_annotations.mkdir(parents=True, exist_ok=True)

        path_annotations_centrioles = path_annotations / 'centrioles'
        path_annotations_centrioles.mkdir(exist_ok=True)

        path_annotations_cells = path_annotations / 'cells'
        path_annotations_cells.mkdir(exist_ok=True)
        logger.info('Done')

        logger.info('Loading labels...')
        foci_in_label = [lab for lab in label.annotations if lab.name == 'Centriole']
        logger.info('Done')

        logger.info('Writing labels...')
        with open(path_annotations_centrioles / annotation_file, 'w') as f:
            for lab in foci_in_label:
                # the coordinates in labelbox are (x, y) and start
                # on the top left corner;
                # thus, they correspond to (col, row).
                x = int(lab.value.x)
                y = int(lab.value.y)
                f.write(f"{x},{y}\n")
        logger.info('Done')

        if download_nuclei_mask:
            if (path_annotations_cells / mask_name).exists():
                logger.info(f'File already exists')
                continue

            nuclei_in_label = [lab for lab in label.annotations if lab.name == 'Nucleus']
            logger.info('Writing mask...')
            res = np.zeros((2048, 2048), dtype='uint16')
            nucleus_id = 1
            for lab in nuclei_in_label:
                try:
                    cell_mask = lab.value.mask.value[:, :, 0]
                    res += ((cell_mask / 255) * nucleus_id).astype('uint16')
                    nucleus_id += 1
                except PIL.UnidentifiedImageError as e:
                    logger.error('Problem with %s (%s)' % external_name, e)
                    continue
            tf.imwrite(path_annotations_cells / mask_name, res)
            logger.info('Done')


if __name__ == '__main__':
    main()
