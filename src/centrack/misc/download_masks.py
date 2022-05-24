import os
from pathlib import Path

import numpy as np
import tifffile as tf
import labelbox
from dotenv import load_dotenv


def main():
    load_dotenv('/home/buergy/projects/centrack/.env')

    path_dataset = Path('/data1/centrioles/cells_rpe')

    path_train = path_dataset / 'train'
    path_train_images = path_train / 'images'
    path_train_masks = path_train / 'masks'
    path_train_images.mkdir(parents=True, exist_ok=True)
    path_train_masks.mkdir(parents=True, exist_ok=True)

    path_test = path_dataset / 'test'
    path_test_images = path_test / 'images'
    path_test_masks = path_test / 'masks'
    path_test_images.mkdir(parents=True, exist_ok=True)
    path_test_masks.mkdir(parents=True, exist_ok=True)

    # Create Labelbox client
    lb = labelbox.Client(api_key=os.environ['LABELBOX_API_KEY'])

    # Get project by ID
    project = lb.get_project('cl3cxyu0o9lnb0884fnsjea91')

    # Export image and text data as an annotation generator:
    labels = project.label_generator()

    for label in labels:
        name = label.data.external_id.replace('.png', '.tif')
        mask_multi = np.zeros((2048, 2048), dtype='uint16')
        for i, struct in enumerate(label.annotations):
            cell_mask = struct.value.mask.value[:, :, 0]
            if struct.name == 'Cell':
                mask_multi += ((cell_mask / 255) * i).astype('uint16')
        tf.imwrite(path_train_masks / name, mask_multi)


if __name__ == '__main__':
    main()
