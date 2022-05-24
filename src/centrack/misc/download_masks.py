import os
from pathlib import Path

import numpy as np
import tifffile as tf
import labelbox
from dotenv import load_dotenv


def main():
    load_dotenv('/home/buergy/projects/centrack/.env')

    path_dataset = Path('/data1/centrioles/rpe')

    path_raw = path_dataset / 'raw'
    path_projections = path_dataset / 'projections'
    path_annotations = path_dataset / 'annotations'

    for folder in [path_raw, path_projections, path_annotations]:
        folder.mkdir(parents=True, exist_ok=True)

    path_annotations_cells = path_annotations / 'cells'
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
        tf.imwrite(path_annotations_cells / name, mask_multi)


if __name__ == '__main__':
    main()
