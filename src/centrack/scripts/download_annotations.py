import os
from pathlib import Path
import PIL
import numpy as np
import tifffile as tf
import labelbox
from dotenv import load_dotenv
import argparse

load_dotenv('/home/buergy/projects/centrack/.env')


def opts_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)

    return parser.parse_args()


def main():
    opts = opts_parse()
    path_dataset = Path(opts.path)

    path_raw = path_dataset / 'raw'
    path_projections = path_dataset / 'projections'
    path_annotations = path_dataset / 'annotations'

    for folder in [path_raw, path_projections, path_annotations]:
        folder.mkdir(parents=True, exist_ok=True)

    lb = labelbox.Client(api_key=os.environ['LABELBOX_API_KEY'])
    project = lb.get_project('cl405ug7g03zk075g26jpaj8a')

    # Foci lists
    path_annotations_centrioles = path_annotations / 'centrioles'
    path_annotations_centrioles.mkdir(exist_ok=True)

    # Cell masks
    path_annotations_cells = path_annotations / 'cells'
    path_annotations_cells.mkdir(exist_ok=True)

    labels = project.label_generator()

    for label in labels:
        # Foci file generation
        foci_in_label = [l for l in label.annotations if l.name == 'Centriole']
        annotation_file = label.data.external_id.replace('.png', '.txt')
        with open(path_annotations_centrioles / annotation_file, 'w') as f:
            for focus in foci_in_label:
                x = int(focus.value.x)
                y = int(focus.value.y)
                f.write(f"{x},{y}\n")

        # Cell mask generation
        mask_multi = np.zeros((2048, 2048), dtype='uint16')
        for i, struct in enumerate(label.annotations):
            if struct.name == 'Cell':
                try:
                    cell_mask = struct.value.mask.value[:, :, 0]
                    mask_multi += ((cell_mask / 255) * i).astype('uint16')
                except PIL.UnidentifiedImageError as e:
                    print(f'Problem with {label} ({e})')
                    continue

        name = label.data.external_id.replace('.png', '.tif')
        tf.imwrite(path_annotations_cells / name, mask_multi)


if __name__ == '__main__':
    main()
