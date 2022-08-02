import os
import PIL
import numpy as np
import tifffile as tf
import labelbox
from dotenv import load_dotenv
from centrack.layout.constants import PREFIX_LOCAL

load_dotenv('/Users/buergy/Dropbox/epfl/projects/centrack/.env')


def main():
    lb = labelbox.Client(api_key=os.environ['LABELBOX_API_KEY'])
    project = lb.get_project('cl64z5mv20gte07vfbsjqdtvb')  # cl5gnefndcjvi08wodvn05thr
    labels = project.label_generator()

    for label in labels:
        ds = label.extra['Dataset Name']
        path_dataset = PREFIX_LOCAL / ds
        path_raw = path_dataset / 'raw'
        path_projections = path_dataset / 'projections'
        path_annotations = path_dataset / 'annotations'

        path_raw.mkdir(parents=True, exist_ok=True)
        path_projections.mkdir(parents=True, exist_ok=True)
        path_annotations.mkdir(parents=True, exist_ok=True)

        external_name = label.data.external_id

        path_annotations_centrioles = path_annotations / 'centrioles'
        path_annotations_centrioles.mkdir(exist_ok=True)
        annotation_file = external_name.replace('.png', '.txt')
        foci_in_label = [lab for lab in label.annotations if lab.name == 'Centriole']
        if len(foci_in_label) > 0:
            nucleus_id = 1
            if not external_name.endswith('C3.png'):
                with open(path_annotations_centrioles / annotation_file, 'w') as f:
                    for lab in foci_in_label:
                        # the coordinates in labelbox are (x, y) and start on the top left corner;
                        # thus, they correspond to (col, row).
                        x = int(lab.value.x)
                        y = int(lab.value.y)
                        f.write(f"{x},{y}\n")
                    print(f'{annotation_file} written')

        nuclei_in_label = [lab for lab in label.annotations if lab.name == 'Nucleus']
        if len(nuclei_in_label) == 0:
            continue
        path_annotations_cells = path_annotations / 'cells'
        path_annotations_cells.mkdir(exist_ok=True)
        mask_name = external_name.replace('.png', '.tif')

        if (path_annotations_cells / mask_name).exists():
            print(f'File already exists {mask_name}')
            continue

        res = np.zeros((512, 512), dtype='uint16')
        for lab in nuclei_in_label:
            try:
                cell_mask = lab.value.mask.value[:, :, 0]
                res += ((cell_mask / 255) * nucleus_id).astype('uint16')
                nucleus_id += 1
            except PIL.UnidentifiedImageError as e:
                print(f'Problem with {label} ({e})')
                continue
        tf.imwrite(path_annotations_cells / mask_name, res)
        print(f'Writing mask for {mask_name}')


if __name__ == '__main__':
    main()
