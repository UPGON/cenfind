import re
from pathlib import Path

import PIL
import labelbox
import numpy as np
import tifffile as tf
from dotenv import dotenv_values
from tqdm import tqdm

from cenfind.core.data import Dataset

config = dotenv_values(".env")


def download_centrioles(label):
    """
    Collect the positions for a given label.
    """
    foci_in_label = [lab for lab in label.annotations if lab.name == "Centriole"]
    positions = []

    for lab in foci_in_label:
        # the coordinates in labelbox are (x, y) and start
        # in the top left corner;
        # thus, they correspond to (col, row).
        x = int(lab.value.x)
        y = int(lab.value.y)
        positions.append((x, y))

    return np.array(positions, dtype=int)


def download_mask(label, object_type):
    """
    Collect the nucleus masks for the label.
    return a 16bit-numpy mask with each nucleus labelled by pixel value
    """
    ALLOWED_OBJECT_TYPES = ["Nucleus", "Cilium"]
    if object_type not in ALLOWED_OBJECT_TYPES:
        raise ValueError("object type must be of: %s" % ALLOWED_OBJECT_TYPES)

    objects_in_label = [lab for lab in label.annotations if lab.name == object_type]

    if len(objects_in_label) == 0:
        raise ValueError("Empty list; no nucleus")

    mask_shape = objects_in_label[0].value.mask.value.shape
    res = np.zeros(mask_shape[:2], dtype="uint16")
    object_id = 1

    for lab in objects_in_label:
        try:
            mask = lab.value.mask.value[:, :, 0]
            res += ((mask / 255) * object_id).astype("uint16")
            object_id += 1
        except PIL.UnidentifiedImageError as e:
            continue

    return res


def main():
    lb = labelbox.Client(api_key=config["LABELBOX_API_KEY"])
    project = lb.get_project(config["PROJECT_CENTRIOLES"])
    labels = project.label_generator()

    centrioles = True
    nuclei = False

    for label in tqdm(labels):
        ds = label.extra["Dataset Name"]

        if ds != "cilia":
            continue

        path_dataset = Path(f'data/{ds}')
        projection_suffix = ""
        dataset = Dataset(
            path_dataset, image_type=".tif", projection_suffix=projection_suffix
        )

        external_name = label.data.external_id
        extension = external_name.split(".")[-1]

        print("Processing %s / %s" % (path_dataset, external_name))

        if centrioles:
            annotation_name = re.sub(f".{extension}$", ".txt", external_name)
            dst_centrioles = dataset.path_annotations_centrioles / annotation_name
            try:
                positions = download_centrioles(label)
                np.savetxt(dst_centrioles, positions, delimiter=",", fmt="%u")
                print(
                    "Saving centriole positions of %s to %s"
                    % (external_name, dst_centrioles)
                )
            except FileExistsError:
                print(
                    "Skipping centriole positions for %s to %s"
                    % (external_name, dst_centrioles)
                )
                continue

        if nuclei:
            mask_name = re.sub(
                f"C\d\.{extension}$", f"{projection_suffix}_C0.tif", external_name
            )
            dst_nuclei = dataset.path_annotations_cells / mask_name
            try:
                mask = download_mask(label, "Nucleus")
                tf.imwrite(dst_nuclei, mask)
                print("Saving mask of %s to %s" % (external_name, dst_nuclei))
            except FileExistsError:
                print("Skipping mask of %s to %s" % (external_name, dst_nuclei))
                continue

    print("FINISHED")


if __name__ == "__main__":
    main()
