import argparse
import sys
from pathlib import Path
import shutil
import re
from os import scandir

import numpy as np
import xmltodict
import tifffile as tf
from tqdm import tqdm


def setup(path_dataset):
    """
    Check that the path exists and create a projections directory.

    :param path_dataset: The path to the dataset home directory e.g. /Volumes/SSD/datasets/RPE1wt_CEP63+CETN2+PCNT_1/
    :return:
    """
    path_dataset = Path(path_dataset)

    if not path_dataset.exists():
        raise FileNotFoundError(path_dataset)

    path_projections = path_dataset / 'projections'

    if not path_projections.exists():
        path_projections.mkdir(parents=True)


def images_collect(path_dataset, extension):
    """
    Collect all files with the given extension present in the path_dataset recursively.

    :param path_dataset:
    :param extension:
    :return:
    """
    files = sorted(tuple(file for file in path_dataset.rglob(f'*{extension}')
                         if not file.stem.startswith('.')))

    if len(files):
        return files
    else:
        raise FileNotFoundError(f'No {extension}-file was found under {path_dataset}')


def sharp_planes(array, reference_channel, threshold):
    """
    Compute the sharpness of the planes and max-project planes above threshold.

    :param: (4D Numpy array): CZYX stack to max-project
    :param: (integer): std threshold above which plane is considered sharp
    :return: array (3D): the stack containing the max-projection of each channel
    """

    profile = array[reference_channel, :, :, :].std(axis=(1, 2))

    if any(plane > threshold for plane in profile):
        projected = array[:, profile > threshold, :, :].max(axis=1)
    else:
        projected = array.max(axis=1)

    return profile, projected


def filename_split(file_name):
    pattern = re.compile(r'([\w\+]+)_([\w\d\+]+)_MMStack_\d+-Pos_(\d+)_(\d+).ome.tif')
    res = re.match(pattern, file_name)
    return res


def data_project(file, sharp_only, threshold):
    with tf.TiffFile(file) as data:
        stack = data.asarray()
        ome_metadata = data.ome_metadata

        # This is really important as fields of view sort their axes
        # differently depending on whether they are master or not.
        axes_order = data.series[0].get_axes()
        if axes_order == 'ZCYX':
            return stack.max(axis=0)
        elif axes_order == 'CZYX':
            return stack.max(axis=1)
        else:
            raise IndexError(axes_order)


def args_parse():
    parser = argparse.ArgumentParser(description='Collect path to dataset')
    parser.add_argument('-p', '--path', type=str, default='.',
                        help='The absolute path to the dataset folder')
    parser.add_argument('-e', '--extension', type=str, default='.ome.tif',
                        help='The absolute path to the dataset folder')

    return vars(parser.parse_args())


def main(args=None):
    if args:
        path_dataset = Path(args['path'])
        extension = args['extension']
    else:
        path_dataset = Path(input("Please enter the absolute path to the dataset: "))
        extension = '.ome.tif'

    setup(path_dataset)

    try:
        files = images_collect(path_dataset / 'raw', extension)
    except FileExistsError:
        sys.exit()

    pbar = tqdm(files)
    for file in pbar:
        pbar.set_description(f"Loading {file.name}")
        projected = data_project(file, sharp_only=True, threshold=0)
        file_name_core = file.name.removesuffix(''.join(file.suffixes))
        dest_name = f'{file_name_core}_max.ome.tif'

        tf.imwrite(path_dataset / 'projections' / dest_name,
                   projected, photometric="minisblack")


if __name__ == '__main__':
    main()
