import sys

import tifffile as tf
import cv2
from utils import image_8bit_contrast, filename_split
from pathlib import Path
from tqdm import tqdm


def tif_split(path_file, path_dst):
    """
    Write 8bit version for each channel of the ome.tif
    :param path:
    :return:
    """

    file_name = path_file.name
    file_stem = file_name.split('.')[0]

    info_tuple = filename_split(file_name)
    genotype, markers, replicate, posx, posy = info_tuple.groups()
    markers_list = markers.split('+')
    markers_list.insert(0, 'DAPI')

    data = tf.imread(path_file, key=range(4))
    c, y, x = data.shape
    for channel in range(c):
        array = image_8bit_contrast(data[channel, :, :])
        cv2.putText(array, f'Genotype: {genotype} Markers: {"+".join(markers_list)} FOV: {posx} / {posy} Plane: {markers_list[channel]}',
                    org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.8, thickness=2, color=255)
        cv2.imwrite(str(path_dst / f'{file_stem}_C{channel}.png'), array)


def main():
    path_dataset = Path(input('Specify the absolute dataset path: '))
    if not path_dataset.exists():
        sys.exit(f'Invalid path {path_dataset}')
    path_channels = path_dataset / 'channels'
    path_channels.mkdir(exist_ok=True)

    files = tuple((file for file in path_dataset.rglob('*_max.ome.tif')))
    pbar = tqdm(files)

    for file in pbar:
        pbar.set_description(desc=f'Splitting {file.name}')
        tif_split(file, path_channels)


if __name__ == '__main__':
    main()
