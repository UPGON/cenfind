import tifffile as tf
import cv2
from utils import image_8bit_contrast
from pathlib import Path
from tqdm import tqdm


def tif_split(path):
    """
    Write 8bit version for each channel of the ome.tif
    :param path:
    :return:
    """
    path_dataset = path.parent
    file_name = path.name
    file_stem = file_name.split('.')[0]

    data = tf.imread(path, key=range(4))
    for plane in range(4):
        array = image_8bit_contrast(data[plane, :, :])
        cv2.imwrite(str(path_dataset / f'{file_stem}_C{plane}.png'), array)


def main():
    path_root = Path('data/20210709_RPE1_deltS6_Lentis_HA-SAS6_FL')
    files = tuple((file for file in path_root.rglob('*.ome.tif')))
    pbar = tqdm(files)
    for file in pbar:
        pbar.set_description(desc=f'Splitting {file.name}')
        tif_split(file)


if __name__ == '__main__':
    main()
