import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tifffile as tf

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


@dataclass
class DataSet:
    """
    Represents a dataset structure
    """
    path: Path

    @property
    def name(self):
        return self.path.name

    @property
    def raw(self):
        """Define the path to raw folder."""
        return self.path / 'raw'

    @property
    def projections(self):
        """Define the path to projections folder."""
        return self.path / 'projections'

    @property
    def vignettes(self):
        """Define the path to the vignettes' folder."""
        return self.path / 'vignettes'

    @property
    def annotations(self):
        return self.path / 'annotations'

    @property
    def predictions(self):
        return self.path / 'predictions'

    @property
    def visualisation(self):
        return self.path / 'visualisation'

    def _check_container(self, container_name: str, file_type: str):
        """
        Check if the folder `container_name` exists and whether it
        contains `file_type` files (recursively)
        :param name:
        :param file_type:
        :return: None
        """
        container_name = self.path / container_name

        if container_name.exists():
            files = [f for f in container_name.iterdir()]
            if len(files) == 0:
                return []
            else:
                recursive_files = fetch_files(container_name,
                                              file_type=file_type)
                return recursive_files
        else:
            container_name.mkdir()

    def check_raw(self):
        self._check_container('raw', '.ome.tif')

    def check_projections(self):
        self._check_container('projections', '_max.tif')

    def check_predictions(self, force=False):
        """
        Check for a set of predictions.
        else if we want to compare the predictions with annotations,
        Compute the predictions.
        Upload the predictions to labelbox.
        Provide the url of the annotated ds on Labelbox
        """
        for image in self.projections.iterdir():
            print(image)

    def check_annotations(self):
        """
        If there is no annotation present, we should fetch them from labelbox.
        If the ds is not on labelbox, we should upload it with the predictions
        :return:
        """
        raise NotImplementedError

    def splits(self, p=.9, suffix='.ome.tif') -> Tuple[List, List]:
        """
        Assign the FOV between train and test.
        :param p: the fraction of train examples, by default .9
        :param suffix: the type of raw files, by default .ome.tif
        :return: a tuple of lists
        """
        random.seed(1993)

        files = fetch_files(self.raw, suffix)
        file_stems = [f.name.removesuffix(suffix) for f in files]
        size = len(file_stems)
        split_idx = int(p * size)
        shuffled = random.sample(file_stems, k=size)
        split_test = shuffled[split_idx:]
        split_train = shuffled[:split_idx]
        return split_train, split_test

    def split_images_channel(self, split_type):
        with open(self.path / f'{split_type}_channels.txt', 'r') as f:
            files = f.read().splitlines()
        files = [f.split(',') for f in files if f]
        return files


def build_name(path: Path, projection_type='max') -> str:
    """
    Extract the file name, remove the suffixes and append the projection type.
    :param path:
    :param channel:
    :param projection_type: the type of projection, by default max
    :return: file name of the projection
    """
    file_name = path.name
    suffixes = ''.join(path.suffixes)
    file_name_no_suffix = file_name.removesuffix(suffixes)
    return file_name_no_suffix + f'_{projection_type}.tif'


def fetch_files(path_source: Path, file_type):
    """
    Collect all ome.tif files in a list.
    :param file_type:
    :param path_source:
    :return: A list of Path to ome.tif files
    """
    if not path_source.exists():
        raise FileExistsError(path_source)
    pattern = f'*{file_type}'
    files_generator = path_source.rglob(pattern)

    return [file for file in files_generator if not file.name.startswith('.')]


@dataclass
class FieldOfView:
    """
    Representation of a projection (CxHxW)
    """
    path: Path

    @property
    def dataset(self) -> DataSet:
        return DataSet(self.path.parent.parent)

    @property
    def name(self):
        return self.path.stem

    @property
    def data(self) -> np.array:
        return tf.imread(str(self.path))

    def __getitem__(self, item):
        return self.data[item, :, :]

    def fetch_annotation(self, channel_id):
        path_annotation = self.dataset.annotations / 'centrioles' / f"{self.name}_C{channel_id}.txt"

        try:
            annotation = np.loadtxt(path_annotation, dtype=int, delimiter=',')
        except FileNotFoundError:
            raise f'Annotation not found for {path_annotation}'

        return annotation
