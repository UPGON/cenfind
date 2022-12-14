import itertools
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import tifffile as tf
from tqdm import tqdm


def extract_info(pattern: re, dataset_name: str):
    res = re.match(pattern, dataset_name)
    res_dict = res.groupdict()
    markers = res_dict['markers'].split('+')
    res_dict['markers'] = tuple(markers)

    return res_dict


@dataclass
class Field:
    name: str
    dataset: 'Dataset'

    @property
    def stack(self) -> np.ndarray:
        if not self.dataset.raw.exists():
            print(f'The path {self.dataset.raw} does not exist')
            sys.exit()
        data = tf.imread(str(self.dataset.raw / f"{self.name}.ome.tif"))
        axes_order = self._axes_order()
        if axes_order == "ZCYX":
            data = np.swapaxes(data, 0, 1)
        return data

    @property
    def projection(self) -> np.ndarray:
        path_projection = self.dataset.path / 'projections' / f"{self.name}{self.dataset.projection_suffix}.tif"

        with tf.TiffFile(str(path_projection)) as tif:
            res = tif.asarray()
        return res

    def channel(self, channel: int) -> np.ndarray:
        return self.projection[channel, :, :]

    def annotation(self, channel) -> np.ndarray:
        """
        Load annotation file from text file given channel
        loaded as row col, row major.
        ! the text format is x, y; origin at top left;
        :param channel:
        :return:
        """
        name = f"{self.name}{self.dataset.projection_suffix}_C{channel}"
        path_annotation = self.dataset.path / 'annotations' / 'centrioles' / f"{name}.txt"
        if path_annotation.exists():
            annotation = np.loadtxt(str(path_annotation), dtype=int, delimiter=',')
            return annotation[:, [1, 0]]
        else:
            raise FileNotFoundError(f"{path_annotation}")

    def mask(self, channel) -> np.ndarray:
        mask_name = f"{self.name}{self.dataset.projection_suffix}_C{channel}.tif"
        path_annotation = self.dataset.path / 'annotations' / 'cells' / mask_name
        if path_annotation.exists():
            return tf.imread(str(path_annotation))
        else:
            raise FileNotFoundError(path_annotation)

    def _axes_order(self) -> str:
        """
        Return a string of the form 'ZYCX' or 'CZYX'
        :return:
        """
        path_raw = str(self.dataset.path / 'raw' / f"{self.name}.ome.tif")
        with tf.TiffFile(path_raw) as tif:
            try:
                order = tif.series[0].axes
            except ValueError(f"Could not retrieve metadata for axes order for {path_raw}"):
                order = None

        return order


@dataclass
class Dataset:
    """
    Represent a dataset structure
    """
    path: Union[str, Path]
    image_type: str = '.ome.tif'
    projection_suffix: str = '_max'
    pixel_size: float = .1025

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.raw = self.path / 'raw'
        self.projections = self.path / 'projections'
        self.predictions = self.path / 'predictions'
        self.visualisation = self.path / 'visualisations'
        self.statistics = self.path / 'statistics'
        self.vignettes = self.path / 'vignettes'

    @property
    def fields(self):
        fields_path = self.path / 'fields.txt'
        with open(fields_path, 'r') as f:
            fields_list = f.read().splitlines()
        return [Field(field_path, self) for field_path in fields_list]

    def pairs(self, split: str = None, channel_id: int = None) -> List[Tuple['Field', int]]:
        """
        Fetch the fields of view for train or test
        :param channel_id:
        :param split: all, test or train
        :return: a list of tuples (fov name, channel id)
        """

        if split is None:
            return self.read_split('train', channel_id) + self.read_split('test', channel_id)
        else:
            return self.read_split(split, channel_id)

    def write_fields(self):
        """
        Write field names to fields.txt.
        """
        if (self.path / 'raw').exists():
            folder = self.path / 'raw'
        elif (self.path / 'projections').exists():
            folder = self.path / 'projections'
        else:
            raise FileNotFoundError(self.path)

        fields = []
        for f in folder.iterdir():
            if f.name.startswith('.'):
                continue
            fields.append(f.name.split('.')[0].rstrip(self.projection_suffix))

        with open(self.path / 'fields.txt', 'w') as f:
            for field in fields:
                f.write(field + '\n')

    def write_projections(self, axis=1):
        for field in tqdm(self.fields):
            projection = field.stack.max(axis)
            tf.imwrite(self.projections / f"{field.name}{self.projection_suffix}.tif",
                       projection,
                       photometric='minisblack',
                       imagej=True,
                       resolution=(1 / self.pixel_size, 1 / self.pixel_size),
                       metadata={'unit': 'um'})

    def write_train_test(self, channels: list):
        train_fields, test_fields = split_pairs(self.fields, p=.9)
        pairs_train = choose_channel(train_fields, channels)
        pairs_test = choose_channel(test_fields, channels)

        with open(self.path / 'train.txt', 'w') as f:
            for fov, channel in pairs_train:
                f.write(f"{fov.name},{channel}\n")

        with open(self.path / 'test.txt', 'w') as f:
            for fov, channel in pairs_test:
                f.write(f"{fov.name},{channel}\n")

    def read_split(self, split_type, channel_id=None) -> List[Tuple[Field, int]]:
        with open(self.path / f'{split_type}.txt', 'r') as f:
            files = f.read().splitlines()

        files = [f.split(',') for f in files if f]
        if channel_id:
            return [(Field(str(f[0]), self), int(channel_id)) for f in files]
        else:
            return [(Field(str(f[0]), self), int(f[1])) for f in files]


def split_pairs(fields: list[Field], p=.9) -> tuple[list[Field], list[Field]]:
    """
    Split a list of pairs (field, channel).

    :param fields
    :param p the train proportion, default to .9
    :return train_split, test_split
    """

    random.seed(1993)
    size = len(fields)
    split_idx = int(p * size)
    shuffled = random.sample(fields, k=size)
    split_test = shuffled[split_idx:]
    split_train = shuffled[:split_idx]

    return split_train, split_test


def choose_channel(fields: list[Field], channels: list[int]) -> list[tuple[Field, int]]:
    """Assign channel to field."""
    return [(field, int(channel)) for field, channel in itertools.product(fields, channels)]
