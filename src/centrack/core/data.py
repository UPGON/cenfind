import re
import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import tifffile as tf


def extract_info(pattern: re, dataset_name: str):
    res = re.match(pattern, dataset_name).groupdict()
    markers = res['markers'].split('+')
    res['markers'] = tuple(markers)

    return res


@dataclass
class Dataset:
    """
    Represents a dataset structure
    """
    path: Union[str, Path]
    image_type: str = '.ome.tif'

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.projections = self.path / 'projections'
        self.projections.mkdir(exist_ok=True)

        self.predictions = self.path / 'predictions'
        self.predictions.mkdir(exist_ok=True)

        self.visualisation = self.path / 'visualisations'
        self.visualisation.mkdir(exist_ok=True)

        self.statistics = self.path / 'statistics'
        self.statistics.mkdir(exist_ok=True)

    def splits_for(self, split_type: str) -> List[Tuple[str, int]]:
        """
        Fetch the fields of view for train or test
        :param split_type:
        :return: a list of tuples (fov name, channel id)
        """
        splits = self.path / f'{split_type}.txt'
        if not splits.exists():
            raise FileNotFoundError(f"{splits} not found. Run train_test.py")

        with open(splits, 'r') as f:
            files = f.read().splitlines()
        files = [f.split(',') for f in files if f]
        files = [(str(f[0]), int(f[1])) for f in files]
        return files

    def fields(self) -> List[str]:
        """
        Read field names from fields.txt.
        """
        if not (self.path / 'fields.txt').exists():
            self.write_fields()
        with open(self.path / 'fields.txt', 'r') as f:
            fields = [line.rstrip() for line in f.readlines()]

        return [field for field in fields]

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

            fields.append(f.name.split('.')[0].rstrip('_max'))

        with open(self.path / 'fields.txt', 'w') as f:
            for field in fields:
                f.write(field + '\n')

    def split_train_test(self, channels: List[int], p=.9) -> Tuple[List, List]:
        """
        Assign the FOV between train and test
        :param channels:
        :param p: the fraction of train examples, by default .9
        :return: a tuple of lists
        """
        random.seed(1993)
        items = self.fields()
        size = len(items)
        split_idx = int(p * size)
        shuffled = random.sample(items, k=size)
        split_test = shuffled[split_idx:]
        split_train = shuffled[:split_idx]

        train_pairs = [(fov, channel)
                       for fov, channel in zip(split_train, itertools.cycle(channels))]
        test_pairs = [(fov, channel)
                      for fov, channel in zip(split_test, itertools.cycle(channels))]
        return train_pairs, test_pairs


@dataclass
class Field:
    name: str
    dataset: Dataset

    @property
    def dataset(self) -> Dataset:
        return self.dataset

    @property
    def stack(self) -> np.ndarray:
        return tf.imread(str(self.dataset.path / 'raw' / f"{self.name}.ome.tif"))

    @property
    def projection(self) -> np.ndarray:
        return tf.imread(str(self.dataset.path / 'projections' / f"{self.name}_max.tif"))

    def channel(self, channel: int) -> np.ndarray:
        return self.projection[channel, :, :]

    def annotation(self, channel) -> np.ndarray:
        name = f"{self.name}_max_C{channel}"
        path_annotation = self.dataset.path / 'annotation' / 'centrioles' / f"{name}.txt"
        if path_annotation.exists():
            annotation = np.loadtxt(str(path_annotation), dtype=int, delimiter=',')
            return annotation
        else:
            raise FileNotFoundError(f"{path_annotation}")

    def mask(self, channel) -> np.ndarray:
        nuclei_name = re.sub('_C\d', f'_C{channel}', self.name)
        path_annotation = self.dataset.path / 'annotation' / 'cells' / f"{nuclei_name}.tif"
        if path_annotation.exists():
            return tf.imread(str(path_annotation))
        else:
            raise FileNotFoundError(path_annotation)
