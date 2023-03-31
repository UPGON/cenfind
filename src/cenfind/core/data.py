import itertools
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pytomlpp
import tifffile as tf
from tqdm import tqdm
from cenfind.core.log import get_logger

logger = get_logger(__name__)


@dataclass
class Field:
    name: str
    dataset: "Dataset"

    @property
    def stack(self) -> np.ndarray:
        path_file = str(self.dataset.raw / f"{self.name}{self.dataset.image_type}")

        try:
            data = tf.imread(path_file)
        except FileNotFoundError:
            logger.error(f"File not found (%s)" % path_file)
            raise

        axes_order = self._axes_order()
        if axes_order == "ZCYX":
            data = np.swapaxes(data, 0, 1)
        return data

    @property
    def projection(self) -> np.ndarray:
        _projection_name = f"{self.name}{self.dataset.projection_suffix}.tif"

        path_projection = self.dataset.projections / _projection_name
        try:
            res = tf.imread(str(path_projection))
            return res
        except FileNotFoundError:
            logger.error("File not found (%s). Check the projection suffix..." % path_projection)
            raise

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
        path_annotation = (
                self.dataset.path / "annotations" / "centrioles" / f"{name}.txt"
        )
        try:
            annotation = np.loadtxt(str(path_annotation), dtype=int, delimiter=",")
            if len(annotation) == 0:
                return annotation
            else:
                return annotation[:, [1, 0]]
        except OSError:
            logger.error(f"No annotation found for %s" % path_annotation, exc_info=True)
            raise

    def mask(self, channel) -> np.ndarray:
        mask_name = f"{self.name}{self.dataset.projection_suffix}_C{channel}.tif"
        path_annotation = self.dataset.path / "annotations" / "cells" / mask_name
        if path_annotation.exists():
            return tf.imread(str(path_annotation))
        else:
            raise FileNotFoundError(path_annotation)

    def _axes_order(self) -> str:
        """
        Return a string of the form 'ZYCX' or 'CZYX'
        :return:
        """
        path_raw = str(
            self.dataset.path / "raw" / f"{self.name}{self.dataset.image_type}"
        )
        with tf.TiffFile(path_raw) as tif:
            try:
                order = tif.series[0].axes
            except ValueError(
                    f"Could not retrieve metadata for axes order for {path_raw}"
            ):
                order = None

        return order


@dataclass
class Dataset:
    """
    Represent a dataset structure
    """

    path: Union[str, Path]
    projection_suffix: str = None
    image_type: str = ".ome.tif"
    pixel_size: float = 0.1025
    has_projections: bool = False
    is_setup: bool = False

    def __post_init__(self):
        self.path = Path(self.path)

        if not self.path.is_dir():
            logger.error(f"Dataset does not exist ({self.path})")
            raise FileNotFoundError

        self.raw = self.path / "raw"
        self.projections = self.path / "projections"
        self.predictions = self.path / "predictions"
        self.visualisation = self.path / "visualisations"
        self.statistics = self.path / "statistics"
        self.vignettes = self.path / "vignettes"
        self.logs = self.path / "logs"
        self.path_annotations = self.path / "annotations"
        self.path_annotations_centrioles = self.path_annotations / "centrioles"
        self.path_annotations_cells = self.path_annotations / "cells"

        if self.projection_suffix is None:
            try:
                metadata = pytomlpp.load(self.path / "metadata.toml")
                self.projection_suffix = metadata["projection_suffix"]
            except FileNotFoundError:
                logger.error("Metadata file not found (%s)" % str(self.path / "metadata.toml"))
                raise

    def setup(self) -> None:
        """
        Create folders for projections, predictions, statistics, visualisation and vignettes.
        Collect field names into fields.txt
        """
        self.projections.mkdir(exist_ok=True)
        self.predictions.mkdir(exist_ok=True)
        (self.predictions / "centrioles").mkdir(exist_ok=True)
        (self.predictions / "nuclei").mkdir(exist_ok=True)
        self.statistics.mkdir(exist_ok=True)
        self.visualisation.mkdir(exist_ok=True)
        self.vignettes.mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        self.path_annotations.mkdir(parents=True, exist_ok=True)
        self.path_annotations_centrioles.mkdir(exist_ok=True)
        self.path_annotations_cells.mkdir(exist_ok=True)

        self.has_projections = bool(len([f for f in self.projections.iterdir()]))
        self.is_setup = True

    @property
    def fields(self) -> List[Field]:
        """
        Construct a list of Fields using the fields listed in fields.txt.
        """
        fields_path = self.path / "fields.txt"
        with open(fields_path, "r") as f:
            fields_list = f.read().splitlines()
        return [Field(field_name, self) for field_name in fields_list]

    def write_fields(self) -> None:
        """
        Write field names to fields.txt.
        """
        if not self.is_setup:
            self.setup()

        if self.has_projections:
            folder = self.projections
        else:
            folder = self.raw

        def _field_name(file_name: str):
            return file_name.split(".")[0].rstrip(self.projection_suffix)

        fields = [
            _field_name(str(f.name))
            for f in folder.iterdir()
            if not str(f).startswith(".")
        ]

        with open(self.path / "fields.txt", "w") as f:
            for field in fields:
                f.write(field + "\n")

    def write_projections(self, axis=1) -> None:
        for field in tqdm(self.fields):
            projection = field.stack.max(axis)
            tf.imwrite(
                self.projections / f"{field.name}{self.projection_suffix}.tif",
                projection,
                photometric="minisblack",
                imagej=True,
                resolution=(1 / self.pixel_size, 1 / self.pixel_size),
                metadata={"unit": "um"},
            )
        self.has_projections = True

    def split_pairs(self, p=0.9, seed=1993) -> tuple[list[Field], list[Field]]:
        """
        Split a list of pairs (field, channel).

        :param fields
        :param p the train proportion, default to .9
        :param seed the seed to reproduce the splitting
        :return train_split, test_split
        """

        random.seed(seed)
        size = len(self.fields)
        split_idx = int(p * size)
        shuffled = random.sample(self.fields, k=size)
        split_test = shuffled[split_idx:]
        split_train = shuffled[:split_idx]

        return split_train, split_test

    def pairs(
            self, split: str = None, channel_id: int = None
    ) -> List[Tuple["Field", int]]:
        """
        Fetch the fields of view for train or test
        :param channel_id:
        :param split: all, test or train
        :return: a list of tuples (fov name, channel id)
        """

        if split is None:
            return self.read_split("train", channel_id) + self.read_split(
                "test", channel_id
            )
        else:
            return self.read_split(split, channel_id)

    def read_split(self, split_type, channel_id=None) -> List[Tuple[Field, int]]:
        with open(self.path / f"{split_type}.txt", "r") as f:
            files = f.read().splitlines()

        files = [f.split(",") for f in files if f]
        if channel_id:
            return [(Field(str(f[0]), self), int(channel_id)) for f in files]
        else:
            return [(Field(str(f[0]), self), int(f[1])) for f in files]


def choose_channel(fields: list[Field], channels: list[int]) -> list[tuple[Field, int]]:
    """Pick a channel for each field."""
    return [
        (field, int(channel)) for field, channel in itertools.product(fields, channels)
    ]
