from pathlib import Path
from typing import List

import numpy as np
import tifffile as tf
from attrs import define, field, validators
from cenfind.core.log import get_logger

logger = get_logger(__name__)


def path_exists(instance, attribute, value):
    if not value.exists():
        raise FileNotFoundError(f"Invalid path ({value})")


def has_projections(instance, attribute, value):
    projections = value / "projections"
    if not projections.is_dir():
        raise FileNotFoundError(f"Invalid path for Projections: ({value}).")


def is_tif(instance, attribute, value):
    if value.suffix != ".tif":
        raise ValueError(f"Not a tif file ({value})")


@define
class Field:
    """
    Represents a Field

    It is responsible for returning the field's name and to load the corresponding image.

    Attributes:
        path: path to the image file
    """

    path: Path = field(validator=[validators.instance_of(Path), path_exists, is_tif])

    @property
    def name(self) -> str:
        """
        Extracts the field name as a string without the extension.
        """
        return self.path.stem

    @property
    def data(self) -> np.ndarray:
        """
        Loads data from path as a numpy array using `tifffile.imread`.
        """
        return tf.imread(str(self.path))


@define
class Dataset:
    """
    Represents a Dataset folder.

    It iterates over the fields and provides a setup method to create the necessary subdirectories.

    Attributes:
       path: Path to the dataset folder.
    """
    path: Path = field(validator=[
        validators.instance_of(Path),
        path_exists,
        has_projections])

    @property
    def logs(self):
        return self.path / "logs"

    @property
    def projections(self):
        return self.path / "projections"

    @property
    def predictions(self):
        return self.path / "predictions"

    @property
    def annotations(self):
        return self.path / "annotations"

    @property
    def visualisation(self):
        return self.path / "visualisation"

    @property
    def statistics(self):
        return self.path / "statistics"

    @property
    def nuclei(self):
        return self.predictions / "nuclei"

    @property
    def centrioles(self):
        return self.predictions / "centrioles"

    @property
    def cilia(self):
        return self.predictions / "cilia"

    @property
    def assignment(self):
        return self.predictions / "assignment"

    def setup(self) -> None:
        """
        Creates the subdirectories inside the Dataset path if not existing.
        The directories visualisation, statistics, predictions,
        nuclei, centrioles, cilia and assignment are created.
        """

        self.logs.mkdir(exist_ok=True)
        self.visualisation.mkdir(exist_ok=True)
        self.statistics.mkdir(exist_ok=True)
        self.predictions.mkdir(exist_ok=True)
        self.annotations.mkdir(exist_ok=True)
        self.nuclei.mkdir(exist_ok=True)
        self.centrioles.mkdir(exist_ok=True)
        self.cilia.mkdir(exist_ok=True)
        self.assignment.mkdir(exist_ok=True)

    @property
    def fields(self) -> List[Field]:
        """
        Collects all Fields found in `projections` into a list and raises a ValueError if no TIF file.
        """

        result = []
        for path in self.projections.iterdir():
            if (path.suffix == '.tif') and not path.name.startswith("."):
                result.append(Field(path))

        if len(result) == 0:
            raise ValueError(f"No field found in {self.projections}")

        return result
