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
    projections = value / 'projections'
    if not projections.is_dir():
        raise FileNotFoundError(f"Invalid path for Projections: ({value}).")


def is_tif(instance, attribute, value):
    if value.suffix != '.tif':
        raise ValueError(f'Not a tif file ({value})')


@define
class Field:
    path: Path = field(validator=[validators.instance_of(Path), path_exists, is_tif])

    @property
    def name(self) -> str:
        """
        Return the field name as a string without the extension
        """
        return self.path.stem

    @property
    def data(self) -> np.ndarray:
        """
        Load the data into a numpy array
        """
        return tf.imread(str(self.path))


@define
class Dataset:
    """
    Represent a dataset structure
    """
    path: Path = field(validator=[
        validators.instance_of(Path),
        path_exists,
        has_projections])

    @property
    def logs(self):
        return self.path / 'logs'

    @property
    def predictions(self):
        return self.path / 'predictions'

    @property
    def visualisation(self):
        return self.path / 'visualisation'

    @property
    def statistics(self):
        return self.path / 'statistics'

    @property
    def nuclei(self):
        return self.predictions / 'nuclei'

    @property
    def centrioles(self):
        return self.predictions / 'centrioles'

    @property
    def cilia(self):
        return self.predictions / 'cilia'

    @property
    def assignment(self):
        return self.predictions / 'assignment'

    def setup(self):
        self.visualisation.mkdir(exist_ok=True)
        self.statistics.mkdir(exist_ok=True)
        self.predictions.mkdir(exist_ok=True)
        self.nuclei.mkdir(exist_ok=True)
        self.centrioles.mkdir(exist_ok=True)
        self.cilia.mkdir(exist_ok=True)
        self.assignment.mkdir(exist_ok=True)

    @property
    def fields(self) -> List[Field]:
        """
        Return a list of Fields found in projections.
        """
        path = self.path / 'projections'
        result = [Field(path) for path in path.iterdir() if
                  (path.suffix == '.tif') and not path.name.startswith('.')]
        if len(result) == 0:
            raise ValueError(f'No field found in {path}')
        return result
