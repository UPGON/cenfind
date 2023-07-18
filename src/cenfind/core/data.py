from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import pytomlpp
import tifffile as tf
from cenfind.core.log import get_logger

logger = get_logger(__name__)


@dataclass
class Field:
    name: str
    dataset: "Dataset"

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


@dataclass
class Dataset:
    """
    Represent a dataset structure
    """

    path: Union[str, Path]
    projection_suffix: str = None
    image_type: str = ".tif"
    pixel_size: float = 0.1025
    has_projections: bool = False
    is_setup: bool = False

    def __post_init__(self):
        self.path = Path(self.path)

        if not self.path.is_dir():
            logger.error(f"Dataset does not exist ({self.path})")
            raise FileNotFoundError

        self.logs = self.path / "logs"
        self.projections = self.path / "projections"
        self.predictions = self.path / "predictions"
        self.visualisation = self.path / "visualisations"
        self.statistics = self.path / "statistics"
        self.path_annotations = self.path / "annotations"

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
        if not self.projections.is_dir():
            logger.error("Projection folder not found")
            raise FileNotFoundError()

        self.logs.mkdir(exist_ok=True)

        self.predictions.mkdir(exist_ok=True)
        self.visualisation.mkdir(exist_ok=True)
        self.statistics.mkdir(exist_ok=True)

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
        folder = self.projections

        def _field_name(file_name: Path):
            return file_name.name.split('.')[0].rstrip(self.projection_suffix)

        fields = [
            _field_name(f)
            for f in folder.iterdir()
            if not f.name.startswith(".")
        ]

        with open(self.path / "fields.txt", "w") as f:
            for field in sorted(fields):
                f.write(field + "\n")
