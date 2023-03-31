import contextlib
import copy
import itertools
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List
from typing import Tuple, Union

import cv2
import numpy as np
import pytomlpp
import tensorflow as tf
import tifffile as tif
from csbdeep.utils import normalize
from spotipy.utils import points_matching
from stardist.models import StarDist2D
from tqdm import tqdm
import pandas as pd

from cenfind.core.log import get_logger
from cenfind.core.measure import full_in_field, assign, save_foci
from cenfind.core.outline import _color_channel

np.random.seed(1)
tf.random.set_seed(2)
from cenfind.core.outline import Centre
from cenfind.core.outline import Contour, resize_image

np.random.seed(1)
tf.random.set_seed(2)
np.random.seed(1)
tf.random.set_seed(2)

logger = get_logger(__name__)


@dataclass
class Field:
    name: str
    dataset: "Dataset"

    @property
    def stack(self) -> np.ndarray:
        path_file = str(self.dataset.raw / f"{self.name}{self.dataset.image_type}")

        try:
            data = tif.imread(path_file)
        except FileNotFoundError:
            print(f"File not found ({path_file})")
            sys.exit()

        axes_order = self._axes_order()
        if axes_order == "ZCYX":
            data = np.swapaxes(data, 0, 1)
        return data

    @property
    def projection(self) -> np.ndarray:
        _projection_name = f"{self.name}{self.dataset.projection_suffix}.tif"

        path_projection = self.dataset.projections / _projection_name
        try:
            res = tif.imread(str(path_projection))
            return res
        except FileNotFoundError as e:
            print(f"File not found ({path_projection}). Check the projection suffix...", e)
            sys.exit()

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
        path_annotation = self.dataset.annotations_centrioles / f"{name}.txt"
        try:
            annotation = np.loadtxt(str(path_annotation), dtype=int, delimiter=",")
            if len(annotation) == 0:
                return annotation
            else:
                return annotation[:, [1, 0]]
        except OSError:
            print(f"No annotation found for {path_annotation}")

    def mask(self, channel) -> np.ndarray:
        mask_name = f"{self.name}{self.dataset.projection_suffix}_C{channel}.tif"
        path_annotation = self.dataset.annotations_cells / mask_name
        if path_annotation.exists():
            return tif.imread(str(path_annotation))
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
        with tif.TiffFile(path_raw) as file:
            try:
                order = file.series[0].axes
            except ValueError(
                    f"Could not retrieve metadata for axes order for {path_raw}"
            ):
                order = None

        return order

    def extract_nuclei(
            self,
            channel: int,
            factor: int,
            model: StarDist2D = None,
            annotation=None
    ) -> List[Contour]:
        """
        Extract the nuclei from the nuclei image
        :param field:
        :param channel:
        :param factor: the factor related to pixel size
        :param model:
        :param annotation: a mask with pixels labelled for each centre

        :return: List of Contours.

        """
        if model is None:
            from stardist.models import StarDist2D
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                model = StarDist2D.from_pretrained("2D_versatile_fluo")

        if annotation is not None:
            labels = annotation
        elif model is not None:
            data = self.channel(channel)
            data_resized = resize_image(data, factor)
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                labels, _ = model.predict_instances(normalize(data_resized))
            labels = cv2.resize(
                labels, dsize=data.shape, fx=1, fy=1, interpolation=cv2.INTER_NEAREST
            )

        else:
            raise ValueError("Please provide either an annotation or a model")

        labels_id = np.unique(labels)

        cnts = []
        for nucleus_id in labels_id:
            if nucleus_id == 0:
                continue
            sub_mask = np.zeros_like(labels, dtype="uint8")
            sub_mask[labels == nucleus_id] = 1
            contour, _ = cv2.findContours(
                sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            cnts.append(contour[0])

        contours = tuple(cnts)
        contours = [
            Contour(c, "Nucleus", c_id, confidence=-1) for c_id, c in enumerate(contours)
        ]

        return contours

    def extract_centrioles(self, method, channel=None, model_path=None, ) -> list[Centre]:
        foci = method(self, foci_model_file=model_path, channel=channel)
        return foci

    def run_detection(self,
                      method,
                      annotation: np.ndarray,
                      tolerance, channel=None,
                      model_path=None) -> Tuple[list[Centre], float]:
        _foci = method(self, foci_model_file=model_path, channel=channel)
        res = points_matching(annotation, _foci, cutoff_distance=tolerance)
        f1 = np.round(res.f1, 3)
        foci = [Centre((r, c), label="Centriole") for r, c in _foci]
        return foci, f1

    def score(
            self,
            nuclei_scored,
            channel: int,
    ) -> List[dict]:
        """
        1. Detect foci in the given channels
        2. Detect nuclei
        3. Assign foci to nuclei
        :param field:
        :param nuclei_scored:
        :param channel:
        :return: list(foci, nuclei, assigned, scores)
        """
        image_shape = self.projection.shape[1:]
        scores = []
        for nucleus in nuclei_scored:
            scores.append(
                {
                    "fov": self.name,
                    "channel": channel,
                    "nucleus": nucleus.centre.to_numpy(),
                    "score": len(nucleus.centrioles),
                    "is_full": full_in_field(nucleus, image_shape, 0.05),
                }
            )
        return scores

    def field_metrics(
            self,
            channel: int,
            annotation: np.ndarray,
            predictions: np.ndarray,
            tolerance: int,
            threshold: float,
    ) -> dict:
        """
        Compute the accuracy of the prediction on one field.
        :param channel:
        :param annotation:
        :param predictions:
        :param tolerance:
        :param threshold:
        :return: dictionary of fields
        """
        if all((len(predictions), len(annotation))) > 0:
            res = points_matching(annotation, predictions, cutoff_distance=tolerance)
        else:
            logger.warning(
                "threshold: %f; detected: %d; annotated: %d... Set precision and accuracy to zero"
                % (threshold, len(predictions), len(annotation))
            )
            res = SimpleNamespace()
            res.precision = 0.0
            res.recall = 0.0
            res.f1 = 0.0
            res.tp = (0,)
            res.fp = (0,)
            res.fn = 0
        perf = {
            "dataset": self.dataset.path.name,
            "field": self.name,
            "channel": channel,
            "n_actual": len(annotation),
            "n_preds": len(predictions),
            "threshold": threshold,
            "tolerance": tolerance,
            "tp": res.tp[0] if type(res.tp) == tuple else res.tp,
            "fp": res.fp[0] if type(res.fp) == tuple else res.fp,
            "fn": res.fn[0] if type(res.fn) == tuple else res.fn,
            "precision": np.round(res.precision, 3),
            "recall": np.round(res.recall, 3),
            "f1": np.round(res.f1, 3),
        }
        return perf

    def create_vignette(self, marker_index: int, nuclei_index: int):
        """
        Normalise all markers
        Represent them as blue
        Highlight the channel in green
        :param field:
        :param nuclei_index:
        :param marker_index:
        :return:
        """
        layer_nuclei = self.channel(nuclei_index)
        layer_marker = self.channel(marker_index)

        nuclei = _color_channel(layer_nuclei, (1, 0, 0), "uint8")
        marker = _color_channel(layer_marker, (0, 1, 0), "uint8")

        res = cv2.addWeighted(marker, 1, nuclei, 0.5, 0)
        res = cv2.putText(
            res,
            f"{self.name} {marker_index}",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return res

    def visualisation(
            self,
            nuclei: list,
            centrioles: list,
            channel_centrioles: int,
            channel_nuclei: int,
    ) -> np.ndarray:
        background = self.create_vignette(
            marker_index=channel_centrioles, nuclei_index=channel_nuclei
        )

        if nuclei is None:
            return background

        for nucleus in nuclei:
            if full_in_field(nucleus, background.shape[:2], 0.05):
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            background = nucleus.draw(background, annotation=False, color=color)
            background = nucleus.centre.draw(background, annotation=False)
            for centriole in centrioles:
                background = centriole.draw(background, annotation=False)

            for centriole in nucleus.centrioles:
                cv2.arrowedLine(
                    background,
                    centriole.to_cv2(),
                    nucleus.centre.to_cv2(),
                    color=(255, 255, 255),
                    thickness=1,
                )
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        return background


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
            print(f"Dataset does not exist ({self.path})")
            sys.exit()

        self.raw = self.path / "raw"
        self.projections = self.path / "projections"
        self.vignettes = self.path / "vignettes"
        self.logs = self.path / "logs"
        self.annotations = self.path / "annotations"
        self.annotations_centrioles = self.annotations / "centrioles"
        self.annotations_cells = self.annotations / "cells"

        self.results = self.path
        self.predictions = self.results / "predictions"
        self.predictions_centrioles = self.predictions / "centrioles"
        self.predictions_cells = self.predictions / "cells"
        self.visualisation = self.results / "visualisations"
        self.statistics = self.results / "statistics"

        if self.projection_suffix is None:
            try:
                metadata = pytomlpp.load(self.path / "metadata.toml")
                self.projection_suffix = metadata["projection_suffix"]
            except FileNotFoundError as e:
                print(e)
                sys.exit()
        else:
            pytomlpp.dump({"projection_suffix": self.projection_suffix}, self.path / "metadata.toml")

    def setup(self) -> None:
        """
        Create folders for projections, predictions, statistics, visualisation and vignettes.
        Collect field names into fields.txt
        """
        self.projections.mkdir(exist_ok=True)
        self.vignettes.mkdir(exist_ok=True)
        self.annotations.mkdir(parents=True, exist_ok=True)
        self.annotations_centrioles.mkdir(exist_ok=True)
        self.annotations_cells.mkdir(exist_ok=True)
        self.results.mkdir(exist_ok=True)

        self.predictions.mkdir(exist_ok=True)
        self.predictions_centrioles.mkdir(exist_ok=True)
        self.predictions_cells.mkdir(exist_ok=True)
        self.statistics.mkdir(exist_ok=True)
        self.visualisation.mkdir(exist_ok=True)
        self.vignettes.mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        self.annotations.mkdir(parents=True, exist_ok=True)
        self.annotations_centrioles.mkdir(exist_ok=True)
        self.annotations_cells.mkdir(exist_ok=True)

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
            tif.imwrite(
                self.projections / f"{field.name}{self.projection_suffix}.tif",
                projection,
                photometric="minisblack",
                imagej=True,
                resolution=(1 / self.pixel_size, 1 / self.pixel_size),
                metadata={"unit": "um"},
            )
        self.has_projections = True

    def _split(self, p=0.9, seed=1993):
        random.seed(seed)
        size = len(self.fields)
        split_idx = int(p * size)
        shuffled = random.sample(self.fields, k=size)
        split_test = shuffled[split_idx:]
        split_train = shuffled[:split_idx]

        return split_train, split_test

    def write_splits(self, channels):
        """
        Write field names and channel to use for train and test splits.
        :param channels: Tuple of integers for the channel indices """
        train_fields, test_fields = self._split()
        pairs_train = choose_channel(train_fields, channels)
        pairs_test = choose_channel(test_fields, channels)

        with open(self.path / "train.txt", "w") as f:
            for fov, channel in pairs_train:
                f.write(f"{fov.name},{channel}\n")

        with open(self.path / "test.txt", "w") as f:
            for fov, channel in pairs_test:
                f.write(f"{fov.name},{channel}\n")

    def read_split(self, split_type: str, channel: int = None) -> List[Tuple[Field, int]]:
        with open(self.path / f"{split_type}.txt", "r") as f:
            files = f.read().splitlines()

        files = [f.split(",") for f in files if f]
        if channel:
            return [(Field(str(f[0]), self), int(channel)) for f in files]
        else:
            return [(Field(str(f[0]), self), int(f[1])) for f in files]

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

    def score(self, channel_nuclei, channel_centrioles, method, model, vicinity):

        if len([f for f in self.projections.iterdir()]) == 0:
            logger.error('Projections folder empty', exc_info=True)
            raise

        channels = self.fields[0].projection.shape[0]

        if channel_nuclei not in range(channels):
            logger.error("Index for nuclei (%s) out of index range" % channel_nuclei, exc_info=True)
            raise

        if not set(channel_centrioles).issubset(set(range(channels))):
            logger.error(
                "Channels (%s) out of channel range %s" % channel_centrioles,
                set(range(channels)),
                exc_info=True)
            raise

        path_visualisation_model = self.visualisation / model.name
        path_visualisation_model.mkdir(exist_ok=True)

        pbar = tqdm(self.fields)
        path_run = self.results / model.name
        path_run.mkdir(exist_ok=True)

        scores = []
        for field in pbar:
            pbar.set_description(f"{field.name}")
            logger.info("Processing %s" % field.name)
            nuclei = field.extract_nuclei(channel_nuclei, 256)

            for channel in channel_centrioles:
                logger.info("Processing %s / %d" % (field.name, channel))
                nuclei_copy = copy.deepcopy(nuclei)

                predictions_path = (
                        self.predictions
                        / "centrioles"
                        / f"{field.name}{self.projection_suffix}_C{channel}.txt"
                )

                foci = field.extract_centrioles(method=method, model_path=model, channel=channel)
                save_foci(foci, predictions_path)

                nuclei_scored = assign(nuclei_copy, foci, vicinity=vicinity)
                scored = field.score(nuclei_scored, channel)
                scores.append(scored)

                vis = field.visualisation(
                    nuclei_scored, foci, channel, channel_nuclei
                )
                tif.imwrite(
                    path_visualisation_model / f"{field.name}_C{channel}_pred.png", vis
                )

                pbar.set_postfix(
                    {
                        "field": field.name,
                        "channel": channel,
                        "nuclei": len(nuclei_copy),
                        "foci": len(foci),
                    }
                )

                logger.info(
                    "(%s), channel %s: nuclei: %s; foci: %s"
                    % (field.name, channel, len(nuclei_copy), len(foci))
                )

                logger.info(
                    "Writing visualisations for (%s), channel %s" % (field.name, channel)
                )

            logger.info("DONE (%s)" % field.name)

        flattened = [leaf for tree in scores for leaf in tree]
        scores_df = pd.DataFrame(flattened)
        scores_df.to_csv(self.statistics / "scores_df.tsv", sep="\t", index=False)
        logger.info("Writing raw scores to %s" % str(self.statistics / "scores_df.tsv"))

    def dataset_metrics(
            self, split: str, model: Path, tolerance, threshold
    ) -> list[dict]:
        if type(tolerance) == int:
            tolerance = [tolerance]
        perfs = []
        for field, channel in self.pairs(split):
            annotation = field.annotation(channel)
            predictions = field.extract_centrioles(model, channel)
            for tol in tolerance:
                perf = field.field_metrics(
                    channel, annotation, predictions, tol, threshold=threshold
                )
                perfs.append(perf)
        return perfs


def choose_channel(
        fields: list[Field], channels: Union[int, List[int]]
) -> list[tuple[Field, int]]:
    """Pick a channel for each field."""
    if type(channels) == int:
        channels = [channels]
    return [
        (field, int(channel))
        for field, channel in itertools.product(fields, channels)
    ]
