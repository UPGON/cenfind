import functools
from pathlib import Path

import cv2
from spotipy.model import SpotNet
import itertools
import random
from typing import Any
from cenfind.core.outline import Centre, Contour


def full_in_field(coordinate, image, fraction) -> bool:
    _, h, w = image.shape
    pad_lower = int(fraction * h)
    pad_upper = h - pad_lower
    if all([pad_lower < c < pad_upper for c in coordinate]):
        return True
    return False


def flag(is_full: bool) -> tuple:
    return (0, 0, 255) if is_full else (0, 255, 0)


def signed_distance(focus: Centre, nucleus: Contour) -> float:
    """Wrapper for the opencv PolygonTest"""

    result = cv2.pointPolygonTest(nucleus.contour,
                                  focus.to_cv2(),
                                  measureDist=True)
    return result


def frac(x):
    return x.sum() / len(x)


def blob2point(keypoint: cv2.KeyPoint) -> tuple[int, ...]:
    res = (int(keypoint.pt[1]), int(keypoint.pt[0]))
    return res


@functools.lru_cache(maxsize=None)
def get_model(model):
    path = Path(model)
    if not path.is_dir():
        raise (FileNotFoundError(f"{path} is not a directory"))

    return SpotNet(None, name=path.name, basedir=str(path.parent))


def resize_image(data):
    height, width = data.shape
    shrinkage_factor = int(height // 256)
    height_scaled = int(height // shrinkage_factor)
    width_scaled = int(width // shrinkage_factor)
    data_resized = cv2.resize(data,
                              dsize=(height_scaled, width_scaled),
                              fx=1, fy=1,
                              interpolation=cv2.INTER_NEAREST)
    return data_resized
