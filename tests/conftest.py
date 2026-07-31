import numpy as np
import pytest
import tifffile as tf

from cenfind.core.data import Field


@pytest.fixture
def make_field(tmp_path):
    """Writes a small synthetic multi-channel TIF and returns a Field for it."""

    def _make(data=None, shape=(2, 200, 200), dtype="uint16", name="field.tif"):
        if data is None:
            data = np.zeros(shape, dtype=dtype)
        path = tmp_path / name
        tf.imwrite(path, data)
        return Field(path)

    return _make
