from pathlib import Path

import pytest

from cenfind.core.data import Dataset, Field

ROOT_DIR = Path(__file__).parent.parent


class TestData:
    path_dataset = ROOT_DIR / "data/dataset_test"
    field_name = "RPE1wt_CEP63+CETN2+PCNT_1_000_000"
    dataset = Dataset(path=path_dataset, image_type=".ome.tif")
    dataset.setup()
    dataset.write_fields()
    field = Field(field_name, dataset)
    projection = field.projection
    channel = field.channel(1)

    def test_write_fields(self):
        self.dataset.write_fields()
        assert (self.dataset.path / "fields.txt").is_file()

    def test_field(self):
        assert self.field.name == self.field_name

    def test_dataset(self):
        assert self.dataset.path == self.path_dataset

    def test_projection(self):
        assert self.projection.ndim == 3

    def test_channel(self):
        assert self.channel.shape == (2048, 2048)


class TestDataNotExisting:
    path_dataset = Path("./not/existing")

    def test_dataset_initialisation(self):
        with pytest.raises(FileNotFoundError):
            Dataset(self.path_dataset)
