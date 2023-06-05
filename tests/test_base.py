from pathlib import Path

import pytest

from cenfind.core.data import Dataset, Field

ROOT_DIR = Path(__file__).parent.parent


class TestData:
    path_dataset = ROOT_DIR / "data/dataset_test"
    field_name = "RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000"
    dataset = Dataset(path=path_dataset, image_type=".ome.tif")
    dataset.setup()
    dataset.write_fields()
    dataset.write_pairs(channels=(1, 2))
    field = Field(field_name, dataset)
    stack = field.stack
    projection = field.projection
    channel = field.channel(1)

    def test_write_fields(self):
        self.dataset.write_fields()
        assert (self.dataset.path / "fields.txt").is_file()

    def test_write_projections(self):
        self.dataset.write_projections()
        projection = self.field.projection
        assert projection.shape == (4, 2048, 2048)

    def test_field(self):
        assert self.field.name == self.field_name

    def test_dataset(self):
        assert self.dataset.path == self.path_dataset

    def test_pairs(self):
        assert self.dataset.pairs() == [
            (Field("RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000", self.dataset), 1),
            (Field("RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_002", self.dataset), 2),
        ]

    def test_stack(self):
        assert self.stack.ndim == 4

    def test_projection(self):
        assert self.projection.ndim == 3

    def test_squash(self):
        projected = self.stack.max(1)
        assert projected.shape == (4, 2048, 2048)

    def test_channel(self):
        assert self.channel.shape == (2048, 2048)
        assert self.dataset == self.dataset


class TestDataNotExisting:
    path_dataset = Path("./not/existing")

    def test_dataset_initialisation(self):
        with pytest.raises(FileNotFoundError):
            Dataset(self.path_dataset)
