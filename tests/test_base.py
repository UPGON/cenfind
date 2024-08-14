from pathlib import Path, PosixPath

import pytest
from cenfind.core.data import Dataset, Field

ROOT_DIR = Path(__file__).parent.parent


class TestData:
    path_dataset = ROOT_DIR / "data/dataset_test"
    dataset = Dataset(path_dataset)
    field_name = "RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000.tif"
    field_path = path_dataset / "projections" / field_name
    field = Field(field_path)
    channel = field.data[1, ...]
    good_field_name = ROOT_DIR / "data/dataset_test/projections" / "RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000.tif"

    def test_field_data(self):
        field = Field(self.good_field_name)
        assert field.data.shape == (4, 2048, 2048)

    def test_field_name(self):
        assert self.field.name == self.field_name.split('.')[0]

    def test_projection(self):
        assert self.field.data.ndim == 3

    def test_channel(self):
        assert self.channel.shape == (2048, 2048)

    def test_fields(self):
        assert self.dataset.fields == [Field(path=PosixPath(
            ROOT_DIR / "data/dataset_test/projections/RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000.tif")),
            Field(path=PosixPath(
                ROOT_DIR / "data/dataset_test/projections/RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_002.tif"))]


class TestDataNotExisting:
    path_dataset = Path("./not/existing")

    def test_dataset_initialisation(self):
        with pytest.raises(FileNotFoundError):
            Dataset(self.path_dataset)
