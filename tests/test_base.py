from pathlib import Path

from cenfind.core.data import Dataset, Field


class TestData:
    path_dataset = Path('./dataset_test')
    field_name = 'RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000'
    dataset = Dataset(path=path_dataset, image_type='.ome.tif')
    field = Field(field_name, dataset)
    stack = field.stack
    projection = field.projection
    channel = field.channel(1)

    def test_field(self):
        assert self.field.name == self.field_name

    def test_dataset(self):
        assert self.dataset.path == self.path_dataset
        assert self.dataset.pairs() == [(Field('RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000', self.dataset), 1),
                                        (Field('RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_002', self.dataset), 1)]

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
