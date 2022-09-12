from pathlib import Path

from centrack.data.base import Dataset, Stack, Field, Projection, Channel


class TestData:
    path_dataset = Path('./dataset_test')
    field_name = 'RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000'
    dataset = Dataset(path=path_dataset, image_type='.ome.tif')
    field = Field(field_name)
    stack = Stack(dataset=dataset, field=field)
    projection = Projection(dataset=dataset, field=field)
    channel = Channel(projection=projection, channel_id=1)
    stack.write_projection()

    def test_field(self):
        assert self.field.name == self.field_name

    def test_dataset(self):
        assert self.dataset.path == self.path_dataset
        assert self.dataset.fields('.ome.tif') == [Field('RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_002'),
                                                   Field('RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000')]

    def test_stack(self):
        assert self.stack.field.name == self.field_name
        assert self.stack.path == self.path_dataset / 'raw/RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000.ome.tif'
        assert self.stack.data().ndim == 4
        assert self.stack.project()[0].ndim == 3

    def test_projection(self):
        assert self.projection.name == 'RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000_max'
        assert self.projection.field == self.field
        assert self.projection.path == self.path_dataset / 'projections/RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000_max.tif'
        assert self.projection.data.ndim == 3

    def test_channel(self):
        assert self.channel.name == 'RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000_max_C1'
        assert self.channel.data.shape == (2048, 2048)
        assert self.dataset == self.dataset
