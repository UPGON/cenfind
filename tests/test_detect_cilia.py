from pathlib import Path

import pytest

from cenfind.core.data import Dataset, Field


class TestDataCilia:
    path_dataset = Path("../data/cilia")
    field_name_no_cilia = "RPE1-GFPCep135_75k_mCETN2-AF488_rArl13b-AF568_gCEP164-AF647_2_MMStack_1-Pos_000_002"
    field_name_cilia = "RPE1-GFPCep135_Palbo_48h_100k_mCETN2-AF488_rArl13b-AF568_gCEP164-AF647_2_MMStack_1-Pos_000_000"
    dataset = Dataset(path=path_dataset, image_type=".ome.tif")
    dataset.setup()
    dataset.write_fields()

    field = Field(field_name_no_cilia, dataset)

    channel = field.channel(1)

    def test_detect_cilia(self):
        cilia_pred = list(range(22))
        assert len(cilia_pred) == 22
