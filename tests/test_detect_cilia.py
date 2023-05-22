from pathlib import Path

from cenfind.core.data import Dataset, Field
from cenfind.core.detectors import extract_cilia


class TestDataCilia:
    path_dataset = Path("../data/cilia")
    field_name_cilia_6 = "RPE1-GFPCep135_NC_48h_100k_mCETN2-AF488_rArl13b-AF568_gCEP164-AF647_1_MMStack_10-Pos_000_001"
    field_name_cilia_22 = "RPE1-GFPCep135_Palbo_48h_100k_mCETN2-AF488_rArl13b-AF568_gCEP164-AF647_2_MMStack_1-Pos_000_000"
    dataset = Dataset(path=path_dataset, image_type=".ome.tif")
    dataset.setup()
    dataset.write_fields()

    field_cilia_6 = Field(field_name_cilia_6, dataset)
    field_cilia_22 = Field(field_name_cilia_22, dataset)

    def test_detect_cilia(self):
        assert abs(len(extract_cilia(self.field_cilia_6, channel=2)) - 4) / 4 <= .1
        assert abs(len(extract_cilia(self.field_cilia_22, channel=2)) - 22) / 22 <= .1
