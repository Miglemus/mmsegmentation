from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class PoquetsDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("bg", "poquets"),
        palette=[[52, 149, 235], [245, 66, 126]],
    )

    def __init__(
            self,
            img_suffix=".jpg",
            seg_map_suffix=".png",
            **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs
        )
