from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class TCD_Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('individual', 'group_of_trees'),
        palette=[[255, 0, 0], [0, 0, 255]]
    )

    # def __init__(self, data_root, pipeline=None, test_mode=False, img_suffix='.png', seg_map_suffix='.png', **kwargs):
    #     super().__init__(
    #         data_root=data_root,
    #         pipeline=pipeline,
    #         test_mode=test_mode,
    #         img_suffix=img_suffix,
    #         seg_map_suffix=seg_map_suffix,
    #         **kwargs
    #     )
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
