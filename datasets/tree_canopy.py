from mmseg.datasets import DATASETS, CustomDataset

@DATASETS.register_module()
class TCD_Dataset(CustomDataset):
    CLASSES = ('background', 'individual', 'group_of_trees')
    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]

    def __init__(self, **kwargs):
        print(">>> TCD_Dataset __init__ called")   # Add this
        super().__init__(
            img_suffix = '.png',
            seg_map_suffix = '.png',
            reduce_zero_label=False,  # keep 0 as background
            # ignore_index=0,           # ignore background pixels during loss
            **kwargs
        )
