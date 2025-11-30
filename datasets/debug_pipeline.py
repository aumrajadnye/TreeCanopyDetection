from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class DebugPrint:
    def __init__(self, msg=""):
        self.msg = msg

    def __call__(self, results):
        print(f"[DEBUG PIPELINE] {self.msg}")
        if 'img_path' in results:
            print(f"   Image: {results['img_path']}")
        if 'gt_seg_map_path' in results:
            print(f"   Seg:   {results['gt_seg_map_path']}")
        if 'img' in results:
            print(f"   Image shape: {results['img'].shape}")
        if 'gt_seg_map' in results:
            print(f"   Mask shape:  {results['gt_seg_map'].shape}")
        return results
    