import numpy as np
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MapWireMask255To1:
    """Map wire mask labels from {0,255} to {0,1}."""

    def __call__(self, results):
        if 'gt_seg_map' in results:
            gt = results['gt_seg_map']
            if isinstance(gt, np.ndarray):
                gt = gt.copy()
                gt[gt == 255] = 1
                results['gt_seg_map'] = gt
        return results
