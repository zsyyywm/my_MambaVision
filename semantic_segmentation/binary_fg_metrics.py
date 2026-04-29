# Copyright (c) OpenMMLab-style. 二分类前景（如输电线）评测：主指标为前景 IoU，并给出 F1/Precision/Recall。
import os.path as osp
from collections import OrderedDict
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from prettytable import PrettyTable
from PIL import Image

from mmseg.evaluation.metrics.iou_metric import IoUMetric
from mmseg.registry import METRICS


def _tensor_2d_hw(x: torch.Tensor) -> torch.Tensor:
    t = x.detach()
    if t.dim() == 1:
        return t
    while t.dim() > 2 and t.size(0) == 1:
        t = t.squeeze(0)
    return t


@METRICS.register_module()
class BinaryForegroundIoUMetric(IoUMetric):
    """在 per-class 统计基础上，只把「前景类」的 IoU/F1/Precision/Recall 写入 metrics（0–100），并保留 aAcc。

    需在 ``iou_metrics`` 中同时包含 ``mIoU`` 与 ``mFscore``，以便得到逐类 IoU 与 P/R/F。
    """

    def __init__(self, foreground_index: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.foreground_index = int(foreground_index)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        class_names = self.dataset_meta['classes']
        fg = self.foreground_index

        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        iou_arr = ret_metrics.get('IoU')
        fscore_arr = ret_metrics.get('Fscore')
        precision_arr = ret_metrics.get('Precision')
        recall_arr = ret_metrics.get('Recall')

        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        out = OrderedDict()
        if 'aAcc' in ret_metrics_summary:
            out['aAcc'] = float(ret_metrics_summary['aAcc'])

        if iou_arr is not None and 0 <= fg < len(iou_arr):
            v = float(np.nan_to_num(iou_arr[fg], nan=0.0))
            out['IoU'] = float(np.round(v * 100, 2))

        if fscore_arr is not None and 0 <= fg < len(fscore_arr):
            v = float(np.nan_to_num(fscore_arr[fg], nan=0.0))
            out['F1'] = float(np.round(v * 100, 2))
        if precision_arr is not None and 0 <= fg < len(precision_arr):
            v = float(np.nan_to_num(precision_arr[fg], nan=0.0))
            out['Precision'] = float(np.round(v * 100, 2))
        if recall_arr is not None and 0 <= fg < len(recall_arr):
            v = float(np.nan_to_num(recall_arr[fg], nan=0.0))
            out['Recall'] = float(np.round(v * 100, 2))

        return out


def _get_seg_logits_tensor(data_sample) -> Optional[torch.Tensor]:
    """自 ``data_sample``/dict 取出 ``seg_logits``，形状 (C, H, W)。"""
    if isinstance(data_sample, dict):
        if 'seg_logits' not in data_sample:
            return None
        seg = data_sample['seg_logits']
    else:
        seg = getattr(data_sample, 'seg_logits', None)
        if seg is None:
            return None
    if isinstance(seg, dict) and 'data' in seg:
        t = seg['data']
    else:
        t = getattr(seg, 'data', seg)
    if t is None:
        return None
    t = t.squeeze(0) if t.dim() == 4 and t.size(0) == 1 else t
    return t


@METRICS.register_module()
class BinaryForegroundThreshIoUMetric(BinaryForegroundIoUMetric):
    """二分类：对 **前景类 softmax 概率** 使用阈值化预测（非 argmax），再算与 ``BinaryForegroundIoUMetric`` 相同的前景摘要。

    需模型在 ``predict`` 流程中在 ``data_sample`` 中保留 ``seg_logits``（与 MMSeg
    ``BaseSegmentor.postprocess_result`` 行为一致，C>=2 时使用多类 logits）。"""

    def __init__(
            self,
            threshold: float = 0.55,
            foreground_index: int = 1,
            **kwargs) -> None:
        super().__init__(foreground_index=foreground_index, **kwargs)
        self.threshold = float(threshold)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            logit2d = _get_seg_logits_tensor(data_sample)
            if logit2d is not None and logit2d.size(0) >= 2:
                p_fg = torch.softmax(
                    logit2d, dim=0)[int(self.foreground_index)]
                pred_label = (p_fg > self.threshold).long()
            elif logit2d is not None and logit2d.size(0) == 1:
                p = torch.sigmoid(logit2d[0])
                pred_label = (p > self.threshold).long()
            else:
                if isinstance(data_sample, dict):
                    pl = data_sample['pred_sem_seg']['data']
                else:
                    pl = data_sample.pred_sem_seg.data
                pred_label = pl.squeeze()
            pred_label = _tensor_2d_hw(pred_label)

            if not self.format_only:
                if isinstance(data_sample, dict):
                    label = data_sample['gt_sem_seg']['data'].squeeze().to(
                        pred_label)
                else:
                    label = data_sample.gt_sem_seg.data.squeeze().to(
                        pred_label)
                self.results.append(
                    self.intersect_and_union(
                        pred_label, label, num_classes, self.ignore_index))
            if self.output_dir is not None and is_main_process():
                mkdir_or_exist(self.output_dir)
                if isinstance(data_sample, dict):
                    pth = (data_sample.get('img_path', '') or
                           data_sample.get('file_name', ''))
                else:
                    pth = getattr(data_sample, 'img_path', '') or ''
                basename = osp.splitext(osp.basename(pth or 'out'))[0] or 'out'
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                if (isinstance(data_sample, dict)
                        and data_sample.get('reduce_zero_label', False)):
                    output_mask = output_mask + 1
                elif (not isinstance(data_sample, dict) and getattr(
                        data_sample, 'reduce_zero_label', False)):
                    output_mask = output_mask + 1
                out_img = Image.fromarray(output_mask.astype(np.uint8))
                out_img.save(png_filename)
