#!/usr/bin/env python3
"""快速检查 DataA/DataB 二分类配置能否构建数据集并打印样本数（不启动训练）。"""
import argparse
import os
import os.path as osp
import sys

_TOOLS = osp.abspath(osp.dirname(__file__))
_ROOT = osp.abspath(osp.join(_TOOLS, '..'))
for _p in (_TOOLS, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mamba_vision  # noqa: F401
import binary_fg_metrics  # noqa: F401

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        nargs='?',
        default='configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py',
        help='wire_iou 配置文件路径（相对 semantic_segmentation）')
    args = parser.parse_args()
    os.chdir(_ROOT)
    register_all_modules()
    init_default_scope('mmseg')
    cfg = Config.fromfile(args.config)
    for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        dl = cfg.get(name)
        if not isinstance(dl, dict) or 'dataset' not in dl:
            print(f'{name}: (skip)')
            continue
        ds_cfg = dl['dataset']
        ds = DATASETS.build(ds_cfg)
        print(f'{name} dataset len = {len(ds)}  (data_root={ds_cfg.get("data_root")})')


if __name__ == '__main__':
    main()
