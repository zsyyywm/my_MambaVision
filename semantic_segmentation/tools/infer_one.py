#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""单张图像推理：与 ``test_pipeline`` 一致预处理，输出「原图 | 上色预测」横向拼接图。

在 ``semantic_segmentation`` 目录下执行::

    python tools/infer_one.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py \\
        --img path/to/xxx.jpg --checkpoint path/to/best.pth

或自动选用最新 best 权重（在 ``work_dir`` / ``checkpoint`` 下搜索 ``best_*.pth``）::

    python tools/infer_one.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py \\
        --img path/to/xxx.jpg --best
"""
import argparse
import glob
import os
import os.path as osp
import sys

from mmengine.config import DictAction

import cv2
import mmcv
import numpy as np
import torch


def _collect_best_checkpoints(search_root):
    if not search_root or not osp.isdir(search_root):
        return []
    hits = []
    patterns = (
        'best_IoU*.pth', 'best_mIoU*.pth', 'best_mDice*.pth',
        'best_*.pth', 'best*.pth')
    for pat in patterns:
        hits.extend(glob.glob(osp.join(search_root, pat)))
    for sub in glob.glob(osp.join(search_root, '*')):
        if osp.isdir(sub):
            for pat in patterns:
                hits.extend(glob.glob(osp.join(sub, pat)))
    out, seen = [], set()
    for p in hits:
        p = osp.abspath(p)
        if osp.isfile(p) and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_best_checkpoint_path(cfg):
    dirs = []
    wd = cfg.get('work_dir')
    if wd:
        dirs.append(osp.abspath(wd))
    dh = cfg.get('default_hooks')
    if dh is not None:
        ch = dh.get('checkpoint') if isinstance(dh, dict) else None
        if isinstance(ch, dict) and ch.get('out_dir'):
            od = osp.abspath(ch['out_dir'])
            if od not in dirs:
                dirs.append(od)
    hits = []
    for d in dirs:
        hits.extend(_collect_best_checkpoints(d))
    if not hits:
        loc = ', '.join(dirs) if dirs else '(work_dir 未设置)'
        raise FileNotFoundError(
            f'未找到验证最优权重（best_*.pth）。已搜索: {loc}')
    hits.sort(key=lambda p: osp.getmtime(p), reverse=True)
    return hits[0]


def _label_to_rgb(seg: np.ndarray, palette):
    h, w = seg.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for i, rgb in enumerate(palette):
        if i > int(seg.max()) + 1:
            break
        vis[seg == i] = np.array(rgb, dtype=np.uint8)
    return vis


def parse_args():
    p = argparse.ArgumentParser(description='单张图推理：原图与预测并排保存')
    p.add_argument('config', help='配置文件路径（相对或绝对）')
    p.add_argument('--img', required=True, help='输入图像路径（jpg/png 等）')
    p.add_argument(
        '--checkpoint',
        default=None,
        help='权重 .pth；与 --best 二选一')
    p.add_argument(
        '--best',
        action='store_true',
        help='在 work_dir / checkpoint.out_dir 下自动选最新 best_*.pth')
    p.add_argument(
        '--out',
        default=None,
        help='输出路径；默认 checkpoint/infer_one/<图名>_compare.png')
    p.add_argument(
        '--device', default='cuda:0', help='cuda:0 或 cpu')
    p.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='同 train.py / test.py')
    args = p.parse_args()
    if not args.best and not args.checkpoint:
        p.error('请提供 --checkpoint，或使用 --best')
    return args


def main():
    args = parse_args()

    root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    os.chdir(root)
    tools_dir = osp.dirname(osp.abspath(__file__))
    for d in (tools_dir, root):
        if d not in sys.path:
            sys.path.insert(0, d)

    import mamba_vision  # noqa: F401
    import binary_fg_metrics  # noqa: F401

    from mmengine.config import Config
    from mmseg.apis import inference_model, init_model

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    wd = cfg.get('work_dir')
    if wd:
        if not osp.isabs(wd):
            wd = osp.join(root, wd)
        cfg.work_dir = osp.abspath(wd)

    if args.best:
        ckpt = find_best_checkpoint_path(cfg)
        print(f'--best: 使用权重\n  {ckpt}')
    else:
        ckpt = osp.abspath(args.checkpoint)

    model = init_model(cfg, ckpt, device=args.device)
    img_path = osp.abspath(args.img)
    if not osp.isfile(img_path):
        raise FileNotFoundError(img_path)

    with torch.no_grad():
        result = inference_model(model, img_path)

    pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int64)
    if pred.ndim > 2:
        pred = pred[0]

    palette = None
    if getattr(model, 'dataset_meta', None):
        palette = model.dataset_meta.get('palette')
    if not palette and cfg.get('test_dataloader'):
        meta = cfg.test_dataloader.dataset.get('metainfo')
        if isinstance(meta, dict):
            palette = meta.get('palette')
    if not palette:
        palette = [[0, 0, 0], [255, 255, 255]]

    img_bgr = mmcv.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f'无法读取图像: {img_path}')

    pred_rgb_small = _label_to_rgb(pred, palette)
    h0, w0 = img_bgr.shape[:2]
    pred_rgb = cv2.resize(
        pred_rgb_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
    img_rgb = mmcv.bgr2rgb(img_bgr)
    combo_rgb = np.concatenate([img_rgb, pred_rgb], axis=1)
    combo_bgr = mmcv.rgb2bgr(combo_rgb)

    if args.out:
        out_path = osp.abspath(args.out)
    else:
        out_dir = osp.join(root, 'checkpoint', 'infer_one')
        os.makedirs(out_dir, exist_ok=True)
        stem = osp.splitext(osp.basename(img_path))[0]
        out_path = osp.join(out_dir, f'{stem}_compare.png')

    out_dir = osp.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mmcv.imwrite(combo_bgr, out_path)
    print(f'已保存: {out_path}')


if __name__ == '__main__':
    main()
