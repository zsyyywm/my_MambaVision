# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import sys
from datetime import datetime
from pathlib import Path

# ``binary_fg_metrics`` 等在 ``semantic_segmentation/`` 根目录；``mamba_vision.py`` 在 ``tools/``。
_TOOLS_DIR = str(Path(__file__).resolve().parent)
_SEM_ROOT = str(Path(__file__).resolve().parent.parent)
for _p in (_SEM_ROOT, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
import mamba_vision

# 二分类电线实验：注册指标与 Hook（非该实验的配置不受影响）。
try:
    import binary_fg_metrics  # noqa: F401
    import wire_seg_hooks  # noqa: F401 — 注册早停 / checkpoint 等 Hook
    import training_viz_hooks  # noqa: F401 — 注册 ConsoleSummaryHook / PlotMetricsHook
    from wire_seg_hooks import apply_wire_seg_training_options
except ImportError:
    apply_wire_seg_training_options = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--no-clean-console',
        action='store_true',
        default=False,
        help='关闭彩色终端摘要（仅 wire_seg 实验配置生效）')
    parser.add_argument(
        '--clean-console-interval',
        type=int,
        default=None,
        help='覆盖配置中的 wire_seg_console_interval（每 N iter 一行 [训练]）')
    parser.add_argument(
        '--no-plot-curves',
        action='store_true',
        default=False,
        help='关闭 val_metrics.csv 与 train_curves.png / val_foreground_trends.png')
    parser.add_argument(
        '--plot-sample-interval',
        type=int,
        default=None,
        help='覆盖配置中的 wire_seg_plot_sample_interval')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('wire_seg_experiment') and (
            cfg.get('work_dir', None) in (None, '', 'checkpoint', './checkpoint')):
        # 课题统一目录：默认训练产物写到仓库根 ``data/checkpoints1/train_<ts>/``。
        # 若要复用同一 run 目录续训，请显式传 ``--work-dir``。
        repo_root = Path(_SEM_ROOT).parent
        run_ts = str(cfg.get('timestamp') or '').strip()
        if not run_ts:
            run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if run_ts.startswith('train_'):
            folder = run_ts
        else:
            folder = f'train_{run_ts}'
        cfg.work_dir = str((repo_root / 'data' / 'checkpoints1' / folder).resolve())
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    if cfg.get('wire_seg_experiment'):
        if getattr(args, 'no_clean_console', False):
            cfg.wire_seg_enable_console = False
        if getattr(args, 'clean_console_interval', None) is not None:
            cfg.wire_seg_console_interval = int(args.clean_console_interval)
        if getattr(args, 'no_plot_curves', False):
            cfg.wire_seg_enable_plots = False
        if getattr(args, 'plot_sample_interval', None) is not None:
            cfg.wire_seg_plot_sample_interval = int(args.plot_sample_interval)

    if apply_wire_seg_training_options is not None:
        apply_wire_seg_training_options(cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
