# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import re
import sys
from datetime import datetime
from pathlib import Path

_RE_BEST_EPOCH = re.compile(r'_epoch(\d+)\.pth$', re.IGNORECASE)
_RE_TRAIN_FOLDER = re.compile(r'^train_(\d{8}_\d{6})$')
# 与权重文件名中 ``..._t20260428_210621_epoch...`` 对齐
_RE_CKPT_EMBEDDED_TS = re.compile(r't(\d{8}_\d{6})_')


def _infer_run_timestamp_for_test(train_out_dir_if_any, resolved_ckpt_abs):
    """与某次训练的 ``train_<YYYYMMDD_HHMMSS>`` 对齐；否则从权重文件名/路径推断，再否则用当前时间。"""
    if train_out_dir_if_any:
        base = osp.basename(train_out_dir_if_any.rstrip(osp.sep))
        m = _RE_TRAIN_FOLDER.match(base)
        if m:
            return m.group(1)
    p = osp.abspath(osp.dirname(resolved_ckpt_abs))
    for _ in range(16):
        m = _RE_TRAIN_FOLDER.match(osp.basename(p))
        if m:
            return m.group(1)
        parent = osp.dirname(p)
        if parent == p:
            break
        p = parent
    m = _RE_CKPT_EMBEDDED_TS.search(osp.basename(resolved_ckpt_abs))
    if m:
        return m.group(1)
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def resolve_checkpoint_argument(raw_checkpoint):
    """将 ``checkpoint`` 参数解析为真实 ``.pth`` 路径。

    - **文件路径**：视为权重文件，转成绝对路径后返回。
    - **目录路径**（如 ``.../data/checkpoints1/train_20260428_210621/``）：在目录树中搜索 ``best*.pth``，
      按文件名中 ``*_epochN.pth`` 的 **epoch 最大值**作为「当前保留的最优」权重（与同目录训练脚本行为一致）。

    Returns:
        tuple: (resolved_pth_abs, train_root_or_none)，传入目录时第二项为该训练产出根目录，
        供 ``tools/test.py`` 默认把测试日志写入 ``data/checkpoints2/test_<同一时间戳>/``。
    """
    raw = raw_checkpoint.strip()
    expanded = osp.abspath(osp.expanduser(raw))

    if osp.isfile(expanded):
        return expanded, None

    if osp.isdir(expanded):
        candidates = []
        for root, _dirs, files in os.walk(expanded):
            for fn in files:
                if fn.startswith('best') and fn.endswith('.pth'):
                    candidates.append(osp.join(root, fn))
        if not candidates:
            raise FileNotFoundError(
                f'目录内未找到 best*.pth: {expanded}\n'
                f'若为课题训练产出，通常应在 ``train_<时间戳>/<同时间戳子目录>/best_*.pth``。\n'
                f'请确认路径前缀为仓库下 ``my_MambaVision-main/my_MambaVision-main/data/checkpoints1/...`` '
                '（不要使用不存在的 ``/data/checkpoints1/...``）。')
        def _epoch_key(p):
            m = _RE_BEST_EPOCH.search(osp.basename(p))
            return int(m.group(1)) if m else -1

        candidates.sort(key=_epoch_key, reverse=True)
        chosen = candidates[0]
        return chosen, expanded

    raise FileNotFoundError(
        f'checkpoint 既不是现存文件也不是目录: {raw} -> {expanded}')

_TOOLS_DIR = str(Path(__file__).resolve().parent)
_SEM_ROOT = str(Path(__file__).resolve().parent.parent)
for _p in (_SEM_ROOT, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import mamba_vision

try:
    import binary_fg_metrics  # noqa: F401
except ImportError:
    pass
try:
    import training_viz_hooks  # noqa: F401
    from wire_seg_hooks import apply_wire_seg_test_options
except ImportError:
    apply_wire_seg_test_options = None

# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        'checkpoint',
        help='权重 ``.pth`` 的路径，或**训练产出目录**（如 ``../../data/checkpoints1/train_<时间戳>``），后者将自动选取目录内 epoch 最大的 ``best*.pth``')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    resolved_ckpt, train_out_dir_if_any = resolve_checkpoint_argument(
        args.checkpoint)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir：CLI 优先；课题 wire_seg 默认 ``<仓库>/data/checkpoints2/test_<与训练一致的时间戳>/``
    repo_root = Path(_SEM_ROOT).parent
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('wire_seg_experiment'):
        run_ts = _infer_run_timestamp_for_test(train_out_dir_if_any, resolved_ckpt)
        cfg.work_dir = str(
            (repo_root / 'data' / 'checkpoints2' / f'test_{run_ts}').resolve())
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = resolved_ckpt

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    if apply_wire_seg_test_options is not None:
        apply_wire_seg_test_options(cfg, args)
    else:
        if args.out is not None and isinstance(cfg.get('test_evaluator'), dict):
            tev = dict(cfg['test_evaluator'])
            tev['output_dir'] = args.out
            tev['keep_results'] = True
            cfg['test_evaluator'] = tev

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
