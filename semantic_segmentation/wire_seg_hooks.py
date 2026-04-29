# Wire / 二分类分割实验用 Hook：与 TransNeXt 工程对齐的 val IoU 早停、checkpoint 与日志同目录。
import logging
import os
import os.path as osp

from mmengine.dist import is_main_process
from mmengine.fileio import FileClient, get_file_backend
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS


@HOOKS.register_module()
class CheckpointToLogDirHook(Hook):
    """将各 ``CheckpointHook`` 的 ``out_dir`` 指到 ``runner.log_dir`` 或其子目录。

    - ``subdirs is None``：与旧行为相同，所有 ``CheckpointHook`` 共用 ``log_dir``。
    - ``subdirs`` 为与 **CheckpointHook 出现顺序一一对应** 的子文件夹名时：每个
      checkpoint 存到 ``join(log_dir, name)``（如方案一/方案二分目录存权重）。"""

    priority = 'LOWEST'

    def __init__(self, subdirs=None):
        super().__init__()
        self.subdirs = subdirs

    def before_train(self, runner):
        logd = runner.log_dir
        chooks = [h for h in runner.hooks
                  if h.__class__.__name__ == 'CheckpointHook']
        if self.subdirs and len(self.subdirs) == len(chooks):
            targets = []
            for name in self.subdirs:
                s = str(name).strip() if name is not None else ''
                t = logd if not s else osp.join(logd, s)
                os.makedirs(t, exist_ok=True)
                targets.append(t)
            for hook, t in zip(chooks, targets):
                hook.out_dir = t
                hook.file_client = FileClient.infer_client(
                    hook.file_client_args, hook.out_dir)
                if hook.file_client_args is None:
                    hook.file_backend = get_file_backend(
                        hook.out_dir, backend_args=hook.backend_args)
                else:
                    hook.file_backend = hook.file_client
            if is_main_process():
                print_log(
                    f'CheckpointToLogDirHook: 各 save_best/周期 权重子目录为\n' +
                    '\n'.join(f'  {s}' for s in targets),
                    logger='current',
                    level=logging.INFO)
        else:
            for hook in chooks:
                hook.out_dir = logd
                hook.file_client = FileClient.infer_client(
                    hook.file_client_args, hook.out_dir)
                if hook.file_client_args is None:
                    hook.file_backend = get_file_backend(
                        hook.out_dir, backend_args=hook.backend_args)
                else:
                    hook.file_backend = hook.file_client
            if is_main_process() and (self.subdirs
                                        and len(self.subdirs) != len(chooks)):
                print_log(
                    f'CheckpointToLogDirHook: subdirs 数量（{len(self.subdirs) if self.subdirs else 0}）与 '
                    f'CheckpointHook 数（{len(chooks)}）不一致，已回退为单目录: {logd}',
                    logger='current',
                    level=logging.WARNING)
            if is_main_process():
                print_log(
                    f'CheckpointToLogDirHook: save_best / 周期权重将保存到\n  {logd}',
                    logger='current',
                    level=logging.INFO)


@HOOKS.register_module()
class ValLossPatienceEarlyStopHook(Hook):
    """连续 ``patience`` 次验证监控指标未刷新最优则 ``stop_training``（与 MMEngine 早停机制一致）。"""

    priority = 'NORMAL'

    def __init__(self, monitor='val/loss', patience=50, rule='less', min_delta=0.0):
        self.monitor = monitor
        self.patience = int(patience)
        self.rule = str(rule).lower()
        self.min_delta = float(min_delta)
        self._best = None
        self._epochs_no_improve = 0

    def after_val_epoch(self, runner, metrics=None):
        if not isinstance(metrics, dict) or self.monitor not in metrics:
            return
        try:
            cur = float(metrics[self.monitor])
        except (TypeError, ValueError):
            return
        if self._best is None:
            self._best = cur
            self._epochs_no_improve = 0
            return
        if self.rule == 'less':
            improved = cur < (self._best - self.min_delta)
        else:
            improved = cur > (self._best + self.min_delta)
        if improved:
            self._best = cur
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1
        if self._epochs_no_improve < self.patience:
            return
        tl = getattr(runner, 'train_loop', None)
        if tl is not None and hasattr(tl, 'stop_training'):
            tl.stop_training = True
        if is_main_process():
            print_log(
                f'ValLossPatienceEarlyStopHook: 连续 {self.patience} 次验证 '
                f'「{self.monitor}」未优于当前最优 {self._best:.6f}，触发早停。',
                logger='current',
                level=logging.WARNING)


def apply_wire_seg_training_options(cfg):
    """读取配置中的 ``wire_seg_*`` 项，注入 ``custom_hooks``。"""
    if not cfg.get('wire_seg_experiment'):
        return
    custom = list(cfg.get('custom_hooks') or [])

    def _has_early_stop(mon):
        for h in custom:
            if not isinstance(h, dict):
                continue
            if h.get('type') != 'ValLossPatienceEarlyStopHook':
                continue
            if h.get('monitor') == mon:
                return True
        return False

    patience = cfg.get('wire_seg_iou_early_stop_patience')
    if patience is not None:
        patience = int(patience)
        if patience > 0:
            ck = (cfg.get('default_hooks') or {}).get('checkpoint') or {}
            monitor = ck.get('save_best') or 'val/IoU'
            rule = str(ck.get('rule', 'greater')).lower()
            if not _has_early_stop(monitor):
                custom.append(
                    dict(
                        type='ValLossPatienceEarlyStopHook',
                        monitor=monitor,
                        patience=patience,
                        rule=rule))

    if cfg.get('wire_seg_checkpoint_to_log_dir', True):
        if not any(
                isinstance(h, dict) and h.get('type') == 'CheckpointToLogDirHook'
                for h in custom):
            sub = cfg.get('wire_seg_checkpoint_subdirs')
            custom.append(dict(type='CheckpointToLogDirHook', subdirs=sub))

    # 与 TransNeXt mask2former 一致：彩色终端、验证后指标表、val_metrics.csv、train_curves.png 等
    if cfg.get('wire_seg_enable_console', True):
        iv = int(cfg.get('wire_seg_console_interval', 1))
        vgroups = cfg.get('wire_seg_val_console_key_groups')
        if not any(
                isinstance(h, dict) and h.get('type') == 'ConsoleSummaryHook'
                for h in custom):
            ch = dict(type='ConsoleSummaryHook', interval=iv)
            if vgroups is not None:
                ch['val_key_groups'] = vgroups
            custom.append(ch)
        else:
            for h in custom:
                if (isinstance(h, dict) and h.get('type') == 'ConsoleSummaryHook'
                        and vgroups is not None
                        and 'val_key_groups' not in h):
                    h['val_key_groups'] = vgroups

    if cfg.get('wire_seg_enable_plots', True):
        si = int(cfg.get('wire_seg_plot_sample_interval', 50))
        vbr = cfg.get('wire_seg_plot_val_branches')
        if not any(
                isinstance(h, dict) and h.get('type') == 'PlotMetricsHook'
                for h in custom):
            ph = dict(type='PlotMetricsHook', sample_interval=si)
            if vbr is not None:
                ph['val_branches'] = vbr
            custom.append(ph)
        else:
            for h in custom:
                if (isinstance(h, dict) and h.get('type') == 'PlotMetricsHook'
                        and vbr is not None
                        and 'val_branches' not in h):
                    h['val_branches'] = vbr

    cfg.custom_hooks = custom


def apply_wire_seg_test_options(cfg, args=None):
    """供 ``tools/test.py`` 使用：在 ``--out`` 且双 evaluator 时，按子目录落盘预测；并注入
    ``WireSegTestSummaryHook`` 以在终端分块输出方案1/2 的测试表（与训练时表结构一致）。"""
    if not cfg.get('wire_seg_experiment'):
        return
    _import_hook()
    _wire_test_out(cfg, args)
    tev = cfg.get('test_evaluator')
    if not isinstance(tev, (list, tuple)) or len(tev) < 2:
        return
    vg = cfg.get('wire_seg_val_console_key_groups')
    if not vg:
        return
    custom = list(cfg.get('custom_hooks') or [])
    if any(
            isinstance(h, dict) and h.get('type') == 'WireSegTestSummaryHook'
            for h in custom):
        cfg.custom_hooks = custom
        return
    custom.append(
        dict(
            type='WireSegTestSummaryHook',
            val_key_groups=vg,
            title_tag='[测试] '))
    cfg.custom_hooks = custom


def _import_hook():
    import training_viz_hooks  # noqa: F401 — 注册 WireSegTestSummaryHook


def _wire_test_out(cfg, args):
    if args is None or not getattr(args, 'out', None):
        return
    out = str(args.out).rstrip(osp.sep)
    if not out:
        return
    tev = cfg.get('test_evaluator')
    subs = (cfg.get('wire_seg_test_out_subdirs') or
            cfg.get('wire_seg_checkpoint_subdirs') or None)
    if (isinstance(tev, (list, tuple)) and len(tev) >= 2 and subs
            and len(subs) >= len(tev)):
        new_te = []
        for i, ev in enumerate(tev):
            d = dict(ev) if isinstance(ev, dict) else ev
            if not isinstance(d, dict):
                new_te.append(d)
                continue
            d = dict(d)
            d['output_dir'] = osp.join(out, str(subs[i]).strip())
            d['keep_results'] = True
            new_te.append(d)
        cfg.test_evaluator = new_te
        return
    if isinstance(tev, dict):
        t = dict(tev)
        t['output_dir'] = out
        t['keep_results'] = True
        cfg.test_evaluator = t
        return
    if isinstance(tev, (list, tuple)) and len(tev) == 1 and isinstance(
            tev[0], dict):
        t = dict(tev[0])
        t['output_dir'] = out
        t['keep_results'] = True
        cfg.test_evaluator = [t]
