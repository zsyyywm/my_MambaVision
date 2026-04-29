# 与 TransNeXt mask2former/train.py 对齐：彩色终端摘要、每轮验证后指标表、val_metrics.csv 与趋势图。
import csv
import logging
import numbers
import os
import os.path as osp

import torch
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS


@HOOKS.register_module()
class ConsoleSummaryHook(Hook):
    """与 ``segmentation/mask2former/train.py`` 中同名 Hook 对齐：训练行 / 本轮结束表 / 验证指标表 / Epoch 小结。"""

    priority = 'LOW'
    _MAX_REASONABLE_EPOCH_LEN = 100000

    def __init__(self, interval=1, val_key_groups=None):
        """val_key_groups: 可选。每项为
        ``{ 'title': '...', 'keys': { 'iou': 'val/IoU', 'f1': 'val/F1', ...} }``，
        为 None 时维持原先单表逻辑。"""
        self.interval = interval
        self.val_key_groups = val_key_groups
        self._epoch_total = None
        self._max_epochs_cfg = None
        self._max_iters_cfg = None
        self._train_blocks = 0
        self._cache = {}

    @staticmethod
    def _c(text, color):
        colors = {
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'magenta': '\033[95m',
            'bold': '\033[1m',
            'white': '\033[97m',
        }
        end = '\033[0m'
        return f"{colors.get(color, '')}{text}{end}"

    def _cell(self, text, width, color):
        s = str(text)
        if len(s) > width:
            s = s[: max(1, width - 2)] + '..'
        s = s.ljust(width)
        return self._c(s, color)

    @staticmethod
    def _safe_scalar(message_hub, key):
        scalar = message_hub.get_scalar(key)
        if scalar is None:
            return None
        return scalar.current()

    @staticmethod
    def _safe_lr(runner):
        if not hasattr(runner, 'optim_wrapper'):
            return None
        lrs = runner.optim_wrapper.get_lr()
        if isinstance(lrs, dict) and len(lrs) > 0:
            first = next(iter(lrs.values()))
            if isinstance(first, (list, tuple)) and len(first) > 0:
                return float(first[0])
            if isinstance(first, (int, float)):
                return float(first)
        return None

    @staticmethod
    def _safe_image_size(data_batch):
        if not isinstance(data_batch, dict):
            return None
        inputs = data_batch.get('inputs', None)
        if isinstance(inputs, torch.Tensor):
            return int(inputs.shape[-1])
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0 and isinstance(
                inputs[0], torch.Tensor):
            return int(inputs[0].shape[-1])
        return None

    @staticmethod
    def _pick_metric(metrics, keys):
        for key in keys:
            if key in metrics:
                return metrics[key]
        for key in keys:
            for metric_key, metric_val in metrics.items():
                if metric_key.endswith(key):
                    return metric_val
        return None

    def before_train(self, runner):
        self._epoch_total = None
        self._max_epochs_cfg = None
        self._max_iters_cfg = None
        tl_obj = getattr(runner, 'train_loop', None)
        if tl_obj is not None:
            self._max_epochs_cfg = getattr(tl_obj, 'max_epochs', None)
            self._max_iters_cfg = getattr(tl_obj, 'max_iters', None)
        try:
            tl = len(runner.train_dataloader)
            mi = getattr(runner.train_loop, 'max_iters', None)
            if tl and 0 < tl < self._MAX_REASONABLE_EPOCH_LEN and mi is not None:
                self._epoch_total = max(1, (int(mi) + tl - 1) // tl)
        except (TypeError, ValueError, AttributeError):
            pass

    def _train_loop_type(self, runner):
        tl = getattr(runner, 'train_loop', None)
        return type(tl).__name__ if tl is not None else ''

    def _progress_tags(self, runner, batch_idx=None):
        loop_type = self._train_loop_type(runner)
        it = runner.iter + 1
        tags = []

        if loop_type == 'EpochBasedTrainLoop':
            ep = runner.epoch + 1
            me = self._max_epochs_cfg
            tags.append(f'epoch={ep}' + (f'/{me}' if me is not None else ''))
            if batch_idx is not None:
                try:
                    n = len(runner.train_dataloader)
                    if n and n < self._MAX_REASONABLE_EPOCH_LEN:
                        tags.append(f'batch={batch_idx + 1}/{n}')
                except TypeError:
                    pass
            tags.append(f'global_iter={it}')
        elif loop_type == 'IterBasedTrainLoop':
            mi = self._max_iters_cfg
            tags.append(
                f'iter={it}' + (f'/{mi}' if mi is not None else ''))
            me = self._max_epochs_cfg
            if me is not None:
                tags.append(f'epoch={runner.epoch + 1}/{me}')
        else:
            tags.append(f'global_iter={it}')

        return ' | '.join(tags)

    def _epoch_label(self, runner):
        cur = runner.epoch + 1
        if self._epoch_total:
            return f'{cur}/{self._epoch_total}'
        return str(cur)

    def _print_train_block(self, runner, batch_idx=None):
        c = self._cache
        if self._train_blocks > 0:
            print(flush=True)

        w_ep, w_dn, w_mem, w_loss, w_lr, w_img = 10, 12, 14, 14, 14, 10
        row1 = (
            self._cell('Epoch', w_ep, 'green')
            + self._cell('data_num', w_dn, 'yellow')
            + self._cell('GPU Mem', w_mem, 'yellow')
            + self._cell('Loss', w_loss, 'yellow')
            + self._cell('LR', w_lr, 'yellow')
            + self._cell('Image_size', w_img, 'yellow'))

        loss = c.get('loss')
        lr = c.get('lr')
        loss_str = f'{loss:.8f}' if loss is not None else '?'
        lr_str = f'{lr:.8f}' if lr is not None else '?'
        img = c.get('image_size')
        img_str = str(img) if img is not None else '?'
        mem = c.get('gpu_mem', 0.0)
        dn = c.get('data_num', '?/?')
        ep = self._epoch_label(runner)
        row2 = (
            self._cell(ep, w_ep, 'bold')
            + self._cell(dn, w_dn, 'white')
            + self._cell(f'{mem:.2f} MB', w_mem, 'white')
            + self._cell(loss_str, w_loss, 'white')
            + self._cell(lr_str, w_lr, 'white')
            + self._cell(img_str, w_img, 'white'))
        tag = self._progress_tags(runner, batch_idx=batch_idx)
        print(self._c(f'[本轮训练结束] {tag}', 'cyan'), flush=True)
        print(row1, flush=True)
        print(row2, flush=True)
        self._train_blocks += 1

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        if not is_main_process():
            return

        message_hub = runner.message_hub
        loss = self._safe_scalar(message_hub, 'train/loss')
        lr = self._safe_lr(runner)
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            gpu_mem = 0.0

        data_num = '?/?'
        total = None
        if hasattr(runner, 'train_dataloader'):
            try:
                total = len(runner.train_dataloader)
                data_num = f'{batch_idx + 1}/{total}'
            except TypeError:
                data_num = f'{batch_idx + 1}/?'

        image_size = self._safe_image_size(data_batch)
        self._cache = dict(
            loss=loss,
            lr=lr,
            gpu_mem=gpu_mem,
            data_num=data_num,
            image_size=image_size,
        )

        epoch_end = (
            total is not None and 0 < total < self._MAX_REASONABLE_EPOCH_LEN
            and (batch_idx + 1) == total)
        if epoch_end:
            self._print_train_block(runner, batch_idx=batch_idx)
            return

        if not self.interval or self.interval <= 0:
            return
        if (runner.iter + 1) % self.interval != 0:
            return
        image_size_str = str(image_size) if image_size is not None else '?'
        loss_str = f'{loss:.6f}' if loss is not None else '?'
        lr_str = f'{lr:.8f}' if lr is not None else '?'
        prog = self._progress_tags(runner, batch_idx=batch_idx)
        print(
            f"{self._c('[训练]', 'green')} {prog} | "
            f"{self._c('batch进度', 'yellow')} {data_num} | "
            f"{self._c('GPU Mem', 'magenta')} {gpu_mem:.2f} MB | "
            f"{self._c('Loss', 'red')} {loss_str} | "
            f"{self._c('LR', 'yellow')} {lr_str} | "
            f"{self._c('Img', 'blue')} {image_size_str}",
            flush=True)

    @staticmethod
    def _get_metric_explicit(metrics, full_key: str):
        if not full_key:
            return None
        return metrics.get(full_key)

    def after_val_epoch(self, runner, metrics=None):
        if not is_main_process() or not isinstance(metrics, dict):
            return

        vtag = self._progress_tags(runner, batch_idx=None)
        print(self._c(f'[验证] {vtag}', 'red'), flush=True)

        w_dn, w_iou, w_f1, w_p, w_r, w_a = 12, 12, 10, 12, 12, 10
        row3 = (
            self._cell('data_num', w_dn, 'red')
            + self._cell('IoU(fg)', w_iou, 'red')
            + self._cell('F1', w_f1, 'red')
            + self._cell('Precision', w_p, 'red')
            + self._cell('Recall', w_r, 'red')
            + self._cell('aAcc', w_a, 'red'))

        def fmt_pct(v):
            if not isinstance(v, numbers.Real):
                return 'N/A'
            x = float(v)
            if x <= 1.0 + 1e-6:
                x = x * 100.0
            return f'{x:.2f}'

        val_total = None
        try:
            val_total = len(runner.val_dataloader)
        except TypeError:
            pass
        dn_val = f'{val_total}/{val_total}' if val_total else '?/?'

        def _one_block(m_iou, m_f1, m_p, m_r, aacc):
            row4 = (
                self._cell(dn_val, w_dn, 'white')
                + self._cell(fmt_pct(m_iou), w_iou, 'white')
                + self._cell(fmt_pct(m_f1), w_f1, 'white')
                + self._cell(fmt_pct(m_p), w_p, 'white')
                + self._cell(fmt_pct(m_r), w_r, 'white')
                + self._cell(fmt_pct(aacc), w_a, 'white'))
            print(row3, flush=True)
            print(row4, flush=True)
            print(flush=True)

        vgroups = self.val_key_groups
        if vgroups:
            for g in vgroups:
                if not isinstance(g, dict):
                    continue
                title = g.get('title', '')
                keys = g.get('keys') or {}
                if title:
                    print(
                        self._c(title, 'yellow'),
                        flush=True)
                miou = self._get_metric_explicit(metrics, keys.get('iou'))
                mf1 = self._get_metric_explicit(metrics, keys.get('f1'))
                mp = self._get_metric_explicit(metrics, keys.get('p'))
                mr = self._get_metric_explicit(metrics, keys.get('r'))
                aacc = self._get_metric_explicit(metrics, keys.get('a'))
                _one_block(miou, mf1, mp, mr, aacc)
            return

        miou = self._pick_metric(metrics, ['IoU', 'mIoU'])
        mf1 = self._pick_metric(metrics, ['F1', 'mFscore', 'mF1'])
        mp = self._pick_metric(metrics, ['Precision', 'mPrecision'])
        mr = self._pick_metric(metrics, ['Recall', 'mRecall'])
        aacc = self._pick_metric(metrics, ['aAcc'])
        _one_block(miou, mf1, mp, mr, aacc)

    def after_train_epoch(self, runner, metrics=None):
        if not is_main_process():
            return
        if self._train_loop_type(runner) != 'EpochBasedTrainLoop':
            return
        hub = runner.message_hub
        loss = self._safe_scalar(hub, 'train/loss')
        lr = self._safe_lr(runner)
        loss_str = f'{loss:.6f}' if loss is not None else '?'
        lr_str = f'{lr:.8f}' if lr is not None else '?'
        ep_done = runner.epoch + 1
        me = self._max_epochs_cfg
        ep_tag = f'{ep_done}/{me}' if me is not None else str(ep_done)
        it = runner.iter + 1
        print(
            self._c('[Epoch]', 'magenta')
            + f' 第 {ep_tag} 轮训练阶段结束 | global_iter={it} | '
            f'loss≈{loss_str} | lr={lr_str} | 随后验证集…',
            flush=True)
        print(flush=True)


def _scalar_first(hub, keys):
    for k in keys:
        v = ConsoleSummaryHook._safe_scalar(hub, k)
        if v is not None:
            return float(v)
    return float('nan')


@HOOKS.register_module()
class PlotMetricsHook(Hook):
    """采样 train/loss 与 UPer 常见子 loss；每次验证追加 val_metrics.csv 并更新折线图。"""

    priority = 'LOW'

    def __init__(self, sample_interval=50, val_branches=None):
        """val_branches: 可选。每项为 ``{ 'log_subdir': 'scheme1_xxx', 'prefix': 'val' }``，
        会分别把 ``{prefix}/IoU`` 等写入 ``log_dir/log_subdir/val_metrics.csv`` 与子目录下趋势图。
        为 None 时与旧行为相同（单份 val 写在 ``log_dir`` 根下）。"""
        self.sample_interval = max(1, int(sample_interval))
        self.val_branches = val_branches
        self._t_iters = []
        self._t_loss = []
        self._t_lr = []
        self._t_decode_ce = []
        self._t_aux_ce = []
        self._v_epoch = []
        self._v_step = []
        self._v_iou = []
        self._v_f1 = []
        self._v_precision = []
        self._v_recall = []
        # { subdir: { 'epoch', 'step', 'iou', ... } }
        self._branch_val = {}

    def before_train(self, runner):
        self._t_iters.clear()
        self._t_loss.clear()
        self._t_lr.clear()
        self._t_decode_ce.clear()
        self._t_aux_ce.clear()
        self._v_epoch.clear()
        self._v_step.clear()
        self._v_iou.clear()
        self._v_f1.clear()
        self._v_precision.clear()
        self._v_recall.clear()
        self._branch_val.clear()
        if self.val_branches:
            for b in self.val_branches:
                if not isinstance(b, dict):
                    continue
                sd = b.get('log_subdir') or b.get('name') or 'branch'
                self._branch_val[sd] = {
                    'epoch': [],
                    'step': [],
                    'iou': [],
                    'f1': [],
                    'p': [],
                    'r': [],
                }

    @staticmethod
    def _log_dir(runner):
        ld = getattr(runner, '_log_dir', None)
        if ld:
            return ld
        ts = getattr(runner, 'timestamp', None)
        if ts and getattr(runner, 'work_dir', None):
            return osp.join(runner.work_dir, ts)
        return getattr(runner, 'work_dir', '.')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not is_main_process():
            return
        if (runner.iter + 1) % self.sample_interval != 0:
            return
        hub = runner.message_hub
        loss = ConsoleSummaryHook._safe_scalar(hub, 'train/loss')
        lr = ConsoleSummaryHook._safe_lr(runner)
        it = runner.iter + 1
        self._t_iters.append(it)
        self._t_loss.append(float(loss) if loss is not None else float('nan'))
        self._t_lr.append(float(lr) if lr is not None else float('nan'))
        dec = _scalar_first(
            hub,
            ('train/decode.loss_ce', 'train/decode.loss_decode',
             'train/loss_ce'))
        aux = _scalar_first(
            hub,
            ('train/aux.loss_ce', 'train/aux.loss_decode'))
        self._t_decode_ce.append(dec)
        self._t_aux_ce.append(aux)

    def after_train_epoch(self, runner):
        if not is_main_process():
            return
        self._save_figure(runner)

    @staticmethod
    def _m_pref(metrics, prefix, short):
        p = prefix.rstrip('/')
        return metrics.get(f'{p}/{short}')

    def _append_val_csv_one(
            self, logd, metrics, ep, step, path, pfx):
        p = pfx.rstrip('/') if pfx else ''
        if p:
            iou = self._m_pref(metrics, p, 'IoU')
            f1 = self._m_pref(metrics, p, 'F1')
            pr = self._m_pref(metrics, p, 'Precision')
            rc = self._m_pref(metrics, p, 'Recall')
            aacc = self._m_pref(metrics, p, 'aAcc')
        else:
            iou = ConsoleSummaryHook._pick_metric(
                metrics, ['IoU', 'mIoU'])
            f1 = ConsoleSummaryHook._pick_metric(
                metrics, ['F1', 'mFscore', 'mF1'])
            pr = ConsoleSummaryHook._pick_metric(
                metrics, ['Precision', 'mPrecision'])
            rc = ConsoleSummaryHook._pick_metric(
                metrics, ['Recall', 'mRecall'])
            aacc = ConsoleSummaryHook._pick_metric(metrics, ['aAcc'])
        vl = metrics.get('val/loss')
        row = [
            ep, step,
            float(iou) if isinstance(iou, numbers.Real) else '',
            float(f1) if isinstance(f1, numbers.Real) else '',
            float(pr) if isinstance(pr, numbers.Real) else '',
            float(rc) if isinstance(rc, numbers.Real) else '',
            float(aacc) if isinstance(aacc, numbers.Real) else '',
            float(vl) if isinstance(vl, numbers.Real) else '',
        ]
        write_header = not osp.isfile(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    'epoch', 'global_iter', 'IoU_fg', 'F1', 'Precision',
                    'Recall', 'aAcc', 'val_loss',
                ])
            w.writerow(row)

    def _append_val_csv(self, runner, metrics):
        logd = self._log_dir(runner)
        os.makedirs(logd, exist_ok=True)
        path = osp.join(logd, 'val_metrics.csv')
        ep = runner.epoch + 1
        step = runner.iter + 1
        self._append_val_csv_one(
            logd, metrics, ep, step, path, '')

    def after_val_epoch(self, runner, metrics=None):
        if not is_main_process():
            return
        if not isinstance(metrics, dict):
            metrics = {}
        step = runner.iter + 1
        ep = runner.epoch + 1
        if self.val_branches and self._branch_val:
            logd = self._log_dir(runner)
            for b in self.val_branches:
                if not isinstance(b, dict):
                    continue
                sub = b.get('log_subdir', 'branch')
                pfx = b.get('prefix', 'val')
                pth = osp.join(logd, str(sub), 'val_metrics.csv')
                os.makedirs(osp.dirname(pth), exist_ok=True)
                self._append_val_csv_one(
                    logd, metrics, ep, step, pth, pfx)
                d = self._branch_val.get(sub, {})
                if d:
                    iou = self._m_pref(metrics, pfx.rstrip('/'), 'IoU')
                    f1 = self._m_pref(metrics, pfx.rstrip('/'), 'F1')
                    pr = self._m_pref(
                        metrics, pfx.rstrip('/'), 'Precision')
                    rc = self._m_pref(
                        metrics, pfx.rstrip('/'), 'Recall')
                    d['epoch'].append(ep)
                    d['step'].append(step)
                    d['iou'].append(
                        float(iou) if isinstance(iou, numbers.Real) else
                        float('nan'))
                    d['f1'].append(
                        float(f1) if isinstance(f1, numbers.Real) else
                        float('nan'))
                    d['p'].append(
                        float(pr) if isinstance(pr, numbers.Real) else
                        float('nan'))
                    d['r'].append(
                        float(rc) if isinstance(rc, numbers.Real) else
                        float('nan'))
        else:
            iou = ConsoleSummaryHook._pick_metric(metrics, ['IoU', 'mIoU'])
            f1 = ConsoleSummaryHook._pick_metric(
                metrics, ['F1', 'mFscore', 'mF1'])
            pr = ConsoleSummaryHook._pick_metric(
                metrics, ['Precision', 'mPrecision'])
            rc = ConsoleSummaryHook._pick_metric(
                metrics, ['Recall', 'mRecall'])
            self._append_val_csv(runner, metrics)
            self._v_epoch.append(ep)
            self._v_step.append(step)
            self._v_iou.append(
                float(iou) if isinstance(iou, numbers.Real) else
                float('nan'))
            self._v_f1.append(
                float(f1) if isinstance(f1, numbers.Real) else
                float('nan'))
            self._v_precision.append(
                float(pr) if isinstance(pr, numbers.Real) else
                float('nan'))
            self._v_recall.append(
                float(rc) if isinstance(rc, numbers.Real) else
                float('nan'))
        self._save_figure(runner)

    def after_train(self, runner):
        if is_main_process():
            self._save_figure(runner)

    def _save_figure(self, runner):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print_log(
                'PlotMetricsHook: 未安装 matplotlib，跳过绘图。可 pip install matplotlib',
                logger='current',
                level=logging.WARNING)
            return

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        exp = getattr(runner, 'experiment_name', 'train')
        fig.suptitle(f'Train (sampled) — {exp}', fontsize=12)

        def _plot_xy(ax, xs, ys, title, ylabel, xlabel='global_iter', style='-'):
            if xs and any(v == v for v in ys):
                ax.plot(xs, ys, style, lw=1, alpha=0.88)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        if self._t_iters:
            _plot_xy(axes[0, 0], self._t_iters, self._t_loss, 'Train loss', 'loss')
            _plot_xy(axes[0, 1], self._t_iters, self._t_lr, 'Learning rate', 'lr', style='g-')
            _plot_xy(
                axes[1, 0], self._t_iters, self._t_decode_ce,
                'Train decode CE', 'loss', style='c-')
            ax = axes[1, 1]
            ax.plot(
                self._t_iters, self._t_aux_ce, '-',
                color='darkorange', lw=1, alpha=0.9, label='aux CE')
            ax.set_title('Train aux CE')
            ax.set_xlabel('global_iter')
            ax.set_ylabel('loss')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'no train samples yet', ha='center', va='center')

        plt.tight_layout()
        logd = self._log_dir(runner)
        os.makedirs(logd, exist_ok=True)
        overview_png = osp.join(logd, 'train_curves.png')
        plt.savefig(overview_png, dpi=150, bbox_inches='tight')
        plt.close()

        if self.val_branches and self._branch_val:
            for sub, d in self._branch_val.items():
                st = d.get('step', [])
                if not st or not any(
                        v == v
                        for v in d.get('iou', [])):
                    continue
                xsv = st
                fig2, ax2 = plt.subplots(2, 2, figsize=(10, 8))
                sufx = f' — {sub}'
                fig2.suptitle(
                    f'Val: foreground IoU / F1 / P / R (%){sufx}', fontsize=11)
                vi, vf, vp, vr = d['iou'], d['f1'], d['p'], d['r']
                ax2[0, 0].plot(xsv, vi, 'r-o', ms=3, lw=1)
                ax2[0, 0].set_title('IoU (fg)')
                ax2[0, 0].set_xlabel('global_iter @ val')
                ax2[0, 0].grid(True, alpha=0.3)
                ax2[0, 1].plot(xsv, vf, 'm-s', ms=3, lw=1)
                ax2[0, 1].set_title('F1 (fg)')
                ax2[0, 1].set_xlabel('global_iter @ val')
                ax2[0, 1].grid(True, alpha=0.3)
                ax2[1, 0].plot(xsv, vp, 'b-^', ms=3, lw=1)
                ax2[1, 0].set_title('Precision (fg)')
                ax2[1, 0].set_xlabel('global_iter @ val')
                ax2[1, 0].grid(True, alpha=0.3)
                ax2[1, 1].plot(xsv, vr, 'g-d', ms=3, lw=1)
                ax2[1, 1].set_title('Recall (fg)')
                ax2[1, 1].set_xlabel('global_iter @ val')
                ax2[1, 1].grid(True, alpha=0.3)
                plt.tight_layout()
                sdir = osp.join(logd, str(sub))
                os.makedirs(sdir, exist_ok=True)
                vpng = osp.join(sdir, 'val_foreground_trends.png')
                fig2.savefig(vpng, dpi=150, bbox_inches='tight')
                plt.close()
        elif self._v_step:
            fig2, ax2 = plt.subplots(2, 2, figsize=(10, 8))
            fig2.suptitle('Val: foreground IoU / F1 / Precision / Recall (%)', fontsize=11)
            xsv = self._v_step
            ax2[0, 0].plot(xsv, self._v_iou, 'r-o', ms=3, lw=1)
            ax2[0, 0].set_title('IoU (fg)')
            ax2[0, 0].set_xlabel('global_iter @ val')
            ax2[0, 0].grid(True, alpha=0.3)
            ax2[0, 1].plot(xsv, self._v_f1, 'm-s', ms=3, lw=1)
            ax2[0, 1].set_title('F1 (fg)')
            ax2[0, 1].set_xlabel('global_iter @ val')
            ax2[0, 1].grid(True, alpha=0.3)
            ax2[1, 0].plot(xsv, self._v_precision, 'b-^', ms=3, lw=1)
            ax2[1, 0].set_title('Precision (fg)')
            ax2[1, 0].set_xlabel('global_iter @ val')
            ax2[1, 0].grid(True, alpha=0.3)
            ax2[1, 1].plot(xsv, self._v_recall, 'g-d', ms=3, lw=1)
            ax2[1, 1].set_title('Recall (fg)')
            ax2[1, 1].set_xlabel('global_iter @ val')
            ax2[1, 1].grid(True, alpha=0.3)
            plt.tight_layout()
            val_png = osp.join(logd, 'val_foreground_trends.png')
            plt.savefig(val_png, dpi=150, bbox_inches='tight')
            plt.close()

        msg = f'PlotMetricsHook: 已更新 {overview_png}；'
        if self.val_branches and self._branch_val:
            for sub in self._branch_val:
                msg += (
                    f' {osp.join(logd, str(sub), "val_metrics.csv")}、'
                    f'{osp.join(logd, str(sub), "val_foreground_trends.png")}；'
                )
        else:
            msg += f' {osp.join(logd, "val_metrics.csv")}；验证趋势 {osp.join(logd, "val_foreground_trends.png")}'
        print_log(msg, logger='current', level=logging.INFO)


@HOOKS.register_module()
class WireSegTestSummaryHook(Hook):
    """在 ``test.py`` 跑多 evaluator 时，测试结束后在终端分块输出与训练时风格一致的表。"""

    priority = 'LOW'

    def __init__(self, val_key_groups=None, title_tag='[测试] '):
        self.val_key_groups = val_key_groups or []
        self.title_tag = str(title_tag) if title_tag is not None else ''
        self._printed = False

    def _fmt_pct(self, v):
        if not isinstance(v, numbers.Real):
            return 'N/A'
        x = float(v)
        if x <= 1.0 + 1e-6:
            x = x * 100.0
        return f'{x:.2f}'

    @staticmethod
    def _g(metrics, k):
        if not k or not isinstance(metrics, dict):
            return None
        return metrics.get(k)

    def _test_data_num(self, runner):
        try:
            tdl = getattr(runner, 'test_dataloader', None)
            n = len(tdl) if tdl is not None else None
            if n:
                return f'{n}/{n}'
        except TypeError:
            pass
        return 'test'

    def _scrape_from_hub(self, runner, keys):
        m = {}
        hub = getattr(runner, 'message_hub', None)
        if hub is None or not keys:
            return m
        get = getattr(hub, 'get_scalar', None)
        if get is None:
            return m
        for k in keys:
            if not k or k in m:
                continue
            try:
                sc = get(k)
                if sc is not None and hasattr(sc, 'current'):
                    m[k] = sc.current()
            except (TypeError, AttributeError, KeyError, RuntimeError):
                continue
        return m

    def _all_metric_keys(self):
        kset = []
        for g in (self.val_key_groups or []):
            if not isinstance(g, dict):
                continue
            for _lk, fk in (g.get('keys') or {}).items():
                if fk and fk not in kset:
                    kset.append(fk)
        return kset

    def _merge_metrics(self, passed, runner):
        keys = self._all_metric_keys()
        m = self._scrape_from_hub(runner, keys) if keys else {}
        if isinstance(passed, dict) and passed:
            for a, b in passed.items():
                m[a] = b
        if not m and isinstance(passed, dict):
            m = {k: v for k, v in passed.items() if v is not None}
        return m

    def _print_vgroups(self, metrics, runner):
        if not self.val_key_groups or self._printed:
            return
        c = ConsoleSummaryHook
        w_dn, w_iou, w_f1, w_p, w_r, w_a = 12, 12, 10, 12, 12, 10
        row3 = (
            c._cell('data_num', w_dn, 'red') + c._cell('IoU(fg)', w_iou, 'red')
            + c._cell('F1', w_f1, 'red') + c._cell('Precision', w_p, 'red')
            + c._cell('Recall', w_r, 'red') + c._cell('aAcc', w_a, 'red'))
        if self.title_tag:
            print(c._c(
                f'{self.title_tag} 测试集 — data_num | IoU | F1 | P | R | aAcc',
                'magenta'),
                flush=True)
        else:
            print(
                c._c('[测试] 测试集 — 指标', 'magenta'),
                flush=True)
        print(flush=True)
        dn_val = self._test_data_num(runner)
        for g in self.val_key_groups:
            if not isinstance(g, dict):
                continue
            t = g.get('title', '') or '指标'
            print(c._c(t, 'yellow'), flush=True)
            keys = (g.get('keys') or {})
            row4 = (
                c._cell(dn_val, w_dn, 'white') + c._cell(
                    self._fmt_pct(self._g(metrics, keys.get('iou'))), w_iou, 'white')
                + c._cell(
                    self._fmt_pct(self._g(metrics, keys.get('f1'))), w_f1, 'white')
                + c._cell(
                    self._fmt_pct(self._g(metrics, keys.get('p'))), w_p, 'white')
                + c._cell(
                    self._fmt_pct(self._g(metrics, keys.get('r'))), w_r, 'white')
                + c._cell(
                    self._fmt_pct(self._g(metrics, keys.get('a'))), w_a, 'white'))
            print(row3, flush=True)
            print(row4, flush=True)
            print(flush=True)
        self._printed = True

    def after_test_epoch(self, runner, metrics=None):
        if not is_main_process() or self._printed:
            return
        passed = metrics if isinstance(metrics, dict) else None
        m = self._merge_metrics(passed or {}, runner)
        if m and self._all_metric_keys():
            self._print_vgroups(m, runner)

    def after_test(self, runner):
        if (not is_main_process() or self._printed or
                not self.val_key_groups):
            return
        m = self._merge_metrics({}, runner)
        if m and self._all_metric_keys():
            self._print_vgroups(m, runner)
