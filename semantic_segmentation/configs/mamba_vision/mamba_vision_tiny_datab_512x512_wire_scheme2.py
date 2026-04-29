# 方案2：DataB，512 + 前景概率阈值 P>0.55（独立训练任务）。
# 用法（在 ``semantic_segmentation`` 目录）::
#   python tools/train.py configs/mamba_vision/mamba_vision_tiny_datab_512x512_wire_scheme2.py

_base_ = ['./mamba_vision_tiny_datab_512x512_wire_iou.py']

_wire_ts = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
_CKPT_NO_PERIODIC = 10**9

wire_seg_val_console_key_groups = [
    dict(
        title='[方案2] 512 验证 前景P>0.55',
        keys=dict(
            iou='val_t055/IoU',
            f1='val_t055/F1',
            p='val_t055/Precision',
            r='val_t055/Recall',
            a='val_t055/aAcc',
        )),
]
wire_seg_plot_val_branches = [dict(log_subdir='', prefix='val_t055')]

val_evaluator = [
    dict(
        type='BinaryForegroundThreshIoUMetric',
        threshold=0.55,
        foreground_index=1,
        iou_metrics=['mIoU', 'mFscore'],
        nan_to_num=0,
        prefix='val_t055'),
]
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=_CKPT_NO_PERIODIC,
        save_best='val_t055/IoU',
        rule='greater',
        save_last=False,
        out_dir='checkpoint',
        filename_tmpl=(
            f'mav_t_datab512_s2_pfg055_t{_wire_ts}_epoch{{}}.pth')))

custom_hooks = []
