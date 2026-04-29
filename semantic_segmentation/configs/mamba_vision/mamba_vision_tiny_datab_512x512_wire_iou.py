# 方案1：DataB，512 + argmax（与 ``mamba_vision_tiny_dataa_512x512_wire_iou.py`` 语义一致）。
# 用法：``python tools/train.py configs/mamba_vision/mamba_vision_tiny_datab_512x512_wire_iou.py``
# 数据根目录可用环境变量 ``WIRE_SEG_DATAB_ROOT`` 覆盖。

_base_ = ['./mamba_vision_tiny_dataa_512x512_wire_iou.py']

_wire_ts = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
_WIRE_DATA_ROOT = __import__('os').getenv('WIRE_SEG_DATAB_ROOT') or '../../DataA-B/DataB'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=_WIRE_DATA_ROOT,
        data_prefix=dict(img_path='image/train', seg_map_path='mask/train'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=dict(
            classes=('background', 'foreground'),
            palette=[[0, 0, 0], [255, 255, 255]]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='RandomChoiceResize',
                scales=[int(512 * x * 0.1) for x in range(5, 21)],
                resize_type='ResizeShortestEdge',
                max_size=2048),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=_WIRE_DATA_ROOT,
        data_prefix=dict(img_path='image/val', seg_map_path='mask/val'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=dict(
            classes=('background', 'foreground'),
            palette=[[0, 0, 0], [255, 255, 255]]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 512), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=_WIRE_DATA_ROOT,
        data_prefix=dict(img_path='image/test', seg_map_path='mask/test'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=dict(
            classes=('background', 'foreground'),
            palette=[[0, 0, 0], [255, 255, 255]]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 512), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs')
        ]))

_CKPT_NO = 10**9
default_hooks = dict(
    _delete_=True,
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=_CKPT_NO,
        save_best='val/IoU',
        rule='greater',
        save_last=False,
        out_dir='checkpoint',
        filename_tmpl=(
            f'mav_t_datab512_s1_argmax_t{_wire_ts}_epoch{{}}.pth')),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
custom_hooks = []
timestamp = f'train_{_wire_ts}'
