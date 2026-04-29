# 方案3：DataA 输入 256×256、验证按前景 P>0.5（与二分类 argmax 等价，显式走 logits 路径）。
# 与 512 双方案配置独立；单次训练仅单套 ``val/`` 指标，权重与曲线在 ``checkpoint/train_*`` 下（不分子目录）。
# 用法：在 ``semantic_segmentation`` 下
#   python tools/train.py configs/mamba_vision/mamba_vision_tiny_dataa_256x256_wire_t05.py
#
# 与 ``_512_`` 配置的差异：``crop``、多尺度、验证 Resize 短边 256 对齐、``BinaryForegroundThreshIoUMetric(0.5)``。

_base_ = ['./mamba_vision_160k_ade20k-512x512_tiny.py']

custom_imports = dict(imports=['binary_fg_metrics'], allow_failed_imports=False)

wire_seg_experiment = True
wire_seg_checkpoint_to_log_dir = True
wire_seg_iou_early_stop_patience = 50
wire_seg_enable_console = True
wire_seg_console_interval = 1
wire_seg_enable_plots = True
wire_seg_plot_sample_interval = 15
# 单套 checkpoint，不分子目录
wire_seg_checkpoint_subdirs = None
wire_seg_val_console_key_groups = None
wire_seg_plot_val_branches = None

_wire_ts = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
_pretrained = (__import__('os').getenv('MAMBAVISION_TINY_PRETRAINED') or '').strip() or (
    'checkpoint/pretrained/mambavision_tiny_1k.pth.tar'
    if __import__('os').path.isfile(__import__('os').path.join(
        'checkpoint', 'pretrained', 'mambavision_tiny_1k.pth.tar')) else
    'https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar'
)

_WIRE_DATA_ROOT = __import__('os').getenv('WIRE_SEG_DATAA_ROOT') or '../../DataA-B/DataA'
_WIRE_MAX_EPOCHS = 200
_WIRE_VAL_INTERVAL = 1
_WIRE_TRAIN_BATCH = 4
_CKPT_NO_PERIODIC = 10**9
_CROP = 256

work_dir = 'checkpoint'
timestamp = f'train_{_wire_ts}'

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

_wire_meta = dict(
    classes=('background', 'foreground'),
    palette=[[0, 0, 0], [255, 255, 255]])

crop_size = (_CROP, _CROP)
data_preprocessor = dict(size=crop_size)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[int(_CROP * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
# 与 512 的 (2048,512) 成比例
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, _CROP), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=_WIRE_TRAIN_BATCH,
    num_workers=4,
    persistent_workers=True,
    # 最后一 iter 若只有 1 张图，BN 在 train 下会报
    # ``Expected more than 1 value per channel when training``
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=_WIRE_DATA_ROOT,
        data_prefix=dict(img_path='image/train', seg_map_path='mask/train'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=_wire_meta,
        pipeline=train_pipeline))
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
        metainfo=_wire_meta,
        pipeline=test_pipeline))
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
        metainfo=_wire_meta,
        pipeline=test_pipeline))

# 单指标：0.5 显式；与 2 类 argmax 数值一致
val_evaluator = [
    dict(
        type='BinaryForegroundThreshIoUMetric',
        threshold=0.5,
        foreground_index=1,
        iou_metrics=['mIoU', 'mFscore'],
        nan_to_num=0,
        prefix='val'),
]
test_evaluator = val_evaluator

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MM_mamba_vision',
        pretrained=_pretrained,
    ),
    decode_head=dict(
        num_classes=2,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(num_classes=2, norm_cfg=norm_cfg))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00005, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}))

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=_WIRE_MAX_EPOCHS,
        by_epoch=True)
]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=_WIRE_MAX_EPOCHS,
    val_interval=_WIRE_VAL_INTERVAL)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

log_processor = dict(by_epoch=True)

default_hooks = dict(
    _delete_=True,
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=_CKPT_NO_PERIODIC,
        save_best='val/IoU',
        rule='greater',
        save_last=False,
        out_dir='checkpoint',
        filename_tmpl=(
            f'mav_t_dataa256_t05_t{_wire_ts}_epoch{{}}.pth')),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

custom_hooks = []
