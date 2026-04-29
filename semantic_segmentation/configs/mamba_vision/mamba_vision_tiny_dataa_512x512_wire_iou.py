# MambaVision-Tiny + UPerNet，DataA 电线/背景二分类；**仅按验证集前景 IoU（val/IoU）存 best**。
# 主干结构同官方 ``mamba_vision_160k_ade20k-512x512_tiny.py``，仅数据、类别数、训练调度与 Hook 策略为课题定制。
#
# 用法（在 ``semantic_segmentation`` 目录）::
#   python tools/train.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py
# 测试（将 CHECKPOINT 换为 ``checkpoint/train_<时间戳>/`` 下最新的 ``best_*.pth``）::
#   python tools/test.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py CHECKPOINT --work-dir checkpoint/train_<时间戳>/test_eval
#
# 单张图可视化（与 TransNeXt infer_one 一致）::
#   python tools/infer_one.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py --img path/to.jpg --best
#
# 数据：默认相对本仓库 ``my_TransNext/DataA-B/DataA``（由 ``_WIRE_DATA_ROOT`` 调整）。
# 预训练（优先级）：① 环境变量 ``MAMBAVISION_TINY_PRETRAINED``（绝对或相对路径）② 若存在
# ``checkpoint/pretrained/mambavision_tiny_1k.pth.tar`` 则直接用本地文件 ③ 否则用 HF URL。
# 一次性下载到②：在 ``semantic_segmentation`` 下执行
# ``python tools/download_mambavision_pretrained.py``（需能访问外网）。
#
# 训练：最多 200 epoch；每 epoch 验证。当前配置仅对应「方案1」：
#  512 + argmax 前景 IoU（``val/IoU``）；方案2/方案3 请使用各自独立配置。
#  连续 50 次验证 ``val/IoU`` 未刷新最优则早停（见 ``wire_seg_hooks``）。

_base_ = ['./mamba_vision_160k_ade20k-512x512_tiny.py']

custom_imports = dict(
    imports=['binary_fg_metrics'],
    allow_failed_imports=False)

wire_seg_experiment = True
wire_seg_checkpoint_to_log_dir = True
wire_seg_iou_early_stop_patience = 50
# 终端彩色摘要 + 验证表；log 目录内 val_metrics.csv、train_curves.png、val_foreground_trends.png
wire_seg_enable_console = True
wire_seg_console_interval = 1
wire_seg_enable_plots = True
# 小数据集约 40 iter/epoch，采样略密便于曲线
wire_seg_plot_sample_interval = 15
wire_seg_checkpoint_subdirs = None
wire_seg_val_console_key_groups = None
wire_seg_plot_val_branches = None

# 勿使用 ``os.environ.get`` 赋给变量：合并进 cfg 后 ``pretty_text`` 会序列化出
# ``<bound method ...>``，yapf 无法解析。用 ``getenv`` 内联即可。
_wire_ts = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
_pretrained = (__import__('os').getenv('MAMBAVISION_TINY_PRETRAINED') or '').strip() or (
    'checkpoint/pretrained/mambavision_tiny_1k.pth.tar'
    if __import__('os').path.isfile(__import__('os').path.join(
        'checkpoint', 'pretrained', 'mambavision_tiny_1k.pth.tar')) else
    'https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar')

# 自 ``semantic_segmentation/`` 起算：上至 ``my_TransNext/DataA-B/DataA``
_WIRE_DATA_ROOT = __import__('os').getenv('WIRE_SEG_DATAA_ROOT') or '../../DataA-B/DataA'
_WIRE_MAX_EPOCHS = 200
_WIRE_VAL_INTERVAL = 1
_WIRE_TRAIN_BATCH = 4
_CKPT_NO_PERIODIC = 10**9

work_dir = 'checkpoint'
timestamp = f'train_{_wire_ts}'

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

_wire_meta = dict(
    classes=('background', 'foreground'),
    palette=[[0, 0, 0], [255, 255, 255]])
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=_WIRE_TRAIN_BATCH,
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

val_evaluator = [
    dict(
        type='BinaryForegroundIoUMetric',
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
            f'mav_t_dataa512_s1_argmax_t{_wire_ts}_epoch{{}}.pth')),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

custom_hooks = []
