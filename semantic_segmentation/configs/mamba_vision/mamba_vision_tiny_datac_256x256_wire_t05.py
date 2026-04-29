# 方案3：DataC 输入 256x256，验证按前景 P>0.5（独立训练任务）。
# 用法（在 ``semantic_segmentation`` 目录）::
#   python tools/train.py configs/mamba_vision/mamba_vision_tiny_datac_256x256_wire_t05.py

_base_ = ['./mamba_vision_tiny_dataa_256x256_wire_t05.py']

_wire_ts = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
_WIRE_DATA_ROOT = __import__('os').getenv('WIRE_SEG_DATAC_ROOT') or '../../../DataC/DataC'
custom_imports = dict(
    imports=['binary_fg_metrics', 'wire_label_transforms'],
    allow_failed_imports=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='MapWireMask255To1'),
    dict(
        type='RandomChoiceResize',
        scales=[int(256 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 256), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='MapWireMask255To1'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    dataset=dict(
        data_root=_WIRE_DATA_ROOT,
        pipeline=train_pipeline,
        data_prefix=dict(img_path='image/train', seg_map_path='mask/train')))
val_dataloader = dict(
    dataset=dict(
        data_root=_WIRE_DATA_ROOT,
        pipeline=test_pipeline,
        data_prefix=dict(img_path='image/val', seg_map_path='mask/val')))
test_dataloader = dict(
    dataset=dict(
        data_root=_WIRE_DATA_ROOT,
        pipeline=test_pipeline,
        data_prefix=dict(img_path='image/test', seg_map_path='mask/test')))

default_hooks = dict(
    checkpoint=dict(
        filename_tmpl=f'mav_t_datac256_t05_t{_wire_ts}_epoch{{}}.pth'))
timestamp = f'train_{_wire_ts}'
