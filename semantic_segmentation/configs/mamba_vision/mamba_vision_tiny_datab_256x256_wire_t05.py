# 方案3：DataB 输入 256x256，验证按前景 P>0.5（独立训练任务）。
# 用法（在 ``semantic_segmentation`` 目录）::
#   python tools/train.py configs/mamba_vision/mamba_vision_tiny_datab_256x256_wire_t05.py

_base_ = ['./mamba_vision_tiny_dataa_256x256_wire_t05.py']

_wire_ts = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
_WIRE_DATA_ROOT = __import__('os').getenv('WIRE_SEG_DATAB_ROOT') or '../../DataA-B/DataB'

train_dataloader = dict(
    dataset=dict(
        data_root=_WIRE_DATA_ROOT,
        data_prefix=dict(img_path='image/train', seg_map_path='mask/train')))
val_dataloader = dict(
    dataset=dict(
        data_root=_WIRE_DATA_ROOT,
        data_prefix=dict(img_path='image/val', seg_map_path='mask/val')))
test_dataloader = dict(
    dataset=dict(
        data_root=_WIRE_DATA_ROOT,
        data_prefix=dict(img_path='image/test', seg_map_path='mask/test')))

default_hooks = dict(
    checkpoint=dict(
        filename_tmpl=f'mav_t_datab256_t05_t{_wire_ts}_epoch{{}}.pth'))
timestamp = f'train_{_wire_ts}'
