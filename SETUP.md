# MambaVision 电线二分类 - 环境与安装（SETUP）

本文件只包含环境、依赖、数据自检与冒烟步骤。  
训练/测试命令请看 `MambaVision.md`。

---

## 1) 工程目录与执行位置

- 仓库根：`my_MambaVision-main/my_MambaVision-main`
- 分割工程根：`my_MambaVision-main/my_MambaVision-main/semantic_segmentation`
- 训练/测试命令必须在 `semantic_segmentation/` 下执行。

```bash
cd /root/autodl-tmp/my_MambaVision-main/my_MambaVision-main/semantic_segmentation
```

---

## 2) Conda 环境

```bash
conda activate /root/autodl-tmp/conda_envs/segman
```

如 `conda activate` 失败，先：

```bash
source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/segman
```

---

## 3) 推荐版本（本课题已验证）

- Python: 3.10
- PyTorch: 2.1.2+cu118
- mmengine: 0.10.5
- mmsegmentation: 1.2.2
- mmcv: 2.1.0
- mmdet: 3.3.0
- mamba-ssm: 2.2.4
- transformers: 4.41.2
- tokenizers: 0.19.1
- numpy: 1.26.4
- ftfy: 已安装

---

## 4) 依赖修复（若环境漂移）

如果环境被污染或版本冲突，按顺序执行：

```bash
cd /root/autodl-tmp/my_MambaVision-main/my_MambaVision-main/semantic_segmentation
conda activate /root/autodl-tmp/conda_envs/segman

python -m pip uninstall -y mmseg mmsegmentation mmcv mmcv-full
python -m pip install -U openmim
python -m pip install --no-cache-dir "numpy<2"
mim install "mmcv==2.1.0"
python -m pip install --no-cache-dir "mmsegmentation==1.2.2" "mmengine==0.10.5" "mmdet==3.3.0" ftfy

python -m pip uninstall -y transformers tokenizers
python -m pip install --no-cache-dir --force-reinstall "transformers==4.41.2" "tokenizers==0.19.1"

python -m pip install --no-build-isolation --no-cache-dir "mamba-ssm==2.2.4"
python -m pip install --no-cache-dir "setuptools>=61"
```

---

## 5) 数据路径（默认）

当前默认数据根（相对 `semantic_segmentation/`）：

- DataA: `../../DataA-B/DataA`
- DataB: `../../DataA-B/DataB`
- DataC: `../../../DataC/DataC`（与 `DataA-B` 同级，均在 `my_MambaVision-main/` 上一级；常为数据盘 **`/root/autodl-tmp/DataC/DataC`**）

你也可用环境变量覆盖：

```bash
export WIRE_SEG_DATAA_ROOT=/abs/path/to/DataA
export WIRE_SEG_DATAB_ROOT=/abs/path/to/DataB
export WIRE_SEG_DATAC_ROOT=/abs/path/to/DataC
```

数据目录约定：

- `image/train`, `image/val`, `image/test`
- `mask/train`, `mask/val`, `mask/test`

---

## 6) 一键自检

```bash
cd /root/autodl-tmp/my_MambaVision-main/my_MambaVision-main/semantic_segmentation
conda activate /root/autodl-tmp/conda_envs/segman

python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import mmengine, mmseg, mmcv, mmdet; print(mmengine.__version__, mmseg.__version__, mmcv.__version__, mmdet.__version__)"
python -c "import mamba_ssm; print('mamba_ssm ok')"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('selective_scan_fn ok')"
python -c "import binary_fg_metrics, wire_seg_hooks, training_viz_hooks, ftfy; print('project imports ok')"
```

---

## 7) 冒烟测试（只检查数据集可构建）

```bash
cd /root/autodl-tmp/my_MambaVision-main/my_MambaVision-main/semantic_segmentation
python tools/smoke_test_wire_data.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py
# DataC（划分与 A/B 相同，仅根目录不同）：
# python tools/smoke_test_wire_data.py configs/mamba_vision/mamba_vision_tiny_datac_512x512_wire_iou.py
```

预期会输出：

- `train_dataloader dataset len = ...`
- `val_dataloader dataset len = ...`
- `test_dataloader dataset len = ...`

---

## 8) 常见错误速查

- **`ModuleNotFoundError: No module named 'mmseg.registry'`**  
  通常是环境里装成了 **SegMAN 自带的 editable `mmsegmentation` 0.x**（源码在 `my_SegMAN-.../segmentation`），与 **`train.py` 使用的 MMEngine + mmseg 1.x API** 不匹配。处理：在已激活的 **`segman`** 中执行 **`pip uninstall mmsegmentation`**，再按上文 **§4** 安装 **`mmsegmentation==1.2.2`**、**`mmcv==2.1.0`**（与当前 PyTorch/CUDA 对应的预编译 wheel）。

- `No module named 'mmseg.evaluation'`  
  说明 mmseg 版本太旧（0.x），需切回 1.2.2。

- `ImportError ... GreedySearchDecoderOnlyOutput`  
  说明 transformers 版本过高，固定到 `4.41.2`。

- `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`  
  说明 numpy 过高，降到 `<2`（推荐 1.26.4）。

- `FileNotFoundError: ... DataA/image/train`  
  说明数据根路径不对，检查默认路径或设置 `WIRE_SEG_DATAA_ROOT`。

- `FileNotFoundError: ... DataC/image/train`（或路径中含 `DataC`）  
  说明 DataC 根路径不对，检查默认 `../../../DataC/DataC` 或设置 `WIRE_SEG_DATAC_ROOT`。
