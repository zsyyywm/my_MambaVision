# MambaVision（本仓库说明）

本目录为 **[MambaVision (CVPR 2025)](https://arxiv.org/abs/2407.08083)** **[官方 PyTorch 实现](https://github.com/NVlabs/MambaVision)** 的一份可运行拷贝：保留上游 **分类 / 检测 / 语义分割** 能力，并在 **`semantic_segmentation/`** 内扩展了与课题一致的 **DataA / DataB / DataC 电线二分类**（MambaVision-Tiny + UPerNet + MMSeg）及三套训练—验证方案。

**工程根**：同时包含 **`mambavision/`**、**`semantic_segmentation/`**、**`requirements.txt`** 的目录（常见路径：`my_MambaVision-main/my_MambaVision-main/`）。

---

## 文档分工（请先读本节）

| 文档 | 用途 |
|------|------|
| **本文 `README.md`** | **项目综述**：课题二分类与目录、**官方基准测试结果表**（节选保留）、与 SegMAN / BEVANet 数据约定对照。**不写长安装与长命令块。** |
| **[`SETUP.md`](SETUP.md)** | **环境与依赖**：Conda、PyTorch、MMSeg / mmcv、mamba-ssm、`pip`、**数据与预训练权重路径**、自检与冒烟。 |
| **[`MambaVision.md`](MambaVision.md)** | **命令速查**：进入 `semantic_segmentation/` 后可复制的 **训练 / 测试 / 推理 / 续训**；三方案与 Hook 索引。 |

若以 **电线二分类** 为主，请先 **`SETUP.md`** → **`MambaVision.md`**。

---

## 重要目录与产物

| 路径 | 说明 |
|------|------|
| **`semantic_segmentation/`** | **日常命令在此执行**：`tools/train.py`、`tools/test.py`、`tools/infer_one.py`、`configs/mamba_vision/`。 |
| **`semantic_segmentation/checkpoint/train_<时间戳>/`**（或仓库根 **`data/checkpoints1/train_<时间戳>/`**） | 二分类训练 **work_dir**（日志、`best_*.pth`、`val_metrics.csv`）；课题归档时常用 **`data/checkpoints1/`**。 |
| **`data/checkpoints2/test_<时间戳>/`**（相对工程根：`my_MambaVision-main/my_MambaVision-main/`） | **`tools/test.py`** 默认测试输出（与同次 **`train_<时间戳>`** 对齐时见 [`MambaVision.md`](MambaVision.md)）。 |
| **`../../DataA-B/DataA`**、**`../../DataA-B/DataB`**（相对 `semantic_segmentation/`） | 默认 DataA / DataB 根；可用 **`WIRE_SEG_DATAA_ROOT`** / **`WIRE_SEG_DATAB_ROOT`** 覆盖（见 SETUP）。 |
| **`../../../DataC/DataC`** 或 **`WIRE_SEG_DATAC_ROOT`** | DataC；AutoDL 上常见 **`/root/autodl-tmp/DataC/DataC`**。 |
| **`semantic_segmentation/checkpoint/pretrained/mambavision_tiny_1k.pth.tar`** | MambaVision-Tiny-1K 骨干（可选用 `tools/download_mambavision_pretrained.py` 或 `MAMBAVISION_TINY_PRETRAINED`）。 |
| **`semantic_segmentation/binary_fg_metrics.py`**、**`wire_seg_hooks.py`**、**`training_viz_hooks.py`** | 前景 IoU / 阈值指标、early stop 与测试导出、训练可视化。 |

**云服务器（AutoDL）**：工作区多在 **`/root/autodl-tmp`**；环境与 Conda 路径示例见 [`SETUP.md`](SETUP.md)。

**Git**：勿提交大数据集、整份 `checkpoint/`、体积过大的 `*.pth` / `*.pth.tar`；依赖本地 `.gitignore`。

---

## 1. 任务与模型概要

| 项目 | 说明 |
|------|------|
| **上游能力** | ImageNet 分类、**COCO 检测**（本仓库 `object_detection/`）、**ADE20K 等语义分割**（`semantic_segmentation/` 官方 UPerNet 配置）。 |
| **本课题二分类** | **前景 / 背景**；配置在 **`configs/mamba_vision/`** 下 `*_dataa_*` / `*_datab_*` / `*_datac_*`；与 SegMAN / BEVANet 侧 **DataA-B-C**、`mask` **0/1** 约定对齐。 |
| **骨干** | **MambaVision-T**（电线实验默认）；预训练见上表与 SETUP。 |
| **解码器** | **UPerNet**（MMSeg）。 |

实现致谢：**NVIDIA MambaVision**、**OpenMMLab MMSegmentation**、**timm** 等（详见原论文与上游仓库）。

---

## 2. 二分类三方案（与 SegMAN / BEVANet 对齐）

| 方案 | 配置示例 | 输入 / 验证要点 | 验证解码与主监控键 |
|------|-----------|----------------|-------------------|
| **一** | `mamba_vision_tiny_*_512x512_wire_iou.py` | 512，`Resize(2048,512)` keep ratio | **argmax**；**`val/IoU`**（前景 IoU）。 |
| **二** | `mamba_vision_tiny_*_512x512_wire_scheme2.py` | 同上 | 前景 **P > 0.55**；**`val_t055/IoU`**。 |
| **三** | `mamba_vision_tiny_*_256x256_wire_t05.py` | 256 基尺度 | **`BinaryForegroundThreshIoUMetric(0.5)`**（与二分类 argmax 等价路径）。 |

`*` 可为 `dataa` / `datab` / `datac`。三种方案为**独立配置、独立训练**；测试时 **config 与 checkpoint 须同方案**。详细命令见 **[`MambaVision.md`](MambaVision.md)**。

**指标提示**：日志中的 **mIoU** 常为两类 IoU 的算术平均；选优 / 早停请以各方案配置的 **`save_best`** 与 **`val/IoU` / `val_t055/IoU`** 为准，勿与「只看 mIoU」混读。

---

## 3. 官方基准：分类与分割、检测（节选，保留论文/仓库结果）

下列数值来自 **NVlabs / 论文** published 结果，便于与公开报告对照；本地二分类数值以 **`checkpoint/`** 与 **`MambaVision.md`** 为准。

### ImageNet-21K

<table>
  <tr>
    <th>Name</th>
    <th>Acc@1(%)</th>
    <th>Acc@5(%)</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
    <th>Resolution</th>
    <th>HF</th>
    <th>Download</th>
  </tr>
<tr>
    <td>MambaVision-B-21K</td>
    <td>84.9</td>
    <td>97.5</td>
    <td>97.7</td>
    <td>15.0</td>
    <td>224x224</td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-B-21K">link</a></td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-B-21K/resolve/main/mambavision_base_21k.pth.tar">model</a></td>
</tr>
<tr>
    <td>MambaVision-L-21K</td>
    <td>86.1</td>
    <td>97.9</td>
    <td>227.9</td>
    <td>34.9</td>
    <td>224x224</td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L-21K">link</a></td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L-21K/resolve/main/mambavision_large_21k.pth.tar">model</a></td>
</tr>
<tr>
    <td>MambaVision-L2-512-21K</td>
    <td>87.3</td>
    <td>98.4</td>
    <td>241.5</td>
    <td>196.3</td>
    <td>512x512</td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L2-512-21K">link</a></td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L2-512-21K/resolve/main/mambavision_L2_21k_240m_512.pth.tar">model</a></td>
</tr>
<tr>
    <td>MambaVision-L3-256-21K</td>
    <td>87.3</td>
    <td>98.3</td>
    <td>739.6</td>
    <td>122.3</td>
    <td>256x256</td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L3-256-21K">link</a></td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L3-256-21K/resolve/main/mambavision_L3_21k_740m_256.pth.tar">model</a></td>
</tr>
<tr>
    <td>MambaVision-L3-512-21K</td>
    <td>88.1</td>
    <td>98.6</td>
    <td>739.6</td>
    <td>489.1</td>
    <td>512x512</td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L3-512-21K">link</a></td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L3-512-21K/resolve/main/mambavision_L3_21k_740m_512.pth.tar">model</a></td>
</tr>
</table>

### ImageNet-1K

<table>
  <tr>
    <th>Name</th>
    <th>Acc@1(%)</th>
    <th>Acc@5(%)</th>
    <th>Throughput(Img/Sec)</th>
    <th>Resolution</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
    <th>HF</th>
    <th>Download</th>
  </tr>
<tr>
    <td>MambaVision-T</td>
    <td>82.3</td>
    <td>96.2</td>
    <td>6298</td>
    <td>224x224</td>
    <td>31.8</td>
    <td>4.4</td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-T-1K">link</a></td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar">model</a></td>
</tr>
<tr>
    <td>MambaVision-T2</td>
    <td>82.7</td>
    <td>96.3</td>
    <td>5990</td>
    <td>224x224</td>
    <td>35.1</td>
    <td>5.1</td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-T2-1K">link</a></td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar">model</a></td>
</tr>
<tr>
    <td>MambaVision-S</td>
    <td>83.3</td>
    <td>96.5</td>
    <td>4700</td>
    <td>224x224</td>
    <td>50.1</td>
    <td>7.5</td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-S-1K">link</a></td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar">model</a></td>
</tr>
<tr>
    <td>MambaVision-B</td>
    <td>84.2</td>
    <td>96.9</td>
    <td>3670</td>
    <td>224x224</td>
    <td>97.7</td>
    <td>15.0</td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-B-1K">link</a></td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar">model</a></td>
</tr>
<tr>
    <td>MambaVision-L</td>
    <td>85.0</td>
    <td>97.1</td>
    <td>2190</td>
    <td>224x224</td>
    <td>227.9</td>
    <td>34.9</td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L-1K">link</a></td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar">model</a></td>
</tr>
<tr>
    <td>MambaVision-L2</td>
    <td>85.3</td>
    <td>97.2</td>
    <td>1021</td>
    <td>224x224</td>
    <td>241.5</td>
    <td>37.5</td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L2-1K">link</a></td>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L2-1K/resolve/main/mambavision_large2_1k.pth.tar">model</a></td>
</tr>
</table>

### COCO：检测（Cascade Mask R-CNN，节选）

<table>
  <tr>
    <th>Backbone</th>
    <th>Detector</th>
    <th>Lr Schd</th>
    <th>box mAP</th>
    <th>mask mAP</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
    <th>Config</th>
    <th>Log</th>
    <th>Model Ckpt</th>
  </tr>
<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-T-1K">MambaVision-T-1K</a></td>
    <td>Cascade Mask R-CNN</td>
    <td>3x</td>
    <td>51.1</td>
    <td>44.3</td>
    <td>86</td>
    <td>740</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/object_detection/configs/mamba_vision/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/object_detection/tools/work_dirs/cascade_mask_rcnn_mamba_vision_tiny_3x_coco/20250607_142007/20250607_142007.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_tiny_3x_coco/resolve/main/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.pth">model</a></td>
</tr>
<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-S-1K">MambaVision-S-1K</a></td>
    <td>Cascade Mask R-CNN</td>
    <td>3x</td>
    <td>52.3</td>
    <td>45.2</td>
    <td>108</td>
    <td>828</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/object_detection/configs/mamba_vision/cascade_mask_rcnn_mamba_vision_small_3x_coco.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/object_detection/tools/work_dirs/cascade_mask_rcnn_mamba_vision_small_3x_coco/20250607_144612/20250607_144612.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_tiny_3x_coco/resolve/main/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.pth">model</a></td>
</tr>
<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-B-1K">MambaVision-B-1K</a></td>
    <td>Cascade Mask R-CNN</td>
    <td>3x</td>
    <td>52.8</td>
    <td>45.7</td>
    <td>145</td>
    <td>964</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/object_detection/configs/mamba_vision/cascade_mask_rcnn_mamba_vision_base_3x_coco.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/object_detection/tools/work_dirs/cascade_mask_rcnn_mamba_vision_base_3x_coco/20250607_145939/20250607_145939.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/cascade_mask_rcnn_mamba_vision_base_3x_coco/resolve/main/cascade_mask_rcnn_mamba_vision_base_3x_coco.pth">model</a></td>
</tr>
</table>

### ADE20K：语义分割（UPerNet，节选）

<table>
  <tr>
    <th>Backbone</th>
    <th>Method</th>
    <th>Lr Schd</th>
    <th>mIoU</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
    <th>Config</th>
    <th>Log</th>
    <th>Model Ckpt</th>
  </tr>
<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-T-1K">MambaVision-T-1K</a></td>
    <td>UPerNet</td>
    <td>160K</td>
    <td>46.0</td>
    <td>55</td>
    <td>945</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/configs/mamba_vision/mamba_vision_160k_ade20k-512x512_tiny.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/tools/logs/mamba_vision_160k_ade20k-512x512_tiny.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/mamba_vision_160k_ade20k-512x512_tiny/resolve/main/mamba_vision_160k_ade20k-512x512_tiny.pth">model</a></td>
</tr>
<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-S-1K">MambaVision-S-1K</a></td>
    <td>UPerNet</td>
    <td>160K</td>
    <td>48.2</td>
    <td>84</td>
    <td>1135</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/configs/mamba_vision/mamba_vision_160k_ade20k-512x512_small.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/tools/logs/mamba_vision_160k_ade20k-512x512_small.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/mamba_vision_160k_ade20k-512x512_small/resolve/main/mamba_vision_160k_ade20k-512x512_small.pth">model</a></td>
</tr>
<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-B-1K">MambaVision-B-1K</a></td>
    <td>UPerNet</td>
    <td>160K</td>
    <td>49.1</td>
    <td>126</td>
    <td>1342</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/configs/mamba_vision/mamba_vision_160k_ade20k-512x512_base.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/tools/logs/mamba_vision_160k_ade20k-512x512_base.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/mamba_vision_160k_ade20k-512x512_base/resolve/main/mamba_vision_160k_ade20k-512x512_base.pth">model</a></td>
</tr>
<tr>
    <td><a href="https://huggingface.co/nvidia/MambaVision-L3-512-21K">MambaVision-L3-512-21K</a></td>
    <td>UPerNet</td>
    <td>160K</td>
    <td>53.2</td>
    <td>780</td>
    <td>3670</td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/configs/mamba_vision/mamba_vision_160k_ade20k-640x640_l3_21k.py">config</a></td>
    <td><a href="https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/tools/logs/mamba_vision_160k_ade20k-640x640_l3_21k.log">log</a></td>
    <td><a href="https://huggingface.co/nvidia/mamba_vision_160k_ade20k-640x640_l3_21k/resolve/main/mamba_vision_160k_ade20k-640x640_l3_21k.pth">model</a></td>
</tr>
</table>

更多变体、Colab、**`pip install mambavision`** 分类 API、ImageNet 验证命令等见上游 **[NVlabs/MambaVision](https://github.com/NVlabs/MambaVision)**；本仓库分类入口亦可使用根目录 **`validate.py`** / **`mambavision/`**。

---

## 4. 本环境二分类测试记录（来自 `data/checkpoints2`）

下列数值来自 **`semantic_segmentation/tools/test.py`** 在 **test** 集上的一次完整评估（与各自训练时的 `test_dataloader` 一致）；数据根为默认相对路径（DataA：`../../DataA-B/DataA`，DataB：`../../DataA-B/DataB`）。

**mIoU** 为背景类 IoU 与前景类 IoU 的算术平均（由各次 **`*.log`** 中 `per class results` 表读取两类 **IoU** 列相加除 2）。**IoU<sub>fg</sub>** 等指标与 **`test_<时间戳>/<子时间戳>/<子时间戳>.json`** 中字段一致：**方案二**列出的是 **`val_t055/*`**（前景 softmax 阈值 0.55 分支）。

细化日志、配置快照与 MMSeg JSON 位于：  
`<本仓库>/data/checkpoints2/test_<YYYYMMDD_HHmmss>/<运行时子目录>/`。

| data/checkpoints2 | 对应 data/checkpoints1 | 数据集 / 方案（配置） | mIoU | IoU<sub>fg</sub> | Precision<sub>fg</sub> | Recall<sub>fg</sub> | F1<sub>fg</sub> |
|---------------------|------------------------|------------------------|------|------------------|------------------------|---------------------|-----------------|
| `test_20260428_210621` | `train_20260428_210621` | DataA / 方案1 · `mamba_vision_tiny_dataa_512x512_wire_iou.py` | 79.83 | 60.11 | 71.97 | 78.50 | 75.09 |
| `test_20260428_210627` | `train_20260428_210627` | DataA / 方案2 · `mamba_vision_tiny_dataa_512x512_wire_scheme2.py` | 80.69 | 61.80 | 76.42 | 76.35 | 76.39 |
| `test_20260428_210635` | `train_20260428_210635` | DataA / 方案3 · `mamba_vision_tiny_dataa_256x256_wire_t05.py` | 67.84 | 36.45 | 56.38 | 50.76 | 53.42 |
| `test_20260428_210648` | `train_20260428_210648` | DataB / 方案2 · `mamba_vision_tiny_datab_512x512_wire_scheme2.py` | 88.56 | 78.20 | 88.05 | 87.49 | 87.77 |
| `test_20260428_210654` | `train_20260428_210654` | DataB / 方案3 · `mamba_vision_tiny_datab_256x256_wire_t05.py` | 86.85 | 74.98 | 84.29 | 87.16 | 85.70 |
| `test_20260429_082416` | `train_20260429_082416` | DataB / 方案1 · `mamba_vision_tiny_datab_512x512_wire_iou.py` | 88.96 | 78.97 | 87.71 | 88.79 | 88.25 |
| `test_20260429_140111` | `train_20260429_140111` | DataC / 方案3 · `mamba_vision_tiny_datac_256x256_wire_t05.py` | 70.42 | 42.27 | 73.03 | 50.09 | 59.43 |

复现命令见 **[`MambaVision.md`](MambaVision.md)**「测试」节；训练权重见 **`data/checkpoints1/train_<时间戳>/`**（本表第二列）。

---

## 5. 与课题内 SegMAN / BEVANet 的对照（简要）

| 项目 | BEVANet | SegMAN 本仓库 | 本仓库 MambaVision |
|------|---------|----------------|----------------------|
| 框架 | 自有 `wire.py` + yacs | **MMSeg** `train.py` + mmcv | **MMSeg** `train.py` + mmcv |
| 主线骨干 | — | SegMAN Encoder | **MambaVision-T** + UPerNet |

三者共享 **DataA / DataB / DataC** 数据布局时，可进行 **Transformer / SSM vs Mamba 骨干**横向对比。

---

## 6. 引用

```bibtex
@inproceedings{hatamizadeh2025mambavision,
  title={Mambavision: A hybrid mamba-transformer vision backbone},
  author={Hatamizadeh, Ali and Kautz, Jan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={25261--25270},
  year={2025}
}
```

---

## 7. 许可与致谢

上游版权与许可证见 **[LICENSE](LICENSE)**（NVIDIA Source Code License-NC）；预训练权重另见 **[CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** 说明。timm / ImageNet 等见 **[NVlabs/MambaVision README](https://github.com/NVlabs/MambaVision)**。

架构示意图与 teaser 仍可在上游仓库 **`mambavision/assets/`** 查看：`![arch](./mambavision/assets/arch.png)`。
