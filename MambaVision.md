# MambaVision 课题备忘

本文件位于 **`MambaVision-main/MambaVision-main/`**（与官方 `mambavision/`、`semantic_segmentation/` 同级），描述 **`semantic_segmentation/`** 内与 **DataA / DataB / DataC 电线二分类语义分割** 相关的定制内容。与上级仓库根目录的 **`README.md`**（TransNeXt + Mask2Former 主线，路径约为 `../../README.md`）并列阅读。数据不外传，本文只记路径、命令与约定。

---

## 0. 从进入环境到运行（可直接复制）

下面这段用于每次开新终端后的标准起手式：

```bash
cd /root/autodl-tmp/my_MambaVision-main/my_MambaVision-main/semantic_segmentation
conda activate /root/autodl-tmp/conda_envs/segman

# 快速自检（建议先跑）
python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import mmengine, mmseg, mmcv, mmdet; print(mmengine.__version__, mmseg.__version__, mmcv.__version__, mmdet.__version__)"
python -c "import mamba_ssm, ftfy; print('mamba_ssm+ftfy ok')"
python tools/smoke_test_wire_data.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py
# DataC：python tools/smoke_test_wire_data.py configs/mamba_vision/mamba_vision_tiny_datac_512x512_wire_iou.py
```

若最后一行出现 `train/val/test dataset len` 三行数字，说明环境与数据可用。

---

## 项目总结

**任务**：与 TransNeXt 侧对齐的 **二类语义分割**（前景 / 背景），数据为 **`DataA-B/DataA`、`DataA-B/DataB`**，以及 **`DataC/DataC`**（与 `DataA-B` 同属 `my_MambaVision-main/` 上一级；默认相对路径见下）。划分与标注约定与 DataA/DataB **一致**，仅磁盘路径不同时可换根目录。

**模型**：在官方 **MambaVision-Tiny + UPerNet（MMSeg）** 上实验；**不改 Mamba 主干结构**，仅改数据管线、类别数、优化与 Hook。以下训练方案可在本文对应配置中选用（见「训练方案」节）：  
- **方案1**（512）：与 GitHub/默认一致，**argmax** 多类预测后算前景 IoU，主指标为 **`val/IoU`**。  
- **方案2**（512）：**独立训练任务**，前景类 **softmax 概率 > 0.55** 二值化后计算指标，主键为 **`val_t055/IoU`**。  
- **方案3**（256）：**独立训练任务**；输入 **256×256**；用 **`BinaryForegroundThreshIoUMetric(0.5)`**（二分类下与 argmax 决策等价、实现上走 logits+阈值路径）。

三种方案互斥：一次训练只对应一个方案。每次 run 的主日志与权重都落在 `checkpoint/train_<时间戳>/` 下；方案1/2 通过不同 `filename_tmpl` 前缀（如 `mav_t_*_s1_argmax_*` 与 `mav_t_*_s2_pfg055_*`）区分。早停始终跟随当前配置的 `save_best` 监控键（方案1 为 `val/IoU`，方案2 为 `val_t055/IoU`）。

**日常改代码与执行命令的目录**（下文相对路径均相对此处）：

`semantic_segmentation/`

---

## 目录与关键文件

| 说明 | 路径（相对本目录 `MambaVision-main/MambaVision-main/`） |
|------|----------------------------------------------------------|
| 分割工程根（训练/测试请先 `cd` 到这里） | `semantic_segmentation/` |
| DataA 配置（512，方案1） | `semantic_segmentation/configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py` |
| DataA 配置（512，方案2） | `semantic_segmentation/configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_scheme2.py` |
| DataB 配置（512，方案1） | `semantic_segmentation/configs/mamba_vision/mamba_vision_tiny_datab_512x512_wire_iou.py` |
| DataB 配置（512，方案2） | `semantic_segmentation/configs/mamba_vision/mamba_vision_tiny_datab_512x512_wire_scheme2.py` |
| DataC 配置（512，方案1） | `semantic_segmentation/configs/mamba_vision/mamba_vision_tiny_datac_512x512_wire_iou.py` |
| DataC 配置（512，方案2） | `semantic_segmentation/configs/mamba_vision/mamba_vision_tiny_datac_512x512_wire_scheme2.py` |
| DataA 方案3（256 + 前景 P>0.5 显式） | `semantic_segmentation/configs/mamba_vision/mamba_vision_tiny_dataa_256x256_wire_t05.py` |
| DataB 方案3（256 + 前景 P>0.5 显式） | `semantic_segmentation/configs/mamba_vision/mamba_vision_tiny_datab_256x256_wire_t05.py` |
| DataC 方案3（256 + 前景 P>0.5 显式） | `semantic_segmentation/configs/mamba_vision/mamba_vision_tiny_datac_256x256_wire_t05.py` |
| 前景 IoU 等指标注册 | `semantic_segmentation/binary_fg_metrics.py`（含 `BinaryForegroundIoUMetric`、**`BinaryForegroundThreshIoUMetric`**：基于 `seg_logits` 的阈值前景 IoU 等） |
| 早停、**checkpoint 指到 `log_dir` 或子目录**、**`apply_wire_seg_test_options`（`test.py` 里拆 `--out` 与双表）** | `semantic_segmentation/wire_seg_hooks.py` |
| 彩色终端表、CSV/折线图 Hook（**`val_branches` 时按子目录分写**） | `semantic_segmentation/training_viz_hooks.py` |
| 入口脚本 | `semantic_segmentation/tools/train.py`、`test.py`、`infer_one.py`、`smoke_test_wire_data.py`、`download_mambavision_pretrained.py` |
| 主干注册 | `semantic_segmentation/tools/mamba_vision.py` |

**默认输出**：配置中 `work_dir='checkpoint'`（相对 `semantic_segmentation`）；若在配置 / 环境里将实验根指到 **`data/checkpoints1/`**（常见于数据盘归档），则有 **`data/checkpoints1/train_<时间戳>/`** 作为 **`runner.log_dir`**。  
- 三个方案均为单路 evaluator + 单套 best，`CheckpointToLogDirHook` 会把 `save_best` 与周期权重统一放在同一 run 下（常为 **`train_<时间戳>/<同一时间戳子目录>/best_*.pth`**）。  
- **训练主日志**、`val_metrics.csv`、曲线与 **`best_*.pth`** 与该 run 对齐；具体子目录层级以本次训练控制台「save_best … 将保存到」日志为准。

---

## 训练方案（汇总）

| 方案 | 输入/验证尺度要点 | 验证逻辑 | 配置入口 |
|------|-------------------|----------|----------|
| 1（独立训练） | 训练随机裁剪 **512×512**，验证 pipeline 中 **Resize(2048, 512) keep ratio** | argmax（`val/IoU`） | `…_dataa_/datab_/datac_512x512_wire_iou.py`（DataA/DataB/DataC） |
| 2（独立训练） | 训练随机裁剪 **512×512**，验证 pipeline 中 **Resize(2048, 512) keep ratio** | **P(前景) > 0.55**（`val_t055/IoU`） | `…_dataa_/datab_/datac_512x512_wire_scheme2.py` |
| 3（独立训练） | 裁剪与多尺度以 **256** 为基；验证 **(1024, 256) keep ratio** 与 512 的 (2048,512) 成比例 | **P(前景) > 0.5**（`BinaryForegroundThreshIoUMetric(0.5)`） | `…_dataa_/datab_/datac_256x256_wire_t05.py` |

MMSeg 的 **验证** 需在 `data_sample` 中提供 **`seg_logits`**（与 `BaseSegmentor.postprocess_result` 一致）方案2/3 的阈值指标才严格按概率算；无 logits 时阈值类指标会**回退**为 `pred_sem_seg`（与 argmax 一致）。

**说明**：三种方案均为单独配置与单路 evaluator，命令与产物互不干扰。

---

## 环境与依赖（摘要）

- 建议使用 **Conda**；若 `conda activate` 报错，先执行：  
  `source <你的conda安装路径>/etc/profile.d/conda.sh`  
  再 `conda activate <环境名>`。
- 典型组合：**Python 3.10、PyTorch 2.4.x + cu124、mmsegmentation 1.2.x、mmdet 3.3.x、mmengine 0.10.x**；**`mmcv`** 需与 PyTorch 版本匹配的预编译包（安装失败时勿用源码装一半的环境继续训）。
- **`pip install ftfy`**：构建 `SegLocalVisualizer` 时 `mmseg` 会间接依赖，缺了会在训练初始化阶段报错。
- **`train.py` / `test.py`** 已在脚本开头把 **`semantic_segmentation` 根目录** 与 **`tools/`** 加入 `sys.path`，一般**不必**再手写 `export PYTHONPATH=tools`。

---

## 数据路径与环境变量

- 默认数据根（相对 **`semantic_segmentation`**）：  
  - DataA：`../../DataA-B/DataA`  
  - DataB：`../../DataA-B/DataB`  
  - DataC：`../../../DataC/DataC`（与 `DataA-B` 文件夹同级，位于 `my_MambaVision-main/` 的上一级；本机数据盘常见为 **`/root/autodl-tmp/DataC/DataC`**）
- 可用环境变量覆盖（绝对路径或相对当前 shell 的路径均可）：
  - **`WIRE_SEG_DATAA_ROOT`**：DataA 根目录  
  - **`WIRE_SEG_DATAB_ROOT`**：DataB 根目录  
  - **`WIRE_SEG_DATAC_ROOT`**：DataC 根目录（含 `image/`、`mask/` 的那一层）

目录约定与 TransNeXt 侧一致：`image/{train,val,test}`、`mask/{train,val,test}`，jpg/png。

---

## 预训练权重（MambaVision-Tiny-1K）

配置中 **`model.backbone.pretrained`** 解析顺序：

1. 环境变量 **`MAMBAVISION_TINY_PRETRAINED`**（本地 `.pth.tar` 路径）  
2. 若存在 **`semantic_segmentation/checkpoint/pretrained/mambavision_tiny_1k.pth.tar`**，则只用本地、不联网  
3. 否则使用官方 HuggingFace URL（无网或 errno 会失败）

**一次性下载到默认位置**（在 `semantic_segmentation` 下）：

```bash
cd semantic_segmentation
python tools/download_mambavision_pretrained.py
```

若容器访问外网不稳定，可在本机浏览器下载同名文件后上传到上述路径。官方直链见各配置注释或 NVIDIA HuggingFace 上的 **MambaVision-T-1K** 权重。

---

## 常用命令

以下均在 **`semantic_segmentation`** 目录执行，且已 `conda activate` 到正确环境。

### 冒烟（只检查数据集能否构建、打印样本数）

```bash
cd semantic_segmentation
python tools/smoke_test_wire_data.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py
# DataC：python tools/smoke_test_wire_data.py configs/mamba_vision/mamba_vision_tiny_datac_512x512_wire_iou.py
```

### 训练 DataA / DataB / DataC

```bash
cd semantic_segmentation

# 方案1（DataA，512，argmax）
python tools/train.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py

# 方案2（DataA，512，P>0.55）
python tools/train.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_scheme2.py

# 方案1（DataB，512，argmax）
python tools/train.py configs/mamba_vision/mamba_vision_tiny_datab_512x512_wire_iou.py

# 方案2（DataB，512，P>0.55）
python tools/train.py configs/mamba_vision/mamba_vision_tiny_datab_512x512_wire_scheme2.py

# 方案1（DataC，512，argmax；与 DataB 同策略，仅数据根与 checkpoint 前缀不同）
python tools/train.py configs/mamba_vision/mamba_vision_tiny_datac_512x512_wire_iou.py

# 方案2（DataC，512，P>0.55）
python tools/train.py configs/mamba_vision/mamba_vision_tiny_datac_512x512_wire_scheme2.py

# 方案3（DataA，256，P>0.5）
python tools/train.py configs/mamba_vision/mamba_vision_tiny_dataa_256x256_wire_t05.py

# 方案3（DataB，256，P>0.5）
python tools/train.py configs/mamba_vision/mamba_vision_tiny_datab_256x256_wire_t05.py

# 方案3（DataC，256，P>0.5）
python tools/train.py configs/mamba_vision/mamba_vision_tiny_datac_256x256_wire_t05.py
```

说明：三种方案互斥，按需执行其中一条即可，不会同次并行。DataC 与 A/B **同一套三方案**；若 DataC 不在默认路径，先 `export WIRE_SEG_DATAC_ROOT=/你的/DataC根`。

### 测试

与训练 **一一对应**：**配置必须与本 run 的方案、数据集同名**（`dataa` / `datab` / `datac`）；权重须为该次训练的 **`best*.pth`**。

**推荐写法（课题 `tools/test.py`，在 `semantic_segmentation/` 下）**

- **第 2 个参数**：直接传 **`../data/checkpoints1/train_<时间戳>/`**（该次训练文件夹），脚本会在其下递归查找 **`best*.pth`**，并按文件名 **`_epochN.pth`** 取 **epoch 最大**的那份作为当前最优。  
  也可手写 **单个 `.pth` 路径**。
- **测试输出**：未指定 `--work-dir` 时，写入 **`../data/checkpoints2/test_<同一时间戳>/`**（相对 `semantic_segmentation`；与 `train_<时间戳>` 对齐）。Runner 可能会在子目录再多带一层运行时间戳，属 MMEngine 行为。  
- **覆盖输出目录**：`--work-dir /你的/路径`。  
- 勿使用不存在的绝对路径前缀 **`/data/checkpoints1/…`**（非仓库路径）；请以 **`semantic_segmentation` 相对的 `../data/…`**，或 **`$MV/data/checkpoints1/…`**，`$MV` 为本文档上一级 **`MambaVision-main/MambaVision-main`**）。

**占位符通用模板**

```bash
cd semantic_segmentation
python tools/test.py \
  configs/mamba_vision/<与该 run 一致的配置>.py \
  ../data/checkpoints1/train_<时间戳>
```

**本机六次训练 → 依次测试命令（与时间戳对齐；输出均在 `../../data/checkpoints2/test_<同时间戳>/`）**

```bash
cd /root/autodl-tmp/my_MambaVision-main/my_MambaVision-main/semantic_segmentation

# 1 DataA｜512｜方案1 — train_20260428_210621
python tools/test.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py ../data/checkpoints1/train_20260428_210621

# 2 DataA｜512｜方案2 — train_20260428_210627
python tools/test.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_scheme2.py ../data/checkpoints1/train_20260428_210627

# 3 DataA｜256｜方案3 — train_20260428_210635
python tools/test.py configs/mamba_vision/mamba_vision_tiny_dataa_256x256_wire_t05.py ../data/checkpoints1/train_20260428_210635

# 4 DataB｜512｜方案2 — train_20260428_210648
python tools/test.py configs/mamba_vision/mamba_vision_tiny_datab_512x512_wire_scheme2.py ../data/checkpoints1/train_20260428_210648

# 5 DataB｜256｜方案3 — train_20260428_210654
python tools/test.py configs/mamba_vision/mamba_vision_tiny_datab_256x256_wire_t05.py ../data/checkpoints1/train_20260428_210654

# 6 DataB｜512｜方案1 — train_20260429_082416
python tools/test.py configs/mamba_vision/mamba_vision_tiny_datab_512x512_wire_iou.py ../data/checkpoints1/train_20260429_082416
```

**DataC 新跑出权重后**：将配置换为 **`mamba_vision_tiny_datac_512x512_wire_iou.py` / `_scheme2.py`** 或 **`mamba_vision_tiny_datac_256x256_wire_t05.py`**，`train_<时间戳>` 换为你的 run 文件夹名即可。

**续训／旧式写法**：仍可向 `test.py` 传 **单个 `best_val_*.pth` 路径**；需要指定测试落盘时再 `--work-dir …`。

### 单张图推理（可选 `--best` 自动找 best 权重）

```bash
cd semantic_segmentation
python tools/infer_one.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py \
  --img path/to.jpg --best
```

### 从指定 checkpoint 续训（恢复优化器等）

```bash
cd semantic_segmentation
python tools/train.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py \
  --resume \
  --cfg-options load_from=checkpoint/<路径>/best_val_IoU_xxx.pth
```

若需把日志继续写到原实验目录，可再加 **`--work-dir checkpoint/<原时间戳目录>`**。

续训时 **`load_from` 与 `--work-dir`** 应对应该方案的那次实验时间戳。

---

## 训练时终端会看到什么

- **MMEngine 默认行**：`Epoch(train)[epoch][batch/total]` + `loss`、`decode.loss_ce`、`decode.acc_seg`、`aux.*` 等（与 UPerNet 头有关）。  
- **与 TransNeXt `mask2former/train.py` 对齐的彩色行**：`[训练] …`、`[本轮训练结束]` 表、`[Epoch] … 随后验证集…`、`[验证]` 下输出当前方案对应的一套表。  
- **曲线与表格**：**`train_curves.png`**、`val_metrics.csv`、`val_foreground_trends.png` 在 **`train_<时间戳>` 根**（当前为单方案单目录）。

**测试时终端**（`tools/test.py`）：`IoUMetric` 仍会打 **per class** 的日志；摘要层输出当前方案对应的一套指标表。

| 与训练对齐 | 训练权重与日志常用位置 | 测试默认 `work_dir`（未写 `--work-dir`） |
|------------|--------------------------|----------------------------------------|
| 各方案单训（课题归档） | `data/checkpoints1/train_<时间戳>/` | `data/checkpoints2/test_<同一时间戳>/` |

---

## Git 与数据

- **不要**将 `DataA-B`、`DataC` 等数据根、大体积 `*.pth` / `*.pth.tar`、整份 `semantic_segmentation/checkpoint/` 训练产物提交进 Git；用 `.gitignore` 排除。  
- 建议单独分支维护课题改动；上游 MambaVision 更新时按需合并。

---

## 与 `README.md` 的关系

| 项目 | 主干与框架 | 备忘文档 |
|------|-------------|----------|
| TransNeXt 工程 | Mask2Former + TransNeXt-Tiny，见 `TransNeXt-main/.../segmentation/mask2former/` | 上级目录 **`README.md`**（`my_TransNext/README.md`） |
| 本 MambaVision 定制 | MambaVision-Tiny + UPerNet（MMSeg），见 **`semantic_segmentation/`** | **`MambaVision.md`**（本文，位于 `MambaVision-main/MambaVision-main/`） |

二者共用 **DataA-B**（及 **DataC**）数据约定，便于做 **Transformer vs Mamba** 对比实验；具体数值结果以各自 `checkpoint` 下导出的 **json / csv** 为准，可在实验稳定后把摘要表补进本文或上级 **`README.md`**。
