# Bridge / Fractal 本地 PI05 训练指南

这篇文档面向当前仓库里的本地训练方案，说明如何用
`/mnt/ckp/guchenyang/Data/Bridge/bridge_lerobot_v21`
和
`/mnt/ckp/guchenyang/Data/Bridge/fractal_lerobot_v21`
启动 PI05 训练，以及对应的代码框架是什么。

## 1. 训练入口概览

当前有两个推荐入口：

```bash
src/opentau/scripts/pi05_bridge_fractal.sh
src/opentau/scripts/pi05_bridge_fractal_8gpu.sh
```

- `pi05_bridge_fractal.sh`：基础单机入口，适合单卡、少卡或自定义 accelerate 配置
- `pi05_bridge_fractal_8gpu.sh`：单机 8 卡便捷入口，内置 DDP / DeepSpeed 两套 YAML

它们共同封装了两类动作：

- `norm`：聚合本地数据集已有的 `meta/episodes_stats.jsonl`，生成缓存 `meta/stats.json`
- `train-*`：调用 OpenTau 现有训练主程序 `src/opentau/scripts/train.py`

支持的子命令如下：

```bash
bash src/opentau/scripts/pi05_bridge_fractal.sh help
```

```text
norm
train-expert
train-full
all-expert
all-full
```

其中：

- `expert`：只训练 PI05 的 expert 部分，显存压力更低，适合先跑通
- `full`：更完整地微调模型，训练成本更高
- `all-*`：先做 `norm`，再开始训练

## 2. 训练前准备

启动训练前请确认：

- 已安装 OpenTau 依赖，并能正常运行 `python`、`accelerate`
- 已执行过：

```bash
accelerate config
```

- 可访问 `TensorAuto/pi05_base`
- 两个本地数据集目录都存在，并至少包含：
  - `meta/info.json`
  - `meta/tasks.jsonl`
  - `meta/episodes.jsonl`
  - `meta/episodes_stats.jsonl`

如果你使用 Hugging Face 拉模型，通常还需要先设置 token，例如：

```bash
export HUGGINGFACE_HUB_TOKEN=...
```

## 3. 最短启动路径

### 3.1 只做统计缓存

```bash
bash src/opentau/scripts/pi05_bridge_fractal.sh norm
```

这个步骤会分别检查：

- `bridge_lerobot_v21` 是否包含
  - `observation.images.image_0`
  - `observation.state`
  - `action`
- `fractal_lerobot_v21` 是否包含
  - `observation.images.image`
  - `observation.state`
  - `action`

成功后会在每个数据集下生成：

```text
meta/stats.json
```

### 3.2 启动 expert-only 训练

```bash
bash src/opentau/scripts/pi05_bridge_fractal.sh all-expert
```

### 3.3 启动 full-finetune 训练

```bash
bash src/opentau/scripts/pi05_bridge_fractal.sh all-full
```

### 3.4 单机八卡 DDP 训练

8 卡 DDP 入口：

```bash
bash src/opentau/scripts/pi05_bridge_fractal_8gpu.sh all-expert
```

它默认读取：

```text
configs/examples/accelerate_8gpu_ddp_config.yaml
```

默认输出目录是：

```text
outputs/train_8gpu/pi05_bridge_fractal_expert_ddp
```

如果你想把 checkpoint 保存到指定路径，例如 `/mnt/ckp/guchenyang/outputs/pi05_ckpts`：

```bash
OUTPUT_ROOT=/mnt/ckp/guchenyang/outputs/pi05_ckpts \
bash src/opentau/scripts/pi05_bridge_fractal_8gpu.sh all-expert
```

这时 expert 训练结果会落到：

```text
/mnt/ckp/guchenyang/outputs/pi05_ckpts/pi05_bridge_fractal_expert_ddp
```

### 3.5 单机八卡 DeepSpeed 训练

如果你要用 DeepSpeed：

```bash
MODE=deepspeed \
OUTPUT_ROOT=/mnt/ckp/guchenyang/outputs/pi05_ckpts \
bash src/opentau/scripts/pi05_bridge_fractal_8gpu.sh all-expert
```

它默认读取：

```text
configs/examples/accelerate_8gpu_deepspeed_config.yaml
```

对应输出目录默认是：

```text
/mnt/ckp/guchenyang/outputs/pi05_ckpts/pi05_bridge_fractal_expert_deepspeed
```

如果你不想按 `OUTPUT_ROOT` 自动拼接，也可以直接指定最终目录：

```bash
MODE=deepspeed \
EXPERT_OUTPUT_DIR=/mnt/ckp/guchenyang/outputs/pi05_ckpts/bridge_fractal_exp1 \
bash src/opentau/scripts/pi05_bridge_fractal_8gpu.sh train-expert
```

## 4. 训练配置文件

当前使用两份配置：

- `configs/examples/pi05_bridge_fractal_expert_training_config.json`
- `configs/examples/pi05_bridge_fractal_full_training_config.json`
- `configs/examples/accelerate_8gpu_ddp_config.yaml`
- `configs/examples/accelerate_8gpu_deepspeed_config.yaml`

这两份配置的共同点：

- 数据集是本地 `bridge + fractal`
- 混合采样权重固定为 `0.5 / 0.5`
- 单相机输入
- `bridge` 默认使用 `observation.images.image_0`
- `fractal` 默认使用 `observation.images.image`
- 状态归一化：`MIN_MAX`
- 动作归一化：`MEAN_STD`
- 默认关闭 W&B

差异如下：

- `expert`：
  - `train_expert_only=true`
  - `freeze_vision_encoder=true`
  - 默认 `batch_size=8`
- `full`：
  - `train_expert_only=false`
  - `freeze_vision_encoder=false`
  - 默认 `batch_size=4`

## 5. 底层等价命令

Shell 脚本只是便捷封装，本质仍然是调用 OpenTau 主训练入口。

### 5.1 `norm` 的底层命令

Bridge：

```bash
PYTHONPATH=src python -m opentau.scripts.aggregate_dataset_stats \
  /mnt/ckp/guchenyang/Data/Bridge/bridge_lerobot_v21 \
  --require-feature observation.images.image_0 \
  --require-feature observation.state \
  --require-feature action
```

Fractal：

```bash
PYTHONPATH=src python -m opentau.scripts.aggregate_dataset_stats \
  /mnt/ckp/guchenyang/Data/Bridge/fractal_lerobot_v21 \
  --require-feature observation.images.image \
  --require-feature observation.state \
  --require-feature action
```

### 5.2 `train-expert` 的底层命令

```bash
PYTHONPATH=src accelerate launch \
  src/opentau/scripts/train.py \
  --config_path=configs/examples/pi05_bridge_fractal_expert_training_config.json \
  --output_dir=outputs/train/pi05_bridge_fractal_expert
```

### 5.3 `train-full` 的底层命令

```bash
PYTHONPATH=src accelerate launch \
  src/opentau/scripts/train.py \
  --config_path=configs/examples/pi05_bridge_fractal_full_training_config.json \
  --output_dir=outputs/train/pi05_bridge_fractal_full
```

如果你已经有自己的 accelerate 配置文件，可以通过环境变量覆盖：

```bash
ACCELERATE_CONFIG=configs/examples/accelerate_deepspeed_config.yaml \
bash src/opentau/scripts/pi05_bridge_fractal.sh all-expert
```

单机 8 卡包装脚本本质上也是同样的逻辑，只是帮你提前选好了 8 卡 YAML，并自动设置输出目录：

```bash
MODE=ddp|deepspeed \
OUTPUT_ROOT=/path/to/ckpts \
bash src/opentau/scripts/pi05_bridge_fractal_8gpu.sh all-expert
```

## 6. 如何恢复训练

OpenTau 的恢复训练仍然走底层训练入口。典型命令如下：

```bash
PYTHONPATH=src accelerate launch \
  src/opentau/scripts/train.py \
  --config_path=outputs/train/pi05_bridge_fractal_expert/checkpoints/001000/train_config.json \
  --output_dir=outputs/train/pi05_bridge_fractal_expert \
  --resume=true \
  --steps=200000
```

说明：

- `config_path` 指向 checkpoint 目录下保存的 `train_config.json`
- `output_dir` 仍然指向原训练目录
- `steps` 是新的总训练步数，不是“再训多少步”

## 7. 代码框架是什么

这一套训练流程没有单独重写训练框架，而是复用 OpenTau 现有训练主链路。

调用链如下：

```text
pi05_bridge_fractal.sh
├─ norm -> aggregate_dataset_stats.py
│        -> 读取 meta/info.json / episodes_stats.jsonl
│        -> 聚合得到 meta/stats.json
└─ train-* -> accelerate launch src/opentau/scripts/train.py
           -> TrainPipelineConfig
           -> make_dataset_mixture(...)
           -> make_policy(...)
           -> make_optimizer_and_scheduler(...)
           -> Accelerate 训练循环
           -> checkpoint / validation / logging
```

更具体地说：

### 7.1 配置层

`TrainPipelineConfig` 是训练总配置，负责聚合：

- `dataset_mixture`
- `policy`
- `optimizer`
- `scheduler`
- `wandb`
- `eval / val / checkpoint`

在这次本地方案里，数据集定义主要来自：

- `dataset_mixture.datasets[*].root`
- `dataset_mixture.datasets[*].action_freq`
- `dataset_mixture.datasets[*].data_features_name_mapping`

### 7.2 数据层

训练时会进入：

```text
make_dataset_mixture
-> make_dataset
-> LeRobotDataset / WeightedDatasetMixture
```

这里主要做几件事：

- 读取本地 LeRobot v2.1 数据集
- 用 `data_features_name_mapping` 把原始字段映射到标准字段
  - `camera0`
  - `state`
  - `actions`
  - `prompt`
  - `response`
- 用每个数据集自己的 `action_freq` 计算 delta timestamps
- 把两个数据集混成统一的采样流

### 7.3 归一化层

归一化不是单独在训练循环里手写，而是通过数据集统计量和 policy 内部 `Normalize` 模块完成。

来源是：

- `meta/stats.json`（如果存在则优先读）
- 否则回退聚合 `meta/episodes_stats.jsonl`

配置上由 `policy.normalization_mapping` 控制：

- `VISUAL -> IDENTITY`
- `STATE -> MIN_MAX`
- `ACTION -> MEAN_STD`

### 7.4 模型层

模型创建走：

```text
make_policy(cfg=cfg.policy, ds_meta=train_dataset.meta)
```

PI05 会根据：

- 相机数
- 状态维度
- 动作维度
- chunk size
- 数据集统计量

来初始化输入输出结构和归一化模块。

### 7.5 训练循环

训练主循环在 `src/opentau/scripts/train.py` 中，核心职责是：

- 创建 `Accelerator`
- 准备 dataloader / model / optimizer / scheduler
- 前向计算 loss
- 反向传播与梯度裁剪
- 记录日志
- 定期验证
- 定期保存 checkpoint

## 8. 输出目录结构

默认输出目录：

- expert：`outputs/train/pi05_bridge_fractal_expert`
- full：`outputs/train/pi05_bridge_fractal_full`
- 8 卡 DDP expert：`outputs/train_8gpu/pi05_bridge_fractal_expert_ddp`
- 8 卡 DDP full：`outputs/train_8gpu/pi05_bridge_fractal_full_ddp`
- 8 卡 DeepSpeed expert：`outputs/train_8gpu/pi05_bridge_fractal_expert_deepspeed`
- 8 卡 DeepSpeed full：`outputs/train_8gpu/pi05_bridge_fractal_full_deepspeed`

训练后通常会看到：

```text
outputs/train/pi05_bridge_fractal_expert/
├─ checkpoints/
│  ├─ 001000/
│  │  ├─ train_config.json
│  │  ├─ model.safetensors 或 DeepSpeed 分片
│  │  └─ ...
│  └─ ...
├─ train_config.json
└─ ...
```

如果使用 DeepSpeed 且 checkpoint 还是分片形式，可以继续用已有脚本合并：

```bash
bash src/opentau/scripts/convert_checkpoint.sh \
  outputs/train/pi05_bridge_fractal_expert/checkpoints/001000
```

## 9. 常用覆盖方式

Shell 包装脚本支持通过环境变量覆盖默认路径。

例如把输出目录改到别处：

```bash
EXPERT_OUTPUT_DIR=/data/outputs/pi05_bridge_fractal_expert \
bash src/opentau/scripts/pi05_bridge_fractal.sh train-expert
```

例如单机 8 卡时统一指定 ckpt 根目录：

```bash
OUTPUT_ROOT=/data/outputs/pi05_8gpu \
bash src/opentau/scripts/pi05_bridge_fractal_8gpu.sh all-expert
```

例如单机 8 卡时切到 DeepSpeed：

```bash
MODE=deepspeed \
OUTPUT_ROOT=/data/outputs/pi05_8gpu \
bash src/opentau/scripts/pi05_bridge_fractal_8gpu.sh all-full
```

例如指定 accelerate 配置：

```bash
ACCELERATE_CONFIG=configs/examples/accelerate_ddp_config.yaml \
bash src/opentau/scripts/pi05_bridge_fractal.sh train-full
```

例如替换数据集目录：

```bash
BRIDGE_ROOT=/path/to/bridge \
FRACTAL_ROOT=/path/to/fractal \
bash src/opentau/scripts/pi05_bridge_fractal.sh norm
```

## 10. 常见问题

### 10.1 `norm` 时报缺少 metadata 文件

先检查数据集目录下是否存在：

- `meta/info.json`
- `meta/tasks.jsonl`
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl`

如果这些文件不完整，训练前需要先把数据集转换/补齐。

### 10.2 拉取 `TensorAuto/pi05_base` 失败

常见原因：

- 没有设置 Hugging Face token
- 网络无法访问 Hugging Face
- 当前环境没有模型权限

优先检查：

```bash
echo $HUGGINGFACE_HUB_TOKEN
```

### 10.3 显存不够

优先调整：

- `batch_size`
- `dataloader_batch_size`
- `gradient_accumulation_steps`
- expert-only / full-finetune 选择

如果只是先验证链路，建议先跑：

```bash
bash src/opentau/scripts/pi05_bridge_fractal.sh train-expert --steps=100
```

### 10.4 想直接改训练超参

直接复制并修改这两份配置之一：

- `configs/examples/pi05_bridge_fractal_expert_training_config.json`
- `configs/examples/pi05_bridge_fractal_full_training_config.json`

然后用底层命令启动，或者把 `EXPERT_CONFIG` / `FULL_CONFIG` 指到新文件。

## 11. 推荐启动顺序

如果你是第一次在这两套本地数据上训练，推荐顺序如下：

```bash
# 1) 先聚合缓存统计量
bash src/opentau/scripts/pi05_bridge_fractal.sh norm

# 2) 先跑一版 expert-only
bash src/opentau/scripts/pi05_bridge_fractal.sh train-expert --steps=1000

# 3) 稳定后再跑正式训练
bash src/opentau/scripts/pi05_bridge_fractal.sh all-expert

# 4) 如果需要更完整微调，再切 full
bash src/opentau/scripts/pi05_bridge_fractal.sh all-full
```

如果你已经确认是单机八卡环境，可以直接改成：

```bash
OUTPUT_ROOT=/mnt/ckp/guchenyang/outputs/pi05_ckpts \
bash src/opentau/scripts/pi05_bridge_fractal_8gpu.sh all-expert
```
