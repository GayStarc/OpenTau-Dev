# Sharpa Pour-Drink PI05 本地训练说明

本文档说明如何使用本仓库里的脚本，从本地 checkpoint
`/mnt/nas/guchenyang/Pretrain/OpenTau/pi05_base` 出发，训练本地 LeRobot v3.0 数据集
`/mnt/nas/guchenyang/Data/Sharpa/lerobot_v30_pour_drink_0624`。

## 文件

- 训练配置：`configs/examples/pi05_sharpa_pour_drink_local.json`
- 训练启动脚本：`scripts/train_pi05_sharpa_pour_drink.sh`
- 16 卡 DDP Accelerate 配置：`configs/examples/accelerate_ddp_config_16gpu.yaml`
- 推理 Python 入口：`src/opentau/scripts/inference_sharpa_pour_drink.py`
- 推理启动脚本：`scripts/inference_pi05_sharpa_pour_drink.sh`

## 数据与模型配置

当前数据集元信息显示：

- 数据集版本：LeRobot `v3.0`
- episode 数：30
- frame 数：15165
- fps：10
- `observation.state` 维度：58
- `action` 维度：58
- 训练使用 3 路相机：
  - `camera0` -> `observation.images.head_left`
  - `camera1` -> `observation.images.left_wrist`
  - `camera2` -> `observation.images.right_wrist`

因此配置中把 `num_cams` 设为 3，并把 `max_state_dim`、`max_action_dim` 都设为 58。

## 多卡训练说明

本脚本默认面向 **16 卡 DDP** 训练：

- 默认 Accelerate 配置：`configs/examples/accelerate_ddp_config_16gpu.yaml`，其中 `num_processes=16`。
- 配置里的 `dataloader_batch_size=2` 是**每个进程**的 batch size，因此全局等效 batch size 为 `2 × 16 = 32`。
- 如果需要调整全局 batch size，建议保持 `batch_size == dataloader_batch_size × gradient_accumulation_steps` 的关系，并相应调整学习率。

## 启动训练

先确认虚拟环境可用：

```bash
cd /mnt/nas/guchenyang/Code/OpenTau-Dev
source .venv/bin/activate
```

直接运行（默认使用 16 卡 DDP）：

```bash
./scripts/train_pi05_sharpa_pour_drink.sh
```

脚本默认会使用：

- `.venv/bin/python`
- `configs/examples/pi05_sharpa_pour_drink_local.json`
- `configs/examples/accelerate_ddp_config_16gpu.yaml`（`num_processes=16`）
- 输出目录：`outputs/train/pi05_sharpa_pour_drink/<时间戳>/`
- Hugging Face / datasets cache：`.cache/huggingface/`

注意：fresh run 要使用不存在的输出目录。训练程序会自己创建输出目录；不要提前
`mkdir` 这个目录，否则 `resume=false` 时会报
`Output directory ... already exists and resume is False`。

可以在命令行覆盖训练配置字段，例如先跑 20 step 做冒烟测试：

```bash
./scripts/train_pi05_sharpa_pour_drink.sh --steps=20 --save_freq=20 --val_freq=0
```

如果只想用 1 张 GPU 做调试，可以通过 `NUM_PROCESSES=1` 覆盖：

```bash
NUM_PROCESSES=1 ./scripts/train_pi05_sharpa_pour_drink.sh
```

或者指定一个单卡 Accelerate 配置：

```bash
ACCELERATE_CONFIG=/path/to/your_single_gpu_accelerate.yaml \
./scripts/train_pi05_sharpa_pour_drink.sh
```

## 常用覆盖项

脚本支持这些环境变量：

```bash
VENV_PYTHON=/path/to/python \
CONFIG_PATH=/path/to/config.json \
ACCELERATE_CONFIG=/path/to/accelerate.yaml \
NUM_PROCESSES=16 \
OUTPUT_DIR=/path/to/output \
HF_HOME=/path/to/hf_cache \
./scripts/train_pi05_sharpa_pour_drink.sh
```

也可以直接在脚本后追加 OpenTau 训练参数：

```bash
./scripts/train_pi05_sharpa_pour_drink.sh \
  --batch_size=4 \
  --dataloader_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --steps=20000 \
  --save_freq=1000
```

## 恢复训练

如果只是想重新开始训练，请换一个新的 `RUN_NAME`/`OUTPUT_DIR`，或删除之前失败的空目录。
如果要从已有 checkpoint 继续训练，才使用 `--resume=true`。

从某个 checkpoint 恢复时，使用该 checkpoint 下保存的 `train_config.json`：

```bash
./scripts/train_pi05_sharpa_pour_drink.sh \
  --config_path=outputs/train/pi05_sharpa_pour_drink/<run>/checkpoints/001000/train_config.json \
  --resume=true \
  --steps=20000
```

注意：脚本默认也会传入一个 `--output_dir`。恢复训练时如果希望完全沿用 checkpoint 配置里的输出目录，可以直接使用底层命令：

```bash
.venv/bin/python -m accelerate.commands.launch \
  --config_file configs/examples/accelerate_ddp_config.yaml \
  src/opentau/scripts/train.py \
  --config_path=outputs/train/pi05_sharpa_pour_drink/<run>/checkpoints/001000/train_config.json \
  --resume=true \
  --steps=20000
```

## 调参建议

- `batch_size` 是全局 batch size；`dataloader_batch_size * gradient_accumulation_steps` 必须等于 `batch_size`。
- 如果显存不足，优先降低 `dataloader_batch_size`，再用 `gradient_accumulation_steps` 补回全局 batch。
- 当前配置关闭 WandB：`wandb.enable=false`。需要记录到 WandB 时，把配置里的 `wandb.enable` 和 `wandb.mode` 改掉，或用命令行覆盖。
- 数据集是 10 Hz，配置里的 `dataset_mixture.action_freq` 设为 `10.0`，与数据集 fps 对齐。
- 当前环境里 `torchcodec` 加载失败时，OpenTau 会自动回退到 `pyav` 解码。`pyav` 已安装即可继续读取本数据集视频。
- 如果看到 `/root/.cache/huggingface` 只读错误，请确认是通过 `scripts/train_pi05_sharpa_pour_drink.sh` 启动，或手动设置 `HF_HOME`/`HF_DATASETS_CACHE` 到可写目录。
- 如果看到 `Could not write speed percentiles ... using in-memory values`，说明数据集目录只读；这不会阻止训练，但每次加载都会重新计算速度分位数。需要消除该警告时，把数据集目录改为可写，或复制到可写路径后更新配置里的 `dataset_mixture.datasets[0].root`。

## 推理（Inference）

训练完成后，可以使用新生成的 checkpoint 进行推理。我们提供了专用脚本：

- Python 推理入口：`src/opentau/scripts/inference_sharpa_pour_drink.py`
- Shell 包装脚本：`scripts/inference_pi05_sharpa_pour_drink.sh`

### 快速冒烟测试（dummy observation）

加载 checkpoint 并使用全零虚拟观测跑通前向：

```bash
./scripts/inference_pi05_sharpa_pour_drink.sh \
  --checkpoint outputs/train/pi05_sharpa_pour_drink/<run>/checkpoint-10000
```

### 使用真实数据集帧推理

从数据集中取第 `100` 帧作为输入，预测 action chunk，并与该帧对应的真实动作对比 MSE：

```bash
./scripts/inference_pi05_sharpa_pour_drink.sh \
  --checkpoint outputs/train/pi05_sharpa_pour_drink/<run>/checkpoint-10000 \
  --mode dataset \
  --frame-index 100 \
  --output /tmp/pi05_pour_drink_pred.json
```

### 推理脚本常用参数

| 参数 | 说明 |
|------|------|
| `--checkpoint` | 必填。checkpoint 目录，需包含 `config.json` 与 `model.safetensors`。 |
| `--mode` | `dummy`（默认）或 `dataset`。 |
| `--frame-index` | 仅 `--mode dataset` 有效，数据集全局帧索引。 |
| `--dataset-root` | 数据集根目录，默认与训练一致。 |
| `--n-runs` | 用于测速的推理次数，默认 `10`。 |
| `--warmup` | 测速前预热次数，默认 `2`。 |
| `--dtype` | `bfloat16`（默认）、`float16`、`float32`。 |
| `--compile` | 使用 `torch.compile` 编译 `sample_actions`（仅 CUDA）。 |
| `--output` | 可选，把预测结果和耗时写入 JSON。 |

也可以直接调用 Python 入口：

```bash
.venv/bin/python src/opentau/scripts/inference_sharpa_pour_drink.py \
  --checkpoint outputs/train/pi05_sharpa_pour_drink/<run>/checkpoint-10000 \
  --mode dataset \
  --frame-index 100
```

## 输出目录结构

一次典型训练运行会在 `outputs/train/pi05_sharpa_pour_drink/<run>/` 下生成：

```
<run>/
├── train_config.json          # 实际生效的训练配置
├── checkpoints/
│   ├── 001000/                # 每 save_freq 步保存的 checkpoint
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── train_config.json
│   └── ...
└── logs/                      # 训练日志（若启用）
```

checkpoint 目录同时包含 `config.json` 与 `model.safetensors`，可直接作为 `--checkpoint` 传给推理脚本，也可用于继续训练。
