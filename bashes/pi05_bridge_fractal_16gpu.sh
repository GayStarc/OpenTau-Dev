#!/bin/bash

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pi05_bridge_fractal 两机十六卡训练脚本
#   - 2 节点 × 8 GPU = 16 卡
#   - per-GPU batch size = 16, total batch size = 256
#
# DLC 任务会自动设置 MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE / NPROC_PER_NODE
#
# 用法（DLC 自动启动，或手动）:
#   MASTER_ADDR=<node0_ip> RANK=0 WORLD_SIZE=2 bash bashes/pi05_bridge_fractal_16gpu.sh train-expert
#   MASTER_ADDR=<node0_ip> RANK=1 WORLD_SIZE=2 bash bashes/pi05_bridge_fractal_16gpu.sh train-expert

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# ---------- NCCL ----------
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1000}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-""}
unset https_proxy http_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# ---------- wandb ----------
export WANDB_API_KEY=${WANDB_API_KEY:-21a8bda930a645b08f2834efd21ba21e98cd83cf}
TIMESTAMP=$(date +"%m-%d-%H-%M")

# ---------- multi-node (DLC auto-injected) ----------
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${WORLD_SIZE:-2}
NODE_RANK=${RANK:-0}
NUM_PROCESSES=$((NNODES * NPROC_PER_NODE))

# ---------- paths ----------
MODE=${MODE:-ddp}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/train_16gpu}

BRIDGE_ROOT=${BRIDGE_ROOT:-/mnt/ckp/guchenyang/Data/Bridge/bridge_lerobot_v21}
FRACTAL_ROOT=${FRACTAL_ROOT:-/mnt/ckp/guchenyang/Data/Bridge/fractal_lerobot_v21}

EXPERT_CONFIG=$REPO_ROOT/configs/examples/pi05_bridge_fractal_expert_training_config.json
FULL_CONFIG=$REPO_ROOT/configs/examples/pi05_bridge_fractal_full_training_config.json

EXPERT_OUTPUT_DIR=${EXPERT_OUTPUT_DIR:-$OUTPUT_ROOT/pi05_bridge_fractal_expert_${MODE}}
FULL_OUTPUT_DIR=${FULL_OUTPUT_DIR:-$OUTPUT_ROOT/pi05_bridge_fractal_full_${MODE}}

TRAIN_SCRIPT="$REPO_ROOT/src/opentau/scripts/train.py"
PYTHON_BIN=${PYTHON_BIN:-python}
ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ---------- accelerate config (deepspeed only) ----------
ACCELERATE_CONFIG_ARG=""
if [ "$MODE" = "deepspeed" ]; then
    ACCELERATE_CONFIG_ARG="--config_file $REPO_ROOT/configs/examples/accelerate_16gpu_deepspeed_config.yaml"
fi

# ---------- helpers ----------
usage() {
    cat <<EOF
Usage: $0 <command> [train args...]

Commands:
  norm          Aggregate cached stats for both Bridge and Fractal datasets
  train-expert  Launch 2-node 16-GPU expert-only training
  train-full    Launch 2-node 16-GPU full-finetune training
  all-expert    Run norm, then train-expert
  all-full      Run norm, then train-full

Environment overrides (DLC auto-injected):
  MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, NPROC_PER_NODE

Other overrides:
  MODE=ddp|deepspeed
  OUTPUT_ROOT, EXPERT_OUTPUT_DIR, FULL_OUTPUT_DIR
  BRIDGE_ROOT, FRACTAL_ROOT
EOF
}

run_norm() {
    echo "Aggregating cached stats for Bridge dataset: $BRIDGE_ROOT"
    "$PYTHON_BIN" -m opentau.scripts.aggregate_dataset_stats \
        "$BRIDGE_ROOT" \
        --require-feature observation.images.image_0 \
        --require-feature observation.state \
        --require-feature action

    echo "Aggregating cached stats for Fractal dataset: $FRACTAL_ROOT"
    "$PYTHON_BIN" -m opentau.scripts.aggregate_dataset_stats \
        "$FRACTAL_ROOT" \
        --require-feature observation.images.image \
        --require-feature observation.state \
        --require-feature action
}

run_train() {
    local config_path=$1
    local output_dir=$2
    local wandb_name=$3
    shift 3

    mkdir -p "$output_dir"
    cp "$0" "$output_dir/"

    echo "=============================================="
    echo "  pi05_bridge_fractal 两机十六卡训练"
    echo "=============================================="
    echo "  MODE:          $MODE"
    echo "  NNODES:        $NNODES"
    echo "  NODE_RANK:     $NODE_RANK"
    echo "  NPROC_PER_NODE:$NPROC_PER_NODE"
    echo "  NUM_PROCESSES: $NUM_PROCESSES"
    echo "  MASTER_ADDR:   $MASTER_ADDR"
    echo "  MASTER_PORT:   $MASTER_PORT"
    echo "  CONFIG:        $config_path"
    echo "  OUTPUT_DIR:    $output_dir"
    echo "  WANDB_NAME:    $wandb_name"
    echo "=============================================="

    $ACCELERATE_BIN launch \
        $ACCELERATE_CONFIG_ARG \
        --main_process_ip="$MASTER_ADDR" \
        --main_process_port="$MASTER_PORT" \
        --machine_rank="$NODE_RANK" \
        --num_machines="$NNODES" \
        --num_processes="$NUM_PROCESSES" \
        "$TRAIN_SCRIPT" \
        --config_path="$config_path" \
        --output_dir="$output_dir" \
        --wandb.name="$wandb_name" \
        "$@"
}

# ---------- main ----------
if [ $# -lt 1 ]; then
    usage
    exit 1
fi

command=$1
shift

case "$command" in
    norm)
        run_norm
        ;;
    train-expert)
        run_train "$EXPERT_CONFIG" "$EXPERT_OUTPUT_DIR" "pi05-bridge-fractal-expert--${TIMESTAMP}" "$@"
        ;;
    train-full)
        run_train "$FULL_CONFIG" "$FULL_OUTPUT_DIR" "pi05-bridge-fractal-full--${TIMESTAMP}" "$@"
        ;;
    all-expert)
        run_norm
        run_train "$EXPERT_CONFIG" "$EXPERT_OUTPUT_DIR" "pi05-bridge-fractal-expert--${TIMESTAMP}" "$@"
        ;;
    all-full)
        run_norm
        run_train "$FULL_CONFIG" "$FULL_OUTPUT_DIR" "pi05-bridge-fractal-full--${TIMESTAMP}" "$@"
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown command: $command" >&2
        usage
        exit 1
        ;;
esac
