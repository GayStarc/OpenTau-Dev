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

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-}

export WANDB_API_KEY=${WANDB_API_KEY:-21a8bda930a645b08f2834efd21ba21e98cd83cf}
TIMESTAMP=$(date +"%m-%d-%H-%M")

BRIDGE_ROOT=${BRIDGE_ROOT:-/mnt/ckp/guchenyang/Data/Bridge/bridge_lerobot_v21}
FRACTAL_ROOT=${FRACTAL_ROOT:-/mnt/ckp/guchenyang/Data/Bridge/fractal_lerobot_v21}

EXPERT_CONFIG=${EXPERT_CONFIG:-$REPO_ROOT/configs/examples/pi05_bridge_fractal_expert_training_config.json}
FULL_CONFIG=${FULL_CONFIG:-$REPO_ROOT/configs/examples/pi05_bridge_fractal_full_training_config.json}

EXPERT_OUTPUT_DIR=${EXPERT_OUTPUT_DIR:-$REPO_ROOT/outputs/train/pi05_bridge_fractal_expert}
FULL_OUTPUT_DIR=${FULL_OUTPUT_DIR:-$REPO_ROOT/outputs/train/pi05_bridge_fractal_full}

TRAIN_SCRIPT="$REPO_ROOT/src/opentau/scripts/train.py"
PYTHONPATH_VALUE="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
    cat <<EOF
Usage: $0 <command> [train args...]

Commands:
  norm          Aggregate cached stats for both Bridge and Fractal datasets
  train-expert  Train PI05 with the expert-only config
  train-full    Train PI05 with the full-finetune config
  all-expert    Run norm, then train-expert
  all-full      Run norm, then train-full

Environment overrides:
  PYTHON_BIN, ACCELERATE_BIN, ACCELERATE_CONFIG
  BRIDGE_ROOT, FRACTAL_ROOT
  EXPERT_CONFIG, FULL_CONFIG
  EXPERT_OUTPUT_DIR, FULL_OUTPUT_DIR
EOF
}

run_norm() {
    echo "Aggregating cached stats for Bridge dataset: $BRIDGE_ROOT"
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -m opentau.scripts.aggregate_dataset_stats \
        "$BRIDGE_ROOT" \
        --require-feature observation.images.image_0 \
        --require-feature observation.state \
        --require-feature action

    echo "Aggregating cached stats for Fractal dataset: $FRACTAL_ROOT"
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -m opentau.scripts.aggregate_dataset_stats \
        "$FRACTAL_ROOT" \
        --require-feature observation.images.image \
        --require-feature observation.state \
        --require-feature action
}

train_with_config() {
    local config_path=$1
    local output_dir=$2
    shift 2

    local -a cmd=("$ACCELERATE_BIN" "launch")
    if [ -n "$ACCELERATE_CONFIG" ]; then
        cmd+=("--config_file" "$ACCELERATE_CONFIG")
    fi
    cmd+=("$TRAIN_SCRIPT" "--config_path=$config_path" "--output_dir=$output_dir")
    if [ "$#" -gt 0 ]; then
        cmd+=("$@")
    fi

    echo "Executing: ${cmd[*]}"
    PYTHONPATH="$PYTHONPATH_VALUE" "${cmd[@]}"
}

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
        train_with_config "$EXPERT_CONFIG" "$EXPERT_OUTPUT_DIR" "--wandb.name=pi05-bridge-fractal-expert--${TIMESTAMP}" "$@"
        ;;
    train-full)
        train_with_config "$FULL_CONFIG" "$FULL_OUTPUT_DIR" "--wandb.name=pi05-bridge-fractal-full--${TIMESTAMP}" "$@"
        ;;
    all-expert)
        run_norm
        train_with_config "$EXPERT_CONFIG" "$EXPERT_OUTPUT_DIR" "--wandb.name=pi05-bridge-fractal-expert--${TIMESTAMP}" "$@"
        ;;
    all-full)
        run_norm
        train_with_config "$FULL_CONFIG" "$FULL_OUTPUT_DIR" "--wandb.name=pi05-bridge-fractal-full--${TIMESTAMP}" "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: $command" >&2
        usage
        exit 1
        ;;
esac
