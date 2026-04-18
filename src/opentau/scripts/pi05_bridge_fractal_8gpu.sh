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
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
BASE_SCRIPT="$SCRIPT_DIR/pi05_bridge_fractal.sh"

MODE=${MODE:-ddp}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/train_8gpu}
ACCELERATE_CONFIG_DEFAULT=""

case "$MODE" in
    ddp)
        ACCELERATE_CONFIG_DEFAULT="$REPO_ROOT/configs/examples/accelerate_8gpu_ddp_config.yaml"
        ;;
    deepspeed)
        ACCELERATE_CONFIG_DEFAULT="$REPO_ROOT/configs/examples/accelerate_8gpu_deepspeed_config.yaml"
        ;;
    *)
        echo "Unsupported MODE=$MODE. Expected 'ddp' or 'deepspeed'." >&2
        exit 1
        ;;
esac

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-$ACCELERATE_CONFIG_DEFAULT}
EXPERT_OUTPUT_DIR=${EXPERT_OUTPUT_DIR:-$OUTPUT_ROOT/pi05_bridge_fractal_expert_${MODE}}
FULL_OUTPUT_DIR=${FULL_OUTPUT_DIR:-$OUTPUT_ROOT/pi05_bridge_fractal_full_${MODE}}

usage() {
    cat <<EOF
Usage: $0 <command> [train args...]

Commands:
  norm          Aggregate cached stats for both Bridge and Fractal datasets
  train-expert  Launch 8-GPU expert-only training
  train-full    Launch 8-GPU full-finetune training
  all-expert    Run norm, then launch 8-GPU expert-only training
  all-full      Run norm, then launch 8-GPU full-finetune training

Environment overrides:
  MODE=ddp|deepspeed
  OUTPUT_ROOT=/path/to/output_root
  ACCELERATE_CONFIG=/path/to/accelerate_config.yaml
  EXPERT_OUTPUT_DIR=/path/to/expert_output_dir
  FULL_OUTPUT_DIR=/path/to/full_output_dir
  BRIDGE_ROOT=/path/to/bridge_dataset
  FRACTAL_ROOT=/path/to/fractal_dataset

Default accelerate config:
  $ACCELERATE_CONFIG_DEFAULT
EOF
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

if [ ! -f "$BASE_SCRIPT" ]; then
    echo "Base training script not found: $BASE_SCRIPT" >&2
    exit 1
fi

command=$1
shift

case "$command" in
    norm|train-expert|train-full|all-expert|all-full)
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

echo "Using MODE=$MODE"
echo "Using ACCELERATE_CONFIG=$ACCELERATE_CONFIG"
echo "Using EXPERT_OUTPUT_DIR=$EXPERT_OUTPUT_DIR"
echo "Using FULL_OUTPUT_DIR=$FULL_OUTPUT_DIR"

ACCELERATE_CONFIG="$ACCELERATE_CONFIG" \
EXPERT_OUTPUT_DIR="$EXPERT_OUTPUT_DIR" \
FULL_OUTPUT_DIR="$FULL_OUTPUT_DIR" \
bash "$BASE_SCRIPT" "$command" "$@"
