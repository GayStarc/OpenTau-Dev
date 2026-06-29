#!/usr/bin/env bash
# Train OpenTau pi05_base on the local Sharpa pour-drink LeRobot v3.0 dataset.
#
# Usage:
#   ./scripts/train_pi05_sharpa_pour_drink.sh
#
# Environment overrides (all optional):
#   VENV_PYTHON          Python interpreter to use (default: $REPO_ROOT/.venv/bin/python)
#   CONFIG_PATH          Path to the training JSON config
#   ACCELERATE_CONFIG    Path to the Accelerate YAML config
#   OUTPUT_ROOT          Root directory for run outputs
#   RUN_NAME             Name of this run (default: timestamp)
#   NUM_PROCESSES        Number of GPU processes for DDP (overrides the Accelerate config value)
#   OPENTAU_PPU_MODE     Set to 1 on Alibaba Zhenwu/PPU nodes to apply NCCL socket-only settings
#
# Examples:
#   # 16-GPU training (default)
#   ./scripts/train_pi05_sharpa_pour_drink.sh
#
#   # 2-GPU local run (run from the repo root)
#   NUM_PROCESSES=2 ACCELERATE_CONFIG=configs/examples/accelerate_ddp_config_2gpu.yaml \
#     ./scripts/train_pi05_sharpa_pour_drink.sh
#
#   # Alibaba Zhenwu/PPU 2-GPU run
#   NUM_PROCESSES=2 ACCELERATE_CONFIG=configs/examples/accelerate_ddp_config_2gpu.yaml \
#     OPENTAU_PPU_MODE=1 ./scripts/train_pi05_sharpa_pour_drink.sh
#
#   # Single-GPU debug run
#   NUM_PROCESSES=1 ./scripts/train_pi05_sharpa_pour_drink.sh
#
#   # Custom output directory
#   OUTPUT_ROOT=/mnt/nas/guchenyang/Experiments/pi05_sharpa_pour_drink ./scripts/train_pi05_sharpa_pour_drink.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Fixed paths for this recipe
# ---------------------------------------------------------------------------
BASE_MODEL="/mnt/nas/guchenyang/Pretrain/OpenTau/pi05_base"
DATASET_ROOT="/mnt/nas/guchenyang/Data/Sharpa/lerobot_v30_pour_drink_0624"

# ---------------------------------------------------------------------------
# Tunable knobs via environment variables
# ---------------------------------------------------------------------------
VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/pi05_sharpa_pour_drink_local.json}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$REPO_ROOT/configs/examples/accelerate_ddp_config_16gpu.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/train/pi05_sharpa_pour_drink}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Python not found or not executable: $VENV_PYTHON" >&2
    echo "Hint: create the venv with 'uv sync' or set VENV_PYTHON." >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Training config not found: $CONFIG_PATH" >&2
    exit 1
fi

if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
    echo "Accelerate config not found: $ACCELERATE_CONFIG" >&2
    exit 1
fi

if [[ ! -d "$BASE_MODEL" ]]; then
    echo "Base model directory not found: $BASE_MODEL" >&2
    exit 1
fi

if [[ ! -f "$BASE_MODEL/config.json" ]]; then
    echo "Base model config.json not found in $BASE_MODEL" >&2
    exit 1
fi

if [[ ! -f "$BASE_MODEL/model.safetensors" ]]; then
    echo "Base model weights not found in $BASE_MODEL" >&2
    exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Dataset directory not found: $DATASET_ROOT" >&2
    exit 1
fi

if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
    echo "Dataset meta/info.json not found in $DATASET_ROOT" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME
export HF_DATASETS_CACHE
export HUGGINGFACE_HUB_CACHE

# Best-effort PPU (Alibaba Zhenwu 810E) mode. If you hit NCCL "Cuda failure
# 'invalid argument'" during the first collective, set OPENTAU_PPU_MODE=1 to
# apply the socket-only settings recommended by the PPU best-practice docs.
if [[ -n "${OPENTAU_PPU_MODE:-}" ]]; then
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_IB_HCA=
    export NCCL_IB_DISABLE=1
fi

mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

# ---------------------------------------------------------------------------
# Build accelerate launch command
# ---------------------------------------------------------------------------
LAUNCH_ARGS=(
    --config_file "$ACCELERATE_CONFIG"
)

if [[ -n "${NUM_PROCESSES:-}" ]]; then
    LAUNCH_ARGS+=(--num_processes "$NUM_PROCESSES")
fi

echo "======================================================================"
echo "Training OpenTau pi05 on Sharpa pour-drink"
echo "  Base model:   $BASE_MODEL"
echo "  Dataset:      $DATASET_ROOT"
echo "  Config:       $CONFIG_PATH"
echo "  Accelerate:   $ACCELERATE_CONFIG"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Python:       $VENV_PYTHON"
echo "======================================================================"

"$VENV_PYTHON" -m accelerate.commands.launch \
    "${LAUNCH_ARGS[@]}" \
    "$REPO_ROOT/src/opentau/scripts/train.py" \
    --config_path="$CONFIG_PATH" \
    --output_dir="$OUTPUT_DIR" \
    "$@"
