#!/usr/bin/env bash
# Run inference with a fine-tuned OpenTau pi05 policy on the Sharpa pour-drink task.
#
# Usage:
#   ./scripts/inference_pi05_sharpa_pour_drink.sh --checkpoint <checkpoint_dir> [options]
#
# Required environment / CLI overrides:
#   --checkpoint         Path to a trained checkpoint directory
#
# Optional environment overrides:
#   VENV_PYTHON          Python interpreter (default: $REPO_ROOT/.venv/bin/python)
#   DATASET_ROOT         Root of the LeRobot v3.0 dataset
#
# Examples:
#   # Dummy observation smoke test
#   ./scripts/inference_pi05_sharpa_pour_drink.sh \
#       --checkpoint outputs/train/pi05_sharpa_pour_drink/20260101_120000/checkpoint-10000
#
#   # Real dataset frame
#   ./scripts/inference_pi05_sharpa_pour_drink.sh \
#       --checkpoint outputs/train/pi05_sharpa_pour_drink/20260101_120000/checkpoint-10000 \
#       --mode dataset \
#       --frame-index 100 \
#       --output /tmp/pi05_pour_drink_pred.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/nas/guchenyang/Data/Sharpa/lerobot_v30_pour_drink_0624}"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Python not found or not executable: $VENV_PYTHON" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"$VENV_PYTHON" "$REPO_ROOT/src/opentau/scripts/inference_sharpa_pour_drink.py" \
    --dataset-root "$DATASET_ROOT" \
    "$@"
