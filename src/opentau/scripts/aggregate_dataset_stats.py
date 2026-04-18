#!/usr/bin/env python

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

"""Aggregate v2.1 episode statistics into a cached stats.json file."""

import argparse
from pathlib import Path

from opentau.datasets.compute_stats import aggregate_stats
from opentau.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_tasks,
    write_stats,
)


def _require_non_empty_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset metadata file is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Required dataset metadata file is empty: {path}")


def validate_dataset_metadata(dataset_root: Path, required_features: list[str] | None = None) -> dict:
    """Validate the metadata needed to aggregate cached statistics."""
    dataset_root = dataset_root.resolve()
    meta_dir = dataset_root / "meta"
    if not meta_dir.is_dir():
        raise NotADirectoryError(f"Dataset root does not contain a meta directory: {meta_dir}")

    for rel_path in [INFO_PATH, TASKS_PATH, EPISODES_PATH, EPISODES_STATS_PATH]:
        _require_non_empty_file(dataset_root / rel_path)

    info = load_info(dataset_root)
    tasks, _ = load_tasks(dataset_root)
    episodes = load_episodes(dataset_root)
    episodes_stats = load_episodes_stats(dataset_root)

    if not tasks:
        raise ValueError(f"Dataset metadata has no tasks in {dataset_root / TASKS_PATH}")
    if not episodes:
        raise ValueError(f"Dataset metadata has no episodes in {dataset_root / EPISODES_PATH}")
    if not episodes_stats:
        raise ValueError(f"Dataset metadata has no episode stats in {dataset_root / EPISODES_STATS_PATH}")

    required_features = required_features or []
    missing_features = sorted(set(required_features) - set(info["features"]))
    if missing_features:
        raise KeyError(
            f"Dataset is missing required features {missing_features}. "
            f"Available features: {sorted(info['features'])}"
        )

    missing_episode_stats = sorted(set(episodes) - set(episodes_stats))
    if missing_episode_stats:
        preview = missing_episode_stats[:10]
        suffix = "..." if len(missing_episode_stats) > 10 else ""
        raise ValueError(
            "Some episodes do not have statistics entries in "
            f"{dataset_root / EPISODES_STATS_PATH}: {preview}{suffix}"
        )

    return {
        "info": info,
        "tasks": tasks,
        "episodes": episodes,
        "episodes_stats": episodes_stats,
    }


def aggregate_dataset_stats(dataset_root: str | Path, required_features: list[str] | None = None) -> Path:
    """Aggregate per-episode stats and write meta/stats.json."""
    dataset_root = Path(dataset_root).resolve()
    metadata = validate_dataset_metadata(dataset_root, required_features)
    stats = aggregate_stats(list(metadata["episodes_stats"].values()))
    write_stats(stats, dataset_root)
    return dataset_root / STATS_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate v2.1 dataset stats into meta/stats.json.")
    parser.add_argument("dataset_root", type=Path, help="Root directory of the local LeRobot dataset.")
    parser.add_argument(
        "--require-feature",
        action="append",
        dest="required_features",
        default=[],
        help="Feature name that must exist in meta/info.json. May be passed multiple times.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = aggregate_dataset_stats(args.dataset_root, required_features=args.required_features)
    print(f"Wrote aggregated dataset stats to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
