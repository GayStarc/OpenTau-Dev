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

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata


def _write_jsonlines(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _write_dataset_meta(root: Path, include_stats: bool) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_robot",
        "total_episodes": 1,
        "total_frames": 10,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 5,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2], "names": None},
            "action": {"dtype": "float32", "shape": [2], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None}
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info))
    _write_jsonlines(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": "pick object"}])
    _write_jsonlines(meta_dir / "episodes.jsonl", [{"episode_index": 0, "tasks": ["pick object"], "length": 10}])
    _write_jsonlines(
        meta_dir / "episodes_stats.jsonl",
        [
            {
                "episode_index": 0,
                "stats": {
                    "observation.state": {
                        "min": [0.0, 0.0],
                        "max": [1.0, 1.0],
                        "mean": [0.5, 0.25],
                        "std": [0.1, 0.2],
                        "count": [10],
                    },
                    "action": {
                        "min": [-1.0, -1.0],
                        "max": [1.0, 1.0],
                        "mean": [0.1, -0.1],
                        "std": [0.3, 0.4],
                        "count": [10],
                    },
                },
            }
        ],
    )

    if include_stats:
        stats = {
            "observation.state": {
                "min": [0.0, 0.0],
                "max": [2.0, 2.0],
                "mean": [0.9, 0.8],
                "std": [0.7, 0.6],
                "count": [123],
            },
            "action": {
                "min": [-2.0, -2.0],
                "max": [2.0, 2.0],
                "mean": [0.4, -0.4],
                "std": [0.9, 0.8],
                "count": [123],
            },
        }
        (meta_dir / "stats.json").write_text(json.dumps(stats))


def test_v21_metadata_prefers_cached_stats_json(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_dataset_meta(dataset_root, include_stats=True)

    with patch("opentau.datasets.lerobot_dataset.aggregate_stats") as aggregate_stats_mock:
        metadata = LeRobotDatasetMetadata(repo_id="local/test_dataset", root=dataset_root)

    aggregate_stats_mock.assert_not_called()
    np.testing.assert_allclose(metadata.stats["observation.state"]["mean"], np.array([0.9, 0.8]))
    np.testing.assert_allclose(metadata.stats["action"]["count"], np.array([123]))


def test_v21_metadata_falls_back_to_episode_stats_aggregation(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_dataset_meta(dataset_root, include_stats=False)
    sentinel_stats = {
        "action": {
            "min": np.array([-1.0, -1.0]),
            "max": np.array([1.0, 1.0]),
            "mean": np.array([0.2, 0.3]),
            "std": np.array([0.4, 0.5]),
            "count": np.array([10]),
        }
    }

    with patch("opentau.datasets.lerobot_dataset.aggregate_stats", return_value=sentinel_stats) as aggregate_stats_mock:
        metadata = LeRobotDatasetMetadata(repo_id="local/test_dataset", root=dataset_root)

    aggregate_stats_mock.assert_called_once()
    assert metadata.stats == sentinel_stats
