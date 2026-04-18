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

import numpy as np
import pytest

from opentau.datasets.compute_stats import aggregate_stats
from opentau.datasets.utils import (
    STATS_PATH,
    TASKS_PATH,
    load_episodes_stats,
    load_stats,
)
from opentau.scripts.aggregate_dataset_stats import aggregate_dataset_stats


def _write_jsonlines(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _make_dataset_root(root: Path) -> Path:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_robot",
        "total_episodes": 2,
        "total_frames": 20,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 5,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "observation.images.image_0": {"dtype": "image", "shape": [3, 8, 8], "names": ["c", "h", "w"]},
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

    _write_jsonlines(
        meta_dir / "tasks.jsonl",
        [{"task_index": 0, "task": "pick object"}],
    )
    _write_jsonlines(
        meta_dir / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["pick object"], "length": 10},
            {"episode_index": 1, "tasks": ["pick object"], "length": 10},
        ],
    )
    _write_jsonlines(
        meta_dir / "episodes_stats.jsonl",
        [
            {
                "episode_index": 0,
                "stats": {
                    "observation.images.image_0": {
                        "min": [[[0.0]], [[0.0]], [[0.0]]],
                        "max": [[[1.0]], [[1.0]], [[1.0]]],
                        "mean": [[[0.4]], [[0.4]], [[0.4]]],
                        "std": [[[0.1]], [[0.1]], [[0.1]]],
                        "count": [10],
                    },
                    "observation.state": {
                        "min": [0.0, 0.0],
                        "max": [1.0, 1.0],
                        "mean": [0.2, 0.4],
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
            },
            {
                "episode_index": 1,
                "stats": {
                    "observation.images.image_0": {
                        "min": [[[0.0]], [[0.0]], [[0.0]]],
                        "max": [[[1.0]], [[1.0]], [[1.0]]],
                        "mean": [[[0.6]], [[0.6]], [[0.6]]],
                        "std": [[[0.2]], [[0.2]], [[0.2]]],
                        "count": [10],
                    },
                    "observation.state": {
                        "min": [0.0, 0.0],
                        "max": [1.0, 1.0],
                        "mean": [0.6, 0.8],
                        "std": [0.2, 0.3],
                        "count": [10],
                    },
                    "action": {
                        "min": [-1.0, -1.0],
                        "max": [1.0, 1.0],
                        "mean": [-0.2, 0.2],
                        "std": [0.5, 0.6],
                        "count": [10],
                    },
                },
            },
        ],
    )
    return root


def test_aggregate_dataset_stats_writes_stats_json(tmp_path):
    dataset_root = _make_dataset_root(tmp_path / "dataset")

    output_path = aggregate_dataset_stats(
        dataset_root,
        required_features=["observation.images.image_0", "observation.state", "action"],
    )

    assert output_path == dataset_root / STATS_PATH
    assert output_path.exists()

    expected = aggregate_stats(list(load_episodes_stats(dataset_root).values()))
    actual = load_stats(dataset_root)
    assert actual is not None
    assert set(actual) == set(expected)
    for key in expected:
        for stat_name in expected[key]:
            np.testing.assert_allclose(actual[key][stat_name], expected[key][stat_name])


def test_aggregate_dataset_stats_requires_expected_features(tmp_path):
    dataset_root = _make_dataset_root(tmp_path / "dataset")

    with pytest.raises(KeyError, match="missing required features"):
        aggregate_dataset_stats(dataset_root, required_features=["observation.images.image_9"])


def test_aggregate_dataset_stats_rejects_empty_tasks_file(tmp_path):
    dataset_root = _make_dataset_root(tmp_path / "dataset")
    (dataset_root / TASKS_PATH).write_text("")

    with pytest.raises(ValueError, match="Required dataset metadata file is empty"):
        aggregate_dataset_stats(dataset_root)
