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

from pathlib import Path

import draccus
import pytest

from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING

CONFIG_DIR = Path("configs/examples")


@pytest.mark.parametrize(
    "config_name, train_expert_only, freeze_vision_encoder, batch_size",
    [
        ("pi05_bridge_fractal_expert_training_config.json", True, True, 8),
        ("pi05_bridge_fractal_full_training_config.json", False, False, 4),
    ],
)
def test_bridge_fractal_training_config_parses(
    config_name: str, train_expert_only: bool, freeze_vision_encoder: bool, batch_size: int
):
    original_mapping = DATA_FEATURES_NAME_MAPPING.copy()
    try:
        cfg = draccus.parse(TrainPipelineConfig, config_path=CONFIG_DIR / config_name, args=[])
    finally:
        DATA_FEATURES_NAME_MAPPING.clear()
        DATA_FEATURES_NAME_MAPPING.update(original_mapping)

    assert isinstance(cfg, TrainPipelineConfig)
    assert len(cfg.dataset_mixture.datasets) == 2
    assert cfg.dataset_mixture.weights == [0.5, 0.5]
    assert cfg.dataset_mixture.datasets[0].action_freq == 5.0
    assert cfg.dataset_mixture.datasets[1].action_freq == 3.0
    assert cfg.num_cams == 1
    assert cfg.batch_size == batch_size
    assert cfg.policy is not None
    assert cfg.policy.type == "pi05"
    assert cfg.policy.train_expert_only is train_expert_only
    assert cfg.policy.freeze_vision_encoder is freeze_vision_encoder
