# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from turtle import distance
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def weapon_is_assembled(
    env: ManagerBasedRLEnv,
    threshold: float = 0.01,
    frame_cfg_1: SceneEntityCfg = SceneEntityCfg("frame1"),
    frame_cfg_2: SceneEntityCfg = SceneEntityCfg("frame2"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    frame_1: FrameTransformer = env.scene[frame_cfg_1.name]
    frame_2: FrameTransformer = env.scene[frame_cfg_2.name]

    distance = torch.norm(
        frame_1.data.target_pos_w[..., 0, :] - frame_2.data.target_pos_w[..., 0, :],
        dim=1,
    )

    return distance < threshold
