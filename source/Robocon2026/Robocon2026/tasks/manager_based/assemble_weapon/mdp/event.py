# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_shape,
)

def connect_objects(
    env: ManagerBasedRLEnv,
    object1_cfg: SceneEntityCfg = SceneEntityCfg("object_1"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object_2"),
    threshold: float = 0.05,
) -> torch.Tensor:
    """
    if the distance between two objects is less than a threshold,
    then create a fixed joint between them.
    """
    object1: RigidObject = env.scene[object1_cfg.name]
    object2: RigidObject = env.scene[object2_cfg.name]

    pos1 = object1.data.root_pos_w[:, :3]
    pos2 = object2.data.root_pos_w[:, :3]

    dist = torch.norm(pos1 - pos2, dim=-1)

    should_connect = dist < threshold

    for env_idx in range(env.num_envs):
        if should_connect[env_idx]:
            #TODO: create a fixed joint between object1 and object2
            pass

    
    
