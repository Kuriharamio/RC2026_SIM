# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from isaaclab.utils import configclass

##
# Pre-defined configs
##
from Robocon2026.tasks.manager_based.basic_control.config.go2.rough_ppo_env_cfg import Go2RoughEnvCfg


@configclass
class Go2FlatEnvCfg(Go2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        if self.rewards.base_height_l2:
            self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.privilege.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None



class Go2FlatEnvCfg_PLAY(Go2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.randomize_external_force_torque = None
        self.events.push_robot = None
