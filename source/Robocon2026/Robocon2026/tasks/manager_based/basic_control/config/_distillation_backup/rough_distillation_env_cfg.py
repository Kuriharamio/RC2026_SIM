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
from Robocon2026.robots.go2w import UNITREE_GO2W_CFG
from Robocon2026.tasks.manager_based.basic_control.basic_control_env_cfg import BasicControlEnvCfg


@configclass
class GO2WRoughDistillationEnvCfg(BasicControlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 4096

        # assets
        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # assets
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # terminations
        self.terminations.arm_contact = None


@configclass
class GO2WRoughDistillationEnvCfg_PLAY(GO2WRoughDistillationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.privilege.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # terminations
        self.terminations.arm_contact = None
