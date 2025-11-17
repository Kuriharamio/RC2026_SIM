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
class GO2WRoughFintuneEnvCfg(BasicControlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 2048

        # assets
        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # terminations
        self.terminations.arm_contact = None

        # rewards
        # 奖励机器人跟踪xy平面线速度命令的表现
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        # 奖励机器人跟踪绕z轴角速度命令的表现
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        # 惩罚机器人在z轴方向的线速度（避免不必要的上下运动）
        self.rewards.lin_vel_z_l2.weight = -2.0
        # 惩罚机器人绕x、y轴的角速度（避免翻滚和俯仰）
        self.rewards.ang_vel_xy_l2.weight = -0.05
        # 惩罚机器人偏离水平姿态
        self.rewards.flat_orientation_l2.weight = -0.05
        # 惩罚关节加速度，鼓励平滑的动作
        self.rewards.dof_acc_l2.weight = -2.5e-7
        # 惩罚机器人身体部位与环境发生碰撞
        self.rewards.collision.weight = -0.0
        # 惩罚动作变化率，鼓励动作的连续性和平滑性
        self.rewards.action_rate_l2.weight = -0.01
        # 惩罚关节力矩，鼓励节能的动作
        self.rewards.dof_torques_l2.weight = -2.0e-4
        # 奖励足部离地时间，鼓励机器人抬脚行走
        self.rewards.feet_air_time.weight = 0.01
        # 惩罚hip关节位置偏差
        self.rewards.hip_pos_error.weight = -0.05
        # # 惩罚关节位置偏差
        self.rewards.dof_pos_error.weight = -0.0
        # 惩罚足部撞击垂直表面（绊倒）
        self.rewards.feet_stumble.weight = -0.25
        # 惩罚高度过低过高的行为
        self.rewards.base_height_penalty.weight = -0.1
        # 惩罚接近关节位置极限的情况
        self.rewards.dof_pos_limits.weight = -0.0
        # 惩罚足部滑行
        self.rewards.feet_slide.weight = -0.0


@configclass
class GO2WRoughFinetuneEnvCfg_PLAY(GO2WRoughFintuneEnvCfg):
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
