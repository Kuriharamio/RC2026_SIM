# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from Robocon2026.robots.go2w import UNITREE_GO2W_CFG
from Robocon2026.tasks.manager_based.basic_control.basic_control_env_cfg import BasicControlEnvCfg
from Robocon2026.tasks.manager_based.basic_control.mdp import mdp


@configclass
class GO2WRoughEnvCfg(BasicControlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 4096

        self.joint_names = [
            ".*L_hip_joint",
            ".*R_hip_joint",
            "F[L,R]_thigh_joint",
            "R[L,R]_thigh_joint",
            ".*_calf_joint",
        ]

        # * assets
        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # * observations
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.privilege.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None

        # * actions
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.wheel_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.wheel_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names
        self.actions.wheel_vel.joint_names = ['.*foot_joint']
        self.actions.arm_pos = None
        self.actions.gripper_action = None


        # * events
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_base_mass.params["asset_cfg"].body_names = ["base"]
        self.events.randomize_other_mass.params["asset_cfg"].body_names = [f"^(?!.*base).*"]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = ["base"]
        self.events.randomize_external_force_torque.params["asset_cfg"].body_names = ["base"]

        # * rewards
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = ["base"]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = ["base"]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = ".*_foot_joint"
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = ".*_foot_joint"
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*(.*_foot)).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [".*_foot"]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [".*_foot"]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [".*_foot"]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [".*_foot"]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 1.0

        self.disable_zero_weight_rewards()

        # * terminations
        self.terminations.illegal_contact = None

        # * curriculums
        self.curriculum.command_levels = None


@configclass
class GO2WRoughEnvCfg_PLAY(GO2WRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        if self.scene.terrain is not None:
            self.scene.terrain.max_init_terrain_level = None
            # reduce the number of terrains to save memory
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.num_rows = 5
                self.scene.terrain.terrain_generator.num_cols = 5
                self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.privilege.enable_corruption = False
        # remove random pushing event
        self.events.randomize_external_force_torque = None
        self.events.push_robot = None
