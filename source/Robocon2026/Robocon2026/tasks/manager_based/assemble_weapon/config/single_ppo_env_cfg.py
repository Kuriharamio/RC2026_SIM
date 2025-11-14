# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

##
# Pre-defined configs
##
from Robocon2026.tasks.manager_based.assemble_weapon.config.dual_ppo_env_cfg import AssembleWeaponDualEnvCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from Robocon2026.tasks.manager_based.assemble_weapon import mdp


@configclass
class AssembleWeaponSingleEnvCfg(AssembleWeaponDualEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # * scene
        self.scene.num_envs = 2048
        self.scene.robot2 = None
        self.scene.ee_frame_2 = None
        self.scene.gripper_contact_forces_2 = None
        self.scene.jaw_contact_forces_2 = None
        self.scene.other_contact_forces_2 = None

        self.scene.pole = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pole",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, -0.1, 0.33],
                rot=[0.707, 0, 0.707, 0],
            ),
            spawn=UsdFileCfg(
                usd_path=f"assets/Object/pole_combine.usd",
                scale=(0.7, 0.7, 0.7),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=True,
                ),
            ),
        )

        # * actions
        self.actions.gripper_action_2 = None
        self.actions.joint_pos_2 = None

        # * rewards
        self.rewards.lifting_pole = None
        self.rewards.grab_pole = None
        self.rewards.align_pole = None
        self.rewards.reaching_pole = None
        self.rewards.joint_vel_2 = None

        # * observations
        self.observations.policy.joint_pos_2 = None
        self.observations.policy.joint_effort_2 = None
        self.observations.policy.joint_vel_2 = None
        self.observations.privilege.pole_angles.params["robot_cfg"] = SceneEntityCfg("robot1")
        self.observations.privilege.pole_position.params["robot_cfg"] = SceneEntityCfg("robot1")

        # * curriculum
        self.curriculum.joint_vel_2 = None
        self.curriculum.grab_pole = None

        # * events
        self.events.add_ee_mass_2 = None
        self.events.physics_material_2 = None
        self.events.reset_pole_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.10, 0.10), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg(
                    "pole", body_names="pole_combine"
                ),
            },
        )
        self.events.reset_spear_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("spear", body_names="spear_combine"),
            },
        )

        # * debug
        debug = True
        self.scene.ee_frame_1.debug_vis = debug
        self.scene.spear_connect_frame.debug_vis = debug
        self.scene.pole_connect_frame.debug_vis = debug

        #* terminations
        self.terminations.pole_dropping = None


@configclass
class AssembleWeaponSingleEnvCfg_PLAY(AssembleWeaponSingleEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
