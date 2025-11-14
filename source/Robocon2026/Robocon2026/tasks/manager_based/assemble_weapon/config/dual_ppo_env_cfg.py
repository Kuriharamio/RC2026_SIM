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
from Robocon2026.robots.so101 import SO101_FOLLOWER_CFG
from Robocon2026.tasks.manager_based.assemble_weapon.assemble_weapon_env_cfg import AssembleWeaponBaseEnvCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from Robocon2026.tasks.manager_based.assemble_weapon import mdp
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class AssembleWeaponDualEnvCfg(AssembleWeaponBaseEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # * scene
        self.scene.num_envs = 1024
        self.scene.robot1 = SO101_FOLLOWER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot1",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=[0.3, 0.0, 0.125],
                rot=[1, 0, 0, 0],
                joint_pos={
                    "shoulder_pan": 0.0,
                    "shoulder_lift": 0.0,
                    "elbow_flex": 0.0,
                    "wrist_flex": 0.0,
                    "wrist_roll": 1.57,
                    "gripper": 0.3,
                },
                joint_vel={".*": 0.0},
            ),
        )
        self.scene.robot2 = SO101_FOLLOWER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot2",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=[-0.3, 0.0, 0.125],
                rot=[1, 0, 0, 0],
                joint_pos={
                    "shoulder_pan": 0.0,
                    "shoulder_lift": 0.0,
                    "elbow_flex": 0.0,
                    "wrist_flex": 0.0,
                    "wrist_roll": 1.57,
                    "gripper": 0.3,
                },
                joint_vel={".*": 0.0},
            ),
        )

        self.spear_height = 0.3 - 0.15
        self.pole_height = 0.4

        self.scene.spear = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Spear",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.3, -0.35, self.spear_height],
                rot=[0.707, 0, 0, 0.707],
            ),
            spawn=UsdFileCfg(
                usd_path=f"assets/Object/spear_combine.usd",
                scale=(0.7, 0.7, 0.7),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        self.scene.pole = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pole",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[-0.26582, -0.405, self.pole_height],
                rot=[0.9848, 0.17365, 0, 0],
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
                    disable_gravity=False,
                ),
            ),
        )
        # # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame_1 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot1/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot1/gripper",
                    name="end_effector_1",
                    offset=OffsetCfg(
                        pos=[0.005, 0.0, -0.095], rot=[0.707, 0.707, 0, 0]
                    ),
                ),
            ],
        )
        self.scene.ee_frame_2 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot2/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot2/gripper",
                    name="end_effector_2",
                    offset=OffsetCfg(
                        pos=[0.005, 0.0, -0.095], rot=[0.707, 0.707, 0, 0]
                    ),
                ),
            ],
        )
        self.scene.spear_connect_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Spear/spear_combine",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Spear/spear_combine",
                    name="spear_connect",
                    offset=OffsetCfg(pos=[0.0, 0.0, -0.03], rot=[1, 0, 0, 0]),
                ),
            ],
        )
        self.scene.pole_connect_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Pole/pole_combine",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Pole/pole_combine",
                    name="spear_connect",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.09], rot=[1, 0, 0, 0]),
                ),
            ],
        )

        self.scene.gripper_contact_forces_1.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Spear/spear_combine"]
        self.scene.jaw_contact_forces_1.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Spear/spear_combine"]
        self.scene.gripper_contact_forces_2.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Pole/pole_combine"]
        self.scene.jaw_contact_forces_2.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Pole/pole_combine"]

        # * rewards
        error_height = 0.025
        self.rewards.lifting_spear.params["minimal_height"] = self.spear_height + error_height
        self.rewards.lifting_pole.params["minimal_height"] = self.pole_height + error_height
        self.rewards.assemble_dist.params["minimal_height"] = self.spear_height + error_height
        self.rewards.assemble_dist_fine_grained.params["minimal_height"] = self.spear_height + error_height
        self.rewards.assemble_angle.params["minimal_height"] = self.spear_height + error_height
        self.rewards.assemble_angle_fine_grained.params["minimal_height"] = self.spear_height + error_height
        self.rewards.grab_spear.weight = 10.0
        self.rewards.lifting_spear.weight = 10.0
        self.rewards.grab_pole.weight = 10.0
        self.rewards.lifting_pole.weight = 10.0

        # * observations
        # self.observations.jaw_camera = None
        # self.observations.jaw_camera_feature = None
        # self.observations.scene_camera = None
        # self.observations.scene_camera_feature = None

        # * curriculum
        self.curriculum.action_rate.params["num_steps"] = 24 * 20000
        self.curriculum.joint_vel_1.params["num_steps"] = 24 * 20000
        self.curriculum.joint_vel_2.params["num_steps"] = 24 * 20000
        self.curriculum.grab_spear.params["num_steps"] = 24 * 10000
        self.curriculum.grab_pole.params["num_steps"] = 24 * 10000

        # * events
        # self.events.reset_object_position.params["asset_cfg"].body_names = self.object_name
        # self.events.push_object.params["asset_cfg"].body_names = self.object_name

        # * debug
        debug = True
        self.commands.spear_pose_debug.debug_vis = debug
        self.commands.pole_pose_debug.debug_vis = debug
        self.scene.ee_frame_1.debug_vis = debug
        self.scene.ee_frame_2.debug_vis = debug
        self.scene.spear_connect_frame.debug_vis = debug
        self.scene.pole_connect_frame.debug_vis = debug

        # * commands
        self.commands.pole_pose_debug = None
        self.commands.spear_pose_debug = None


@configclass
class AssembleWeaponDualEnvCfg_PLAY(AssembleWeaponDualEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
