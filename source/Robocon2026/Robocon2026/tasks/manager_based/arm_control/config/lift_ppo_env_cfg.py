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
from Robocon2026.tasks.manager_based.arm_control.arm_control_env_cfg import ArmControlEnvCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from Robocon2026.tasks.manager_based.arm_control import mdp
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class ArmControlLiftEnvCfg(ArmControlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # * scene
        self.scene.num_envs = 4096
        self.scene.robot = SO101_FOLLOWER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=[0.0, -0.2, 0.47],
                # pos=[0.02319, -0.48631, 1.01896],
                rot=[0, 0, 0, 1],
                joint_pos={
                    "shoulder_pan": 0.0,
                    "shoulder_lift": 0.0,
                    # "elbow_flex": 0.8,
                    # "wrist_flex": -0.8,
                    # "wrist_roll": 1.57,
                    "elbow_flex": 0.0,
                    "wrist_flex": 0.0,
                    "wrist_roll": 1.57,
                    "gripper": 0.3,
                },
                joint_vel={".*": 0.0},
            ),
        )
        self.scene.jaw_camera = None
        self.scene.scene_camera = None
        self.object_height = 0.55

        self.object_name = "spear_combine"
        self.scale = (0.75, 0.75, 0.75)
        rot = [0.707, 0, 0, 0.707]

        # self.object_name = "cylinder"
        # self.scale = (0.03, 0.03, 0.05)
        # rot = [0.707, 0, 0, 0.707]

        # self.object_name = "pole_combine"
        # self.scale = (0.75, 0.75, 0.75)
        # rot = [0.707, 0, 0.707, 0]

        self.scene.target_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, self.object_height],
                rot=rot,
            ),
            spawn=UsdFileCfg(
                usd_path=f"assets/Object/{self.object_name}.usd",
                scale=self.scale,
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
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.005, 0.0, -0.095], rot=[0.707, 0.707, 0, 0]
                    ),
                ),
            ],
        )
        self.scene.gripper_contact_forces.filter_prim_paths_expr = [
            "{ENV_REGEX_NS}/Table/weapon_table",
            "{ENV_REGEX_NS}/Object/" + self.object_name,
        ]
        self.scene.jaw_contact_forces.filter_prim_paths_expr = [
            "{ENV_REGEX_NS}/Table/weapon_table",
            "{ENV_REGEX_NS}/Object/" + self.object_name,
        ]

        # * rewards
        error_height = 0.025
        self.rewards.lifting_object.params["minimal_height"] = self.object_height + error_height
        self.rewards.object_goal_tracking_dist.params["minimal_height"] = self.object_height + error_height
        self.rewards.object_goal_tracking_dist_fine_grained.params["minimal_height"] = self.object_height + error_height
        self.rewards.object_goal_tracking_angle.params["minimal_height"] = self.object_height + error_height
        self.rewards.object_goal_tracking_angle_fine_grained.params["minimal_height"] = self.object_height + error_height
        self.rewards.squeeze_object_jaw.params["minimal_height"] = self.object_height + error_height
        self.rewards.squeeze_object_gripper.params["minimal_height"] = self.object_height + error_height
        self.rewards.grab_object.weight = 10.0
        self.rewards.lifting_object.weight = 10.0
        self.rewards.object_goal_tracking_dist.weight = 20.0
        self.rewards.object_goal_tracking_angle.weight = 12.0
        self.rewards.squeeze_object_jaw.weight = -1e-5
        self.rewards.squeeze_object_gripper.weight = -1e-5

        # * observations
        self.observations.jaw_camera = None
        self.observations.jaw_camera_feature = None
        self.observations.scene_camera = None
        self.observations.scene_camera_feature = None

        # * curriculum
        self.curriculum.action_rate.params["num_steps"] = 24 * 600
        self.curriculum.joint_vel.params["num_steps"] = 24 * 600
        self.curriculum.grab_object.params["num_steps"] = 24 * 300

        # * events
        self.events.reset_object_position.params["asset_cfg"].body_names = self.object_name
        self.events.push_object.params["asset_cfg"].body_names = self.object_name

        # * commands
        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["gripper"]
        self.commands.object_pose_debug = mdp.UniformPoseCommandCfg(
            asset_name="target_object",
            body_name=self.object_name,
            resampling_time_range=(5.0, 5.0),
            debug_vis=False,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.0, 0.0),
                pos_y=(0.0, 0.0),
                pos_z=(0.0, 0.0),
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(0.0, 0.0),
            ),
            current_pose_visualizer_cfg=FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/body_pose",
                markers={
                    "frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(0.05, 0.05, 0.05),
                    ),
                },
            ),
            goal_pose_visualizer_cfg=FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/body_pose",
                markers={
                    "frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(0.015, 0.015, 0.015),
                    ),
                },
            ),
        )

        # * debug
        debug = True
        self.commands.object_pose.debug_vis = debug
        self.commands.object_pose_debug.debug_vis = debug
        self.scene.jaw_contact_forces.debug_vis = debug
        self.scene.gripper_contact_forces.debug_vis = debug
        self.scene.other_contact_forces.debug_vis = debug
        self.scene.ee_frame.debug_vis = debug
@configclass
class ArmControlLiftEnvCfg_PLAY(ArmControlLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
