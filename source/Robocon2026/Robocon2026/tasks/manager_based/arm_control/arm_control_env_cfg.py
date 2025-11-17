# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from pickle import FALSE

from Robocon2026.tasks.manager_based.arm_control.mdp.rewards import grab_object, table_collision
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from numpy import squeeze
from sympy import false, im

from . import mdp

##
# Pre-defined configs
##
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Scene definition
##
@configclass
class ArmControlSceneCfg(InteractiveSceneCfg):
    """Configuration for the Armdog walking/arm_control scene."""

    # * 穹顶灯光
    # Domelight = AssetBaseCfg(
    #     prim_path="/World/Light",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=3000.0,
    #         color=(0.75, 0.75, 0.75),
    #     ),
    # )
    # * 天空
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file="assets/Materials/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    # * plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0.0]),
        spawn=GroundPlaneCfg(),
    )
    # * 桌面
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        # init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 1.05], rot=[0, 0, 0, 1]),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.25], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=UsdFileCfg(usd_path=f"assets/Table/weapon_table.usd", scale=(1.0, 4.0, 1.0)),
        # spawn=UsdFileCfg(usd_path=f"assets/Table/table_instanceable.usd")
    )
    # * 目标物体
    target_object: AssetBaseCfg | RigidObjectCfg = MISSING

    # * 机器人及传感器
    robot: ArticulationCfg = MISSING
    # * 末端执行器坐标系
    ee_frame: FrameTransformerCfg = MISSING
    gripper_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper",
        track_contact_points=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/weapon_table", "{ENV_REGEX_NS}/Object"],
        debug_vis=False,
        update_period=0.01,
        max_contact_data_count_per_prim=4096,
    )
    jaw_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        track_contact_points=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/weapon_table", "{ENV_REGEX_NS}/Object"],
        debug_vis=False,
        update_period=0.01,
        max_contact_data_count_per_prim=4096,
    )
    other_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*(base|shoulder|upper_arm|lower_arm)",
        debug_vis=False,
        update_period=0.01,
    )

    jaw_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw/camera",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.08, 0.3),
            rot=(0.185, 0.020, 0.274, 0.943),
            convention="opengl",
        ),
    )

    scene_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Table/camera",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.85, -0.13837, 0.36327),
            rot=(0.622, 0.615, 0.343, 0.341),
            convention="opengl",
        ),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["^(?!.*gripper).*$"],
        # joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": 0.9},
        close_command_expr={"gripper": -0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*(gripper)")
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        target_object_position = ObsTerm(
            func=mdp.command_pose_angle,
            params={"command_name": "object_pose"},
            noise=Unoise(n_min=-0.00, n_max=0.00),
        )
        actions = ObsTerm(
            func=mdp.last_action_check, 
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegeCfg(ObsGroup):
        ''' Privileged Observation '''
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        object_angles = ObsTerm(
            func=mdp.object_euler_angles_in_robot_root_frame,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

    @configclass
    class JawImgCfg(ObsGroup):
        ''' Jaw Camera Observation '''
        jaw_img = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("jaw_camera"),
                "data_type": "rgb",
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

    @configclass
    class SceneImgCfg(ObsGroup):
        ''' Jaw Camera Observation '''
        scene_img = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("scene_camera"),
                "data_type": "rgb",
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

    @configclass
    class JawImgFeatureCfg(ObsGroup):
        ''' Jaw Camera ResNet10 Features  Observation '''
        jaw_img_feature = ObsTerm(
            func=mdp.ResNet10Extractor,
            params={
                "sensor_cfg": SceneEntityCfg("jaw_camera"),
                "data_type": "rgb",
                "convert_perspective_to_orthogonal": False,
            },
            noise=Unoise(n_min=0.0, n_max=0.0),
        )

    @configclass
    class SceneImgFeatureCfg(ObsGroup):
        ''' Scene Camera ResNet10 Features  Observation '''
        scene_img_feature = ObsTerm(
            func=mdp.ResNet10Extractor,
            params={
                "sensor_cfg": SceneEntityCfg("scene_camera"),
                "data_type": "rgb",
                "convert_perspective_to_orthogonal": False,
            },
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    privilege: PrivilegeCfg = PrivilegeCfg()
    jaw_camera: JawImgCfg = JawImgCfg()
    jaw_camera_feature: JawImgFeatureCfg = JawImgFeatureCfg()
    scene_camera: SceneImgCfg = SceneImgCfg()
    scene_camera_feature: SceneImgFeatureCfg = SceneImgFeatureCfg()


@configclass
class EventCfg:
    """Configuration for events (resets)."""

    # * startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 4.0),
            "dynamic_friction_range": (0.5, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_ee_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*gripper"),
            "mass_distribution_params": (0.0, 0.5),
            "operation": "add",
        },
    )

    # * reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (0.1, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target_object", body_names="target_object"),
        },
    )

    push_object = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="reset",
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            "asset_cfg": SceneEntityCfg("target_object", body_names="target_object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=2.0)

    align_object = RewTerm(func=mdp.object_ee_angle, params={"std": 0.3}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 1.07}, weight=15.0)

    object_goal_tracking_dist = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 1.07, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_dist_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 1.07, "command_name": "object_pose"},
        weight=5.0,
    )

    object_goal_tracking_angle = RewTerm(
        func=mdp.object_goal_angle,
        params={"std": 1.0, "minimal_height": 1.07, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_angle_fine_grained = RewTerm(
        func=mdp.object_goal_angle,
        params={"std": 0.1, "minimal_height": 1.07, "command_name": "object_pose"},
        weight=5.0,
    )

    grab_object = RewTerm(
        func=mdp.grab_object,
        weight=1.0,
        params={
            "sensor_cfg_jaw": SceneEntityCfg("jaw_contact_forces", body_names="jaw"),
            "sensor_cfg_gripper": SceneEntityCfg("gripper_contact_forces", body_names="gripper"),
            "threshold": 1.5,
        },
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    table_collision_jaw = RewTerm(
        func=mdp.table_collision,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("jaw_contact_forces", body_names="jaw"),
            "threshold": 1.0,
        },
    )

    table_collision_gripper = RewTerm(
        func=mdp.table_collision,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("gripper_contact_forces", body_names="gripper"),
            "threshold": 1.0,
        },
    )

    self_collision = RewTerm(
        func=mdp.self_collision,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("other_contact_forces", body_names="(base|shoulder|upper_arm|lower_arm)",),
            "threshold": 1.0,
        },
    )

    squeeze_object_gripper = RewTerm(
        func=mdp.squeeze_object,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("gripper_contact_forces", body_names="gripper"),
            "minimal_height": 0.04,
            "threshold": 1.0,
        },
    )

    squeeze_object_jaw = RewTerm(
        func=mdp.squeeze_object,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("jaw_contact_forces", body_names="jaw"),
            "minimal_height": 0.04,
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Object dropping
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4, "asset_cfg": SceneEntityCfg("target_object")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 30000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 30000},
    )

    grab_object = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "grab_object", "weight": 5.0, "num_steps": 10000},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.2, -0.2),
            pos_z=(0.2, 0.35),
            roll=(-0.8, 0.8),
            pitch=(-0.8, 0.8),
            yaw=(-1.57, -1.57),
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
                    scale=(0.05, 0.05, 0.05),
                ),
            },
        ),
    )


##
# Environment configuration
##
@configclass
class ArmControlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: ArmControlSceneCfg = ArmControlSceneCfg(
        num_envs=4096, env_spacing=2.5  # , clone_in_fabric=True
    )

    # Basic MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization and simulation tuning for walking."""
        # control settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # viewer settings
        self.viewer.eye = (0.5, 0.5, 2.0)
        self.viewer.target = (-1.0, -2.0, -0.5)
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 8 * 1024 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_collision_stack_size = 256 * 1024 * 1024
