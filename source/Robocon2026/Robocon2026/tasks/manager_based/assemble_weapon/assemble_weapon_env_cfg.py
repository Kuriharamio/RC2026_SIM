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
class AssembleWeaponSceneCfg(InteractiveSceneCfg):
    """Configuration for the Armdog walking/arm_control scene."""

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
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.08], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(usd_path=f"assets/Table/weapon_table.usd", scale=(1.0, 1.0, 0.3)),
    )
    spear_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SpearTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.3, -0.40, 0.126-0.15], rot=[1, 0, 0, 0]

        ),
        spawn=UsdFileCfg(usd_path=f"assets/Table/weapon_table.usd", scale=(0.2, 0.7, 0.5)),
    )
    pole_rack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PoleRack",
        # init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 1.05], rot=[0, 0, 0, 1]),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-0.187, -0.38, 0.036], rot=[0, 0, 0, 1]
        ),
        spawn=UsdFileCfg(usd_path=f"assets/Table/pole_rack.usd", scale=(0.7, 1.0, 0.6)),
    )
    # * 目标物体
    pole: AssetBaseCfg | RigidObjectCfg = MISSING
    pole_connect_frame: FrameTransformerCfg = MISSING
    spear: AssetBaseCfg | RigidObjectCfg = MISSING
    spear_connect_frame: FrameTransformerCfg = MISSING
    # * 机器人及传感器
    robot1: ArticulationCfg = MISSING
    robot2: ArticulationCfg = MISSING
    # * 末端执行器坐标系1
    ee_frame_1: FrameTransformerCfg = MISSING
    gripper_contact_forces_1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot1/gripper",
        track_contact_points=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        debug_vis=False,
        update_period=0.01,
        max_contact_data_count_per_prim=4096,
    )
    jaw_contact_forces_1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot1/jaw",
        track_contact_points=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        debug_vis=False,
        update_period=0.01,
        max_contact_data_count_per_prim=4096,
    )
    other_contact_forces_1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot1/.*(base|shoulder|upper_arm|lower_arm)",
        debug_vis=False,
        update_period=0.01,
    )
    # * 末端执行器坐标系2
    ee_frame_2: FrameTransformerCfg = MISSING
    gripper_contact_forces_2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot2/gripper",
        track_contact_points=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        debug_vis=False,
        update_period=0.01,
        max_contact_data_count_per_prim=4096,
    )
    jaw_contact_forces_2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot2/jaw",
        track_contact_points=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        debug_vis=False,
        update_period=0.01,
        max_contact_data_count_per_prim=4096,
    )
    other_contact_forces_2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot2/.*(base|shoulder|upper_arm|lower_arm)",
        debug_vis=False,
        update_period=0.01,
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos_1 = mdp.JointPositionActionCfg(
        asset_name="robot1",
        joint_names=["^(?!.*gripper).*$"],
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )

    gripper_action_1 = mdp.BinaryJointPositionActionCfg(
        asset_name="robot1",
        joint_names=["gripper"],
        open_command_expr={"gripper": 0.9},
        close_command_expr={"gripper": -0.0},
    )

    joint_pos_2 = mdp.JointPositionActionCfg(
        asset_name="robot2",
        joint_names=["^(?!.*gripper).*$"],
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )

    gripper_action_2 = mdp.BinaryJointPositionActionCfg(
        asset_name="robot2",
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
        joint_pos_1 = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot1")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_1 = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot1")},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        joint_effort_1 = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot1", joint_names=".*(gripper)")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        joint_pos_2 = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot2")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_2 = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot2")},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        joint_effort_2 = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot2", joint_names=".*(gripper)")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
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
        spear_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot1"),
                "object_cfg": SceneEntityCfg("spear")
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        spear_angles = ObsTerm(
            func=mdp.object_euler_angles_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot1"),
                "object_cfg": SceneEntityCfg("spear"),
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        pole_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot2"),
                "object_cfg": SceneEntityCfg("pole"),
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        pole_angles = ObsTerm(
            func=mdp.object_euler_angles_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot2"),
                "object_cfg": SceneEntityCfg("pole"),
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    privilege: PrivilegeCfg = PrivilegeCfg()


@configclass
class EventCfg:
    """Configuration for events (resets)."""

    # * startup
    physics_material_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot1", body_names=".*"),
            "static_friction_range": (0.5, 4.0),
            "dynamic_friction_range": (0.5, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    physics_material_2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot2", body_names=".*"),
            "static_friction_range": (0.5, 4.0),
            "dynamic_friction_range": (0.5, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_ee_mass_1 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot1", body_names=".*gripper"),
            "mass_distribution_params": (0.0, 0.5),
            "operation": "add",
        },
    )
    add_ee_mass_2 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot2", body_names=".*gripper"),
            "mass_distribution_params": (0.0, 0.5),
            "operation": "add",
        },
    )

    # * reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    assembled = RewTerm(
        func=mdp.weapon_is_assembled,
        params={
            "threshold": 0.02,
            "frame_cfg_1": SceneEntityCfg("spear_connect_frame"),
            "frame_cfg_2": SceneEntityCfg("pole_connect_frame"),
        },
        weight=50.0,
    )

    reaching_spear = RewTerm(
        func=mdp.object_ee_distance, 
        params={
            "object_cfg": SceneEntityCfg("spear"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame_1"), 
            "std": 0.1
        }, 
        weight=2.0
    )
    align_spear = RewTerm(
        func=mdp.object_ee_angle,
        params={
            "object_cfg": SceneEntityCfg("spear"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame_1"),
            "std": 0.3,
        },
        weight=1.0,
    )
    lifting_spear = RewTerm(
        func=mdp.object_is_lifted,
        params={
            "object_cfg": SceneEntityCfg("spear"),
            "minimal_height": 1.07,
        },
        weight=15.0,
    )  
    grab_spear = RewTerm(
        func=mdp.grab_object,
        weight=1.0,
        params={
            "sensor_cfg_jaw": SceneEntityCfg("jaw_contact_forces_1", body_names="jaw"),
            "sensor_cfg_gripper": SceneEntityCfg("gripper_contact_forces_1", body_names="gripper"),
            "object_cfg": SceneEntityCfg("spear"),
            "threshold": 1.5,
        },
    )

    reaching_pole = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "object_cfg": SceneEntityCfg("pole"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame_2"),
            "std": 0.1,
        },
        weight=2.0,
    )
    align_pole = RewTerm(
        func=mdp.object_ee_angle,
        params={
            "object_cfg": SceneEntityCfg("pole"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame_2"),
            "std": 0.3,
        },
        weight=1.0,
    )
    lifting_pole = RewTerm(
        func=mdp.object_is_lifted,
        params={
            "object_cfg": SceneEntityCfg("pole"),
            "minimal_height": 1.07,
        },
        weight=15.0,
    )
    grab_pole = RewTerm(
        func=mdp.grab_object,
        weight=1.0,
        params={
            "sensor_cfg_jaw": SceneEntityCfg("jaw_contact_forces_2", body_names="jaw"),
            "sensor_cfg_gripper": SceneEntityCfg(
                "gripper_contact_forces_2", body_names="gripper"
            ),
            "object_cfg": SceneEntityCfg("pole"),
            "threshold": 1.5,
        },
    )

    assemble_dist = RewTerm(
        func=mdp.assemble_distance,
        params={
            "std": 0.2,
            "minimal_height": 1.07,
            "frame_cfg_1": SceneEntityCfg("spear_connect_frame"),
            "frame_cfg_2": SceneEntityCfg("pole_connect_frame"),
        },
        weight=16.0,
    )
    assemble_dist_fine_grained = RewTerm(
        func=mdp.assemble_distance,
        params={
            "std": 0.2,
            "minimal_height": 1.07,
            "frame_cfg_1": SceneEntityCfg("spear_connect_frame"),
            "frame_cfg_2": SceneEntityCfg("pole_connect_frame"),
        },
        weight=5.0,
    )

    assemble_angle = RewTerm(
        func=mdp.assemble_angle,
        params={
            "std": 0.3,
            "minimal_height": 1.07,
            "frame_cfg_1": SceneEntityCfg("spear_connect_frame"),
            "frame_cfg_2": SceneEntityCfg("pole_connect_frame"),
        },
        weight=16.0,
    )
    assemble_angle_fine_grained = RewTerm(
        func=mdp.assemble_angle,
        params={
            "std": 0.3,
            "minimal_height": 1.07,
            "frame_cfg_1": SceneEntityCfg("spear_connect_frame"),
            "frame_cfg_2": SceneEntityCfg("pole_connect_frame"),
        },
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel_1 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot1")},
    )
    joint_vel_2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot2")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Spear dropping
    spear_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.03, 
            "asset_cfg": SceneEntityCfg("spear"),
        },
    )
    # (3) Pole dropping
    pole_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.03,
            "asset_cfg": SceneEntityCfg("pole"),
        },
    )
    # (4) Assemble successfully
    weapon_is_assembled = DoneTerm(
        func=mdp.weapon_is_assembled,
        params={
            "threshold": 0.02,
            "frame_cfg_1": SceneEntityCfg("spear_connect_frame"),
            "frame_cfg_2": SceneEntityCfg("pole_connect_frame"),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 30000},
    )

    joint_vel_1 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel_1", "weight": -1e-2, "num_steps": 30000},
    )

    joint_vel_2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel_2", "weight": -1e-2, "num_steps": 30000},
    )

    grab_spear = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "grab_spear", "weight": 5.0, "num_steps": 10000},
    )
    grab_pole = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "grab_pole", "weight": 5.0, "num_steps": 10000},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    spear_pose_debug = mdp.UniformPoseCommandCfg(
        asset_name="spear",
        body_name="spear_combine",
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

    pole_pose_debug = mdp.UniformPoseCommandCfg(
        asset_name="pole",
        body_name="pole_combine",
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


##
# Environment configuration
##
@configclass
class AssembleWeaponBaseEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: AssembleWeaponSceneCfg = AssembleWeaponSceneCfg(
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
