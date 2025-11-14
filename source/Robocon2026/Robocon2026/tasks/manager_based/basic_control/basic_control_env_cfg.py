# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

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
from wandb import teardown

from . import mdp

##
# Pre-defined configs
##
from isaaclab.sensors import ImuCfg, CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from Robocon2026.map.terrains import TERRAINS_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Scene definition
##
@configclass
class BasicControlSceneCfg(InteractiveSceneCfg):
    """Configuration for the Robot walking/basic_control scene."""

    # * 地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"assets/Matrials/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # * 天空
    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=750.0,
    #         texture_file="assets/Matrials/kloofendal_43d_clear_puresky_4k.hdr",
    #     ),
    # )
    # * 穹顶灯光
    Domelight = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    # * 机器人及传感器
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=True,
        update_period=0.0,
    )
    # imu = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/imu",
    #     debug_vis=True
    # )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*L_hip_joint",
            ".*R_hip_joint",
            "F[L,R]_thigh_joint",
            "R[L,R]_thigh_joint",
            ".*_calf_joint",
        ],
        clip={
            ".*L_hip_joint": (-5.0, 5.0),
            ".*R_hip_joint": (-5.0, 5.0),
            "F[L,R]_thigh_joint": (-5.0, 5.0),
            "R[L,R]_thigh_joint": (-5.0, 5.0),
            ".*_calf_joint": (-5.0, 5.0),
        },
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )

    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*_foot_joint"],
        clip={".*_foot_joint": (-30.0, 30.0)},
        scale=10.0,
        use_default_offset=True,
        preserve_order=True,
    )

    # arm_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*"],
    #     scale=0.25,
    #     use_default_offset=True,
    #     preserve_order=True,
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # imu_gyro = ObsTerm(
        #     func=mdp.imu_ang_vel,
        #     params={"asset_cfg": SceneEntityCfg("imu")},
        #     noise=Unoise(n_min=-0.2, n_max=0.2),
        # )
        # imu_accel = ObsTerm(
        #     func=mdp.imu_lin_acc,
        #     params={"asset_cfg": SceneEntityCfg("imu")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )
        # imu_projected_gravity = ObsTerm(
        #     func=mdp.imu_projected_gravity,
        #     params={"asset_cfg": SceneEntityCfg("imu")},
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*L_hip_joint",
                        ".*R_hip_joint",
                        "F[L,R]_thigh_joint",
                        "R[L,R]_thigh_joint",
                        ".*_calf_joint",
                    ],
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action_check, clip=(-50.0, 50.0))

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegeCfg(ObsGroup):
        """Observations for privilege group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*L_hip_joint",
                        ".*R_hip_joint",
                        "F[L,R]_thigh_joint",
                        "R[L,R]_thigh_joint",
                        ".*_calf_joint",
                    ],
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action_check, clip=(-50.0, 50.0))
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    privilege: PrivilegeCfg = PrivilegeCfg()


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

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # add_ee_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*gripper"),
    #         "mass_distribution_params": (0.0, 0.5),
    #         "operation": "add",
    #     },
    # )

    # * reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # actuator_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.8, 1.2),
    #         "damping_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # * interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # 奖励机器人跟踪xy平面线速度命令的表现
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # 奖励机器人跟踪绕z轴角速度命令的表现
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # 惩罚机器人在z轴方向的线速度（避免不必要的上下运动）
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # 惩罚机器人绕x、y轴的角速度（避免翻滚和俯仰）
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 惩罚机器人偏离水平姿态
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    # 惩罚关节加速度，鼓励平滑的动作
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*L_hip_joint",
                    ".*R_hip_joint",
                    "F[L,R]_thigh_joint",
                    "R[L,R]_thigh_joint",
                    ".*_calf_joint",
                ],
            ),
        },
        weight=-2.5e-7,
    )
    # 惩罚机器人身体部位与环境发生碰撞
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*foot).*$"),
            "threshold": 1.0,
        },
    )
    # 惩罚动作变化率，鼓励动作的连续性和平滑性
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clip, weight=-0.01)
    # 惩罚关节力矩，鼓励节能的动作
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*L_hip_joint",
                    ".*R_hip_joint",
                    "F[L,R]_thigh_joint",
                    "R[L,R]_thigh_joint",
                    ".*_calf_joint",
                ],
            ),
        },
        weight=-2.0e-4,
    )
    # 奖励足部离地时间，鼓励机器人抬脚行走
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    # 惩罚hip关节位置偏差
    hip_pos_error = RewTerm(
        func=mdp.joint_deviation_l1, 
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*hip_joint"),
        },
    )
    # # 惩罚关节位置偏差
    dof_pos_error = RewTerm(
        func=mdp.joint_deviation_l1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*L_hip_joint",
                    ".*R_hip_joint",
                    "F[L,R]_thigh_joint",
                    "R[L,R]_thigh_joint",
                    ".*_calf_joint",
                ],
            ),
        },
        weight=-0.004,
    )
    # 惩罚足部撞击垂直表面（绊倒）
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
        },
    )

    # 惩罚高度过低过高的行为
    if hasattr(SceneEntityCfg, "height_scanner"):
        base_height_penalty = RewTerm(
            func=mdp.base_height_l2,
            weight=-0.1,
            params={
                "target_height": 0.275,
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
    else:
        base_height_penalty = RewTerm(
            func=mdp.base_height_l2,
            weight=-0.1,
            params={
                "target_height": 0.275,
            },
        )
    # 惩罚接近关节位置极限的情况
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*L_hip_joint",
                    ".*R_hip_joint",
                    "F[L,R]_thigh_joint",
                    "R[L,R]_thigh_joint",
                    ".*_calf_joint",
                ],
            ),
        },
        weight=-1.0,
    )

    # 惩罚足部滑行
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
        },
    )
    # 惩罚足部撞击边缘
    # feet_edge = RewTerm(
    #     func=mdp.feet_edge,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
    #         "threshold": 0.5,
    #         "terrain_level_threshold": 3,
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Base contact
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )
    # # (3) Arm contact
    # arm_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(ebow|jaw|arm_base)"),
    #         "threshold": 1.0,
    #     },
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(
                -math.pi,
                math.pi
            ),
        ),
    )


##
# Environment configuration
##
@configclass
class BasicControlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BasicControlSceneCfg = BasicControlSceneCfg(
        num_envs=4096, env_spacing=2.5#, clone_in_fabric=True
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
        self.decimation = 4 
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 3.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.physx.bounce_threshold_velocity = 0.2
        # # default friction material
        # self.sim.physics_material.static_friction = 1.0
        # self.sim.physics_material.dynamic_friction = 1.0
        # self.sim.physics_material.restitution = 0.0
        if self.scene.terrain is not None:
            self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if self.scene.terrain is not None:
            if getattr(self.curriculum, "terrain_levels", None) is not None:
                if self.scene.terrain.terrain_generator is not None:
                    self.scene.terrain.terrain_generator.curriculum = True
            else:
                if self.scene.terrain.terrain_generator is not None:
                    self.scene.terrain.terrain_generator.curriculum = False

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
