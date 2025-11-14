# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import table
from onnx_ir import val
from ray import get

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, ContactSensor
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    # print("object_is_lifted: ",torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0))
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    target_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Position distance between end-effector and object
    position_distance = torch.norm(target_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(position_distance / std)

def object_ee_angle(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    target_quat_w = object.data.root_state_w[:, 3:7]
    # End-effector position: (num_envs, 3)
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]

    # Calculate the z-axis direction of both object and end-effector in world frame
    object_R = matrix_from_quat(target_quat_w)
    object_z_dir_w = object_R[:, :, 2]  # Z-axis is column 2

    ee_R = matrix_from_quat(ee_quat_w)
    ee_z_dir_w = ee_R[:, :, 2]  # Z-axis is column 2

    # Compute alignment of z-axes (dot product, 1 if parallel, 0 if perpendicular)
    z_alignment = torch.abs(torch.sum(object_z_dir_w * ee_z_dir_w, dim=1))

    return torch.tanh(z_alignment / std)

def grab_object(
    env,
    sensor_cfg_jaw: SceneEntityCfg,
    sensor_cfg_gripper: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    threshold: float = 1.0,
) -> torch.Tensor:
    # 获取接触传感器
    jaw_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg_jaw.name]
    gripper_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg_gripper.name]
    object: RigidObject = env.scene[object_cfg.name]

    # 获取两个夹爪的接触力
    jaw_force = jaw_contact_sensor.data.net_forces_w[:, sensor_cfg_jaw.body_ids, :]  # 左夹爪受力
    gripper_force = gripper_contact_sensor.data.net_forces_w[:, sensor_cfg_gripper.body_ids, :] # 右夹爪受力

    # 检查两个夹爪是否都与物体接触
    jaw_contact_pos = jaw_contact_sensor.data.contact_pos_w[:, sensor_cfg_jaw.body_ids, 0, :]
    gripper_contact_pos = gripper_contact_sensor.data.contact_pos_w[:, sensor_cfg_gripper.body_ids, 0, :]
    jaw_contact = ~torch.isnan(jaw_contact_pos).any(dim=-1)
    gripper_contact = ~torch.isnan(gripper_contact_pos).any(dim=-1)

    both_contact = jaw_contact & gripper_contact

    # 归一化力向量
    jaw_force_norm = torch.nn.functional.normalize(jaw_force, dim=-1)
    gripper_force_norm = torch.nn.functional.normalize(gripper_force, dim=-1)

    # 计算两个力向量的点积（越接近-1表示方向越相反）
    force_dot_product = torch.sum(jaw_force_norm * gripper_force_norm, dim=-1)

    # 奖励力方向相反的情况，使用 (1 + dot_product) 使得方向完全相反时奖励最大
    opposite_force_reward = (1 - force_dot_product) > threshold

    # 计算夹爪与物体位置的距离，用于调整奖励系数
    object_pos = object.data.root_pos_w[:, :3]

    valid_indices = both_contact.nonzero(as_tuple=True)[0]
    extra_factor = torch.zeros_like(both_contact.float())

    jaw_distance = torch.norm((jaw_contact_pos[valid_indices, 0, :]) - object_pos[valid_indices, :], dim=-1)
    gripper_distance = torch.norm((gripper_contact_pos[valid_indices, 0,  :]) - object_pos[valid_indices, :], dim=-1)
    avg_distance = (jaw_distance + gripper_distance) / 2
    # 使用指数衰减函数，使得距离越近奖励系数越高
    extra_factor[valid_indices] = (0.5 + torch.exp(-avg_distance / 0.05)).unsqueeze(1)

    # 只有当两个夹爪都接触物体时才给予奖励，并乘以距离奖励系数
    rew = torch.sum(both_contact.float() * opposite_force_reward.float() * extra_factor, dim=1) 

    return rew


def assemble_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    frame_cfg_1: SceneEntityCfg = SceneEntityCfg("frame1"),
    frame_cfg_2: SceneEntityCfg = SceneEntityCfg("frame2"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    frame_1: FrameTransformer = env.scene[frame_cfg_1.name]
    frame_2: FrameTransformer = env.scene[frame_cfg_2.name]

    distance = torch.norm(
        frame_1.data.target_pos_w[..., 0, :] - frame_2.data.target_pos_w[..., 0, :],
        dim=1,
    )

    extra_factor = torch.where(distance < 0.05, 1.5, 1.0)
    # rewarded if the frame is lifted above the threshold
    return (1 - torch.tanh(distance / std)) * extra_factor * torch.where(frame_1.data.target_pos_w[..., 0, 2] > minimal_height, 1.0, 0.0)


def assemble_angle(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    threshold: float = 0.2,
    frame_cfg_1: SceneEntityCfg = SceneEntityCfg("frame1"),
    frame_cfg_2: SceneEntityCfg = SceneEntityCfg("frame2"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal orientation using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    frame_1: FrameTransformer = env.scene[frame_cfg_1.name]
    frame_2: FrameTransformer = env.scene[frame_cfg_2.name]

    frame_1_quat = frame_1.data.target_quat_w[..., 0, :]
    # roll_1, pitch_1, yaw_1 = euler_xyz_from_quat(frame_1_quat)
    # frame_1_euler = torch.stack([roll_1, pitch_1, yaw_1], dim=1)

    frame_2_quat = frame_2.data.target_quat_w[..., 0, :]
    # roll_2, pitch_2, yaw_2 = euler_xyz_from_quat(frame_2_quat)
    # frame_2_euler = torch.stack([roll_2, pitch_2, yaw_2], dim=1)

    # Calculate the z-axis direction of both frame and end-effector in world frame
    frame_1_R = matrix_from_quat(frame_1_quat)
    frame_1_z_dir_w = frame_1_R[:, :, 2]  # Z-axis is column 2

    frame_2_R = matrix_from_quat(frame_2_quat)
    frame_2_z_dir_w = frame_2_R[:, :, 2]  # Z-axis is column 2

    # Compute alignment of z-axes (dot product, 1 if parallel, 0 if perpendicular)
    z_alignment = torch.abs(torch.sum(frame_1_z_dir_w * frame_2_z_dir_w, dim=1))

    distance = torch.norm(
        frame_1.data.target_pos_w[..., 0, :] - frame_2.data.target_pos_w[..., 0, :],
        dim=1,
    )
    close = distance < threshold

    return torch.tanh(z_alignment / std)  * close * torch.where(frame_1.data.target_pos_w[..., 0, 2] > minimal_height, 1.0, 0.0)


# def weapon_is_assembled(
#     env: ManagerBasedRLEnv,
#     threshold: float = 0.02,
#     frame_cfg_1: SceneEntityCfg = SceneEntityCfg("frame1"),
#     frame_cfg_2: SceneEntityCfg = SceneEntityCfg("frame2"),
# ) -> torch.Tensor:
#     # extract the used quantities (to enable type-hinting)
#     frame_1: FrameTransformer = env.scene[frame_cfg_1.name]
#     frame_2: FrameTransformer = env.scene[frame_cfg_2.name]

#     distance = torch.norm(
#         frame_1.data.target_pos_w[..., 0, :] - frame_2.data.target_pos_w[..., 0, :],
#         dim=1,
#     )

#     return distance < threshold
