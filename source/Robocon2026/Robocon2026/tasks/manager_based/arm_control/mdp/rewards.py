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

def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    
    extra_factor = torch.where(distance < 0.05, 1.5, 1.0)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std)) * extra_factor

def object_goal_angle(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal orientation using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired orientation in the world frame
    des_quat_b = command[:, 3:7]
    _, des_quat_w = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], None, des_quat_b)
    # compute the difference between desired and current orientation
    object_quat_w = object.data.root_quat_w

    des_roll, des_pitch, des_yaw = euler_xyz_from_quat(des_quat_w)
    object_roll, object_pitch, object_yaw = euler_xyz_from_quat(object_quat_w)

    des_euler = torch.stack([des_roll, des_pitch, des_yaw], dim=1)
    object_euler = torch.stack([object_roll, object_pitch, object_yaw], dim=1)

    angle_diff = torch.norm(des_euler - object_euler, dim=1)

    extra_factor = 1.0
    extra_factor = torch.where(angle_diff < 0.05, 1.5, 1.0)

    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(angle_diff / std)) * extra_factor

def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it."""
    # Get reaching reward
    reach_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
    # Get lifting reward
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)
    # Combine rewards multiplicatively
    return reach_reward * lift_reward


def table_collision(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 与桌子的接触位置
    table_contact_pos = contact_sensor.data.contact_pos_w[:, sensor_cfg.body_ids, 0, :]

    # z方向受力
    z_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    table_contact = ~torch.isnan(table_contact_pos).any(dim=-1) & (z_force > threshold)
    # table_contact = z_force > threshold

    rew = torch.sum(table_contact, dim=1)

    # if rew > 0:
    #     print(f"table_collision with {sensor_cfg.name}")

    return rew


def self_collision(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold

    rew = torch.sum(is_contact, dim=1)
    # if rew > 0:
    #     print(f"self_collision with {sensor_cfg.name}")
    return rew


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
    jaw_contact_pos = jaw_contact_sensor.data.contact_pos_w[:, sensor_cfg_jaw.body_ids, 1, :]
    gripper_contact_pos = gripper_contact_sensor.data.contact_pos_w[:, sensor_cfg_gripper.body_ids, 1, :]
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


def squeeze_object(
    env,
    sensor_cfg: SceneEntityCfg,
    minimal_height: float,
    threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 获取接触力
    force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    max_idx = torch.argmax(force, dim=-1)
    idx_mask = max_idx == 2

    force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    force_z_mask = force_z > threshold

    # 检查是否与物体接触
    contact_pos = contact_sensor.data.contact_pos_w[:, sensor_cfg.body_ids, 1, :]
    contact = ~torch.isnan(contact_pos).any(dim=-1)

    object: RigidObject = env.scene[object_cfg.name]
    # 只有当物体被提升到一定高度时才计算挤压奖励
    height_mask = object.data.root_pos_w[:, 2] < minimal_height

    rew = torch.sum(contact * idx_mask * height_mask * force_z_mask, dim=1)

    # if rew > 0:
    #     print("squeeze_object with", sensor_cfg.name)

    return rew
