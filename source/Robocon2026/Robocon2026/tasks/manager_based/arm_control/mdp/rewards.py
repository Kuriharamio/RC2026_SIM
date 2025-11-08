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
from ray import get

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, ContactSensor
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat

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
    target_quat_w = object.data.root_state_w[:, 3:7]
    # calculate the vertical direction of the object in world frame
    R = matrix_from_quat(target_quat_w)
    target_vertical_dir_w = torch.matmul(R, torch.tensor([0.0, 0.0, 1.0], device=R.device))
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Error-Vector: (num_envs, 3)
    error_vec = target_pos_w - ee_w
    # dot-product of the error-vector and the vertiacal direction of the object
    vertical_distance = torch.abs(torch.sum(error_vec * target_vertical_dir_w, dim=1))
    # Distance of the end-effector to the object: (num_envs,)
    # object_ee_distance = torch.norm(target_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(vertical_distance / std)


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
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


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
    threshold: float = 1.0,
) -> torch.Tensor:
    # 获取接触传感器
    jaw_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg_jaw.name]
    gripper_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg_gripper.name]

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

    # 只有当两个夹爪都接触物体时才给予奖励
    rew = torch.sum(both_contact.float() * opposite_force_reward.float(), dim=1)
    # print(rew.shape)

    # if rew > 0:
    #     print("grab_object")
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
