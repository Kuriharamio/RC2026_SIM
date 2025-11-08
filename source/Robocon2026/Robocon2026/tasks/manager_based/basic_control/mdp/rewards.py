# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def action_rate_l2_clip(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    action_rate = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    action_rate = torch.clip(action_rate, min=0.0, max=50.0)
    # print("action_rate: ", action_rate)
    return action_rate


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_stumble(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    rew = torch.any(
        torch.norm(net_contact_forces[:, :, :2], dim=2)
        > 4 * torch.abs(net_contact_forces[:, :, 2]),
        dim=1,
    )
    return rew.float()

def feet_edge_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 3.0,
    terrain_level_threshold: int = 3,
) -> torch.Tensor:
    """惩罚足部与地形边缘接触，鼓励足部放置在稳定的地面上"""
    # 提取接触传感器和机器人资产
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    # 获取世界坐标系中的足部位置
    feet_pos_w = robot.data.body_pos_w[ :, sensor_cfg.body_ids, :2]  # (num_envs, num_feet, 2)
    # 获取足部的接触信息
    feet_contact = (contact_sensor.data.net_forces_w.norm(dim=-1)[:, sensor_cfg.body_ids] > 0.1)  # (num_envs, num_feet)

    # 在粗糙地形中，我们检查足部是否靠近边缘
    if env.scene.terrain.terrain_types is not "plane":
        # 计算足部位置的分散度以确定是否处于边缘
        feet_center = torch.mean(feet_pos_w, dim=1, keepdim=True)  # (num_envs, 1, 2)
        feet_spread = torch.std(feet_pos_w, dim=1)  # (num_envs, 2)

        # 基于位置方差的简单边缘检测
        # 当足部处于边缘时，它们的位置分布往往更加分散
        edge_indicator = torch.norm(feet_spread, dim=1)  # (num_envs,)

        # 根据边缘指示器和接触状态应用惩罚
        penalty = torch.sum(feet_contact.float(), dim=1) * edge_indicator

        # 仅对较高难度的地形应用惩罚
        # self.terrain_levels = torch.randint(0, max_init_level + 1, (num_envs,), device=self.device)
        terrain_difficulty = env.scene.terrain.terrain_levels
        # 仅在地形难度高于阈值时应用惩罚
        terrain_mask = terrain_difficulty > terrain_level_threshold

        return penalty * terrain_mask

    else:
        return torch.zeros(env.num_envs, device=env.device)

# class feet_edge(ManagerTermBase):
#     def __init__(self, cfg: RewardTermCfg, env):
#         super().__init__(cfg, env)
#         self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
#         self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
#         self.sensor_cfg = cfg.params["sensor_cfg"]
#         self.asset_cfg = cfg.params["asset_cfg"]
#         self.body_id = self.contact_sensor.find_bodies('base')[0]
#         self.horizontal_scale = env.scene.terrain.cfg.terrain_generator.horizontal_scale
#         size_x, size_y = env.scene.terrain.cfg.terrain_generator.size
#         self.rows_offset = (size_x * env.scene.terrain.cfg.terrain_generator.num_rows/2)
#         self.cols_offset = (size_y * env.scene.terrain.cfg.terrain_generator.num_cols/2)

#         self.parkour_event: ParkourEvent =  env.parkour_manager.get_term(cfg.params["parkour_name"])
#         total_x_edge_maskes = torch.from_numpy(self.parkour_event.terrain.terrain_generator_class.x_edge_maskes).to(device = self.device)
#         self.x_edge_masks_tensor = total_x_edge_maskes.permute(0, 2, 1, 3).reshape(
#             env.scene.terrain.terrain_generator_class.total_width_pixels, env.scene.terrain.terrain_generator_class.total_length_pixels
#         )

#     def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#         feet_pos_x = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,0] + self.rows_offset)
#                       /self.horizontal_scale).round().long()
#         feet_pos_y = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,1] + self.cols_offset)
#                       /self.horizontal_scale).round().long()
#         feet_pos_x = torch.clip(feet_pos_x, 0, self.x_edge_masks_tensor.shape[0]-1)
#         feet_pos_y = torch.clip(feet_pos_y, 0, self.x_edge_masks_tensor.shape[1]-1)
#         feet_at_edge = self.x_edge_masks_tensor[feet_pos_x, feet_pos_y]
#         contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
#         previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
#         contact = torch.norm(contact_forces, dim=-1) > 2.
#         last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
#         contact_filt = torch.logical_or(contact, last_contacts)
#         self.feet_at_edge = contact_filt & feet_at_edge
#         rew = (self.parkour_event.terrain.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
#         return rew
