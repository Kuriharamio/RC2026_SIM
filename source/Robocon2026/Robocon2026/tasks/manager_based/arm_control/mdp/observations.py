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

import torch
import torch.nn as nn

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, euler_xyz_from_quat
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.envs.mdp.observations import image
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_shape,
)

from typing import Dict, Callable, Optional, List
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)
from transformers import AutoModel
from Robocon2026.utils.utils import print_green, print_red, print_yellow

@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action_check(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        last_action = env.action_manager.action
    else:
        last_action = env.action_manager.get_term(action_name).raw_actions

    if torch.max(last_action) > 100:
        print(f"Warning! Action {torch.max(last_action)} is out of range")

    return last_action


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w)
    return object_pos_b


def object_euler_angles_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """The euler angles of the object."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    # object_quat = object.data.root_state_w[:, 3:7]
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        object.data.root_state_w[:, :3],
        object.data.root_state_w[:, 3:7],
    )

    roll, pitch, yaw = euler_xyz_from_quat(object_quat_b)
    object_euler = torch.stack([roll, pitch, yaw], dim=1)
    return object_euler


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def command_pose_angle(env: ManagerBasedRLEnv, command_name: str | None = None) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    command = env.command_manager.get_command(command_name)
    pos = command[:, :3]
    quat = command[:, 3:7]
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    euler_angles = torch.stack([roll, pitch, yaw], dim=1)
    return torch.cat([pos, euler_angles], dim=1)


class ResNet10Extractor(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        print_yellow("Loading ResNet10...")
        self.model = AutoModel.from_pretrained("helper2424/resnet10", trust_remote_code=True)
        self.model = self.model.to(env.device)
        print_green("ResNet10 loaded")
        # print(self.model)
        self.model.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(env.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
    ) -> torch.Tensor:

        image_data = image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=False,
        )
        image_data = image_data.permute(0, 3, 1, 2)

        image_data = image_data.float() / 255.0
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)
        image_data = (image_data - mean) / std

        with torch.no_grad():
            features = self.model(image_data)["pooler_output"].squeeze(-1).squeeze(-1)
    
        return features
