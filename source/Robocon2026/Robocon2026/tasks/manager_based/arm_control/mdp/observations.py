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

def object_euler_angles_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """The euler angles of the object."""
    object: RigidObject = env.scene[object_cfg.name]
    object_quat = object.data.root_state_w[:, 3:7]
    roll, pitch, yaw = euler_xyz_from_quat(object_quat)
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


class ResNetFeatureExtractor(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # 基础配置参数
        self.model_name: str = cfg.params.get("model_name", "resnet18")
        self.model_device: str = cfg.params.get("model_device", env.device)
        self.model_zoo_cfg: Optional[Dict] = cfg.params.get("model_zoo_cfg")

        # 支持的默认模型列表
        self.default_resnet_models: List[str] = ["resnet18", "resnet34", "resnet50", "resnet101"]

        # 加载模型
        self.backbone: nn.Module 
        self._inference_fn: Callable
        self._reset_fn: Optional[Callable] = None

        # 初始化模型
        self._init_model()

    def _init_model(self):
        if self.model_zoo_cfg is not None:
            model_config = self.model_zoo_cfg[self.model_name]
            self.backbone = model_config["model"]()
            self._inference_fn = model_config["inference"]
            self._reset_fn = model_config.get("reset")
        else:
            if self.model_name in self.default_resnet_models:
                self.backbone, self._inference_fn = self._prepare_resnet_backbone()

        self.backbone = self.backbone.to(self.model_device)
        # 冻结主干权重
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone.eval()

    def _prepare_resnet_backbone(self) -> tuple[nn.Module, Callable]:
        # 预训练权重映射
        resnet_weights = {
            "resnet18": ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": ResNet50_Weights.IMAGENET1K_V1,
            "resnet101": ResNet101_Weights.IMAGENET1K_V1,
        }
        full_model = getattr(models, self.model_name)(weights=resnet_weights[self.model_name])
        # 移除最终fc层，保留到全局平均池化层（AdaptiveAvgPool2d）
        backbone = nn.Sequential(*list(full_model.children())[:-1])

        # 推理
        def _resnet_inference(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
            """
            ResNet推理：输入图像 → 中间特征
            Args:
                model: ResNet主干（无fc层）
                images: 原始图像张量，shape=(num_envs, H, W, 3)
            Returns:
                全局池化后的特征，shape=(num_envs, backbone_dim)
            """
            # 图像预处理：(num_envs, H, W, 3) → (num_envs, 3, H, W) + 归一化
            image_proc = images.to(self.model_device)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0 
            # ImageNet标准化
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # 前向传播：得到 (num_envs, backbone_dim, 1, 1)
            features = model(image_proc)
            # 展平为 (num_envs, backbone_dim)
            return features.flatten(1)

        return backbone, _resnet_inference

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if self._reset_fn is not None:
            self._reset_fn(self.backbone, env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        inference_kwargs: Optional[Dict] = None,
    ) -> torch.Tensor:
        # 1. 获取原始图像
        image_data = image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=False,
        )
        image_device = image_data.device

        # 2. 预训练主干提取特征
        with torch.no_grad():
            backbone_features = self._inference_fn(self.backbone, image_data, **(inference_kwargs or {}))

        # 4. 转回图像设备并返回
        return backbone_features.to(image_device)
