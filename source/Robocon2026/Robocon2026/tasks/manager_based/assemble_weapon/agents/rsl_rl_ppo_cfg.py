# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)

from dataclasses import MISSING
from typing import Literal, List
from isaaclab.utils import configclass

from Robocon2026.utils.utils import print_green, print_red, print_yellow


# self.img_obs_keys = kwargs.get("img_obs_keys", ["image", ])
# self.state_obs_keys = kwargs.get("state_obs_keys", ["state", ])
# self.num_spatial_blocks = kwargs.get("num_spatial_blocks", 8)
# self.img_bottleneck_dim = kwargs.get("img_bottleneck_dim", 256)
# self.state_latent_dim = kwargs.get("state_latent_dim", 256)

@configclass
class AssembleWeaponRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    print_green("AssembleWeaponRslRlPpoActorCriticCfg")
    img_obs_keys: List[str] = MISSING
    ''' Keys of the image observations '''

    state_obs_keys: List[str] = MISSING
    ''' Keys of the state observations '''

    img_bottleneck_dim: int = MISSING
    """ The dimension of the bottleneck layer for the image observations. """

    state_latent_dim: int = MISSING
    """ The dimension of the latent state representation. """

    enable_resnet_encoder: bool = MISSING
    """ Whether to use a pre-trained ResNet encoder for the image observations. """


@configclass
class AssembleWeaponRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    print_green("AssembleWeaponRslRlPpoAlgorithmCfg")
    pass


@configclass
class AssembleWeaponRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    print_green("AssembleWeaponRslRlOnPolicyRunnerCfg")
    policy: AssembleWeaponRslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: AssembleWeaponRslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""
    pass


@configclass
class AssembleWeaponPPORunnerCfg(AssembleWeaponRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 50
    experiment_name = "assemble_weapon"
    obs_groups = {"policy": ["policy", "privilege"]}
    policy = AssembleWeaponRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        enable_resnet_encoder=False,
        # img_obs_keys=["rgb"],
        # img_bottleneck_dim=256,
        # state_obs_keys=["state"],
        # state_latent_dim=256,
    )
    algorithm = AssembleWeaponRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

class AssembleWeaponDualPPORunnerCfg(AssembleWeaponPPORunnerCfg):

    save_interval = 100
    obs_groups = {"policy": ["policy", "privilege"]}
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "assemble_weapon_dual"
        self.max_iterations = 30000


class AssembleWeaponSinglePPORunnerCfg(AssembleWeaponPPORunnerCfg):

    obs_groups = {"policy": ["policy", "privilege"]}

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "assemble_weapon_single"
        self.max_iterations = 30000
