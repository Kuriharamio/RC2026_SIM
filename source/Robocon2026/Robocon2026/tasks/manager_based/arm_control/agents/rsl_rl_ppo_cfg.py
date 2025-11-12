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
class ArmControlRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    print_green("ArmControlRslRlPpoActorCriticCfg")
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
class ArmControlRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    print_green("ArmControlRslRlPpoAlgorithmCfg")
    pass


@configclass
class ArmControlRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    print_green("ArmControlRslRlOnPolicyRunnerCfg")
    policy: ArmControlRslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: ArmControlRslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""
    pass


@configclass
class ArmReachPPORunnerCfg(ArmControlRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "arm_control_reach"
    obs_groups = {"policy": ["policy"]}
    policy = ArmControlRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
        enable_resnet_encoder=False,
        img_obs_keys=["rgb"],
        img_bottleneck_dim=256,
        state_obs_keys=["state"],
        state_latent_dim=256,
    )
    algorithm = ArmControlRslRlPpoAlgorithmCfg(
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


@configclass
class ArmLiftPPORunnerCfg(ArmReachPPORunnerCfg):

    obs_groups = {"policy": ["policy", "privilege"]}
    policy = ArmControlRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        enable_resnet_encoder=False,
        # img_obs_keys=["jaw_camera"],
        # img_bottleneck_dim=512,
        # state_obs_keys=["policy"],
        # state_latent_dim=64,
    )

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "arm_control_lift"


#########################
# Student Distillation ##
#########################


@configclass
class ArmLiftDistillationRunnerCfg(ArmReachPPORunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    class_name = "DistillationRunner"
    run_name = "distillation"

    obs_groups = {
        "policy": ["policy", "privilege", "jaw_camera_feature", "scene_camera_feature"],
        "teacher": ["policy", "privilege"],
        "critic": ["policy", "privilege", "jaw_camera_feature", "scene_camera_feature"],
    }
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[256, 128, 64],
        teacher_obs_normalization=True,
        student_obs_normalization=True,
        activation="elu",
        init_noise_std=0.1,
        class_name="StudentTeacherRecurrent",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=3,
        teacher_recurrent=False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1e-3,
        gradient_length=5*(2048 / 16),
        optimizer="adam",
        loss_type="mse",
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arm_control_lift"
        self.max_iterations = 15000


#########################
# Student Fine Tuning ###
#########################


@configclass
class ArmLiftFinetunePPORunnerCfg(ArmLiftPPORunnerCfg):
    obs_groups = {"policy": ["distillation"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 15000
        self.experiment_name = "arm_control_lift"
        self.run_name = "student_finetune"
