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

@configclass
class RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 50
    experiment_name = "armdog_basic_control_rough"
    obs_groups = {"policy": ["policy"], "critic": ["privilege"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
class FlatPPORunnerCfg(RoughPPORunnerCfg):

    obs_groups = {"policy": ["policy"], "critic": ["privilege"]}
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "armdog_basic_control_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]

#########################
# Student Distillation ##
#########################


@configclass
class RoughDistillationRunnerCfg(RoughPPORunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    class_name = "DistillationRunner"
    run_name = "distillation"

    obs_groups = {"policy": ["policy"], "teacher": ["privilege"], "critic": ["privilege"]}
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        teacher_obs_normalization=False,
        student_obs_normalization=False,
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
        gradient_length=5,
        max_grad_norm=0.5,
        optimizer="adam",
        loss_type="mse",
    )

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 15000

#########################
# Student Fine Tuning ###
#########################


@configclass
class RoughStudentPPORunnerCfg(RoughPPORunnerCfg):
    obs_groups = {"policy": ["policy"], "critic": ["privilege"]}
    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=0.1,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
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
        self.run_name = "student_finetune"
