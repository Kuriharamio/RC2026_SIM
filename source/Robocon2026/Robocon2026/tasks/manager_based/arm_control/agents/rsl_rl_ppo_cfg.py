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
class ArmReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "arm_control_reach"
    obs_groups = {"policy": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ArmLiftPPORunnerCfg(ArmReachPPORunnerCfg):

    obs_groups = {"policy": ["policy"]}

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "arm_control_lift"
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [256, 128, 64]


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
        "policy": ["distillation"],
        "teacher": ["policy"],
        "critic": ["policy"],
    }
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        student_hidden_dims=[256, 128, 64],
        teacher_hidden_dims=[256, 128, 64],
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
        learning_rate=3e-4,
        gradient_length=5,
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
