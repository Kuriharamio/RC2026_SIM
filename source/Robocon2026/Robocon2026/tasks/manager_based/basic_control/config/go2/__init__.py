# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from Robocon2026.tasks.manager_based.basic_control import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Basic-Control-Flat-GO2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_ppo_env_cfg:FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Basic-Control-Flat-Play-GO2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_ppo_env_cfg:FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Basic-Control-Rough-GO2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_ppo_env_cfg:RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Template-Basic-Control-Rough-Play-GO2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_ppo_env_cfg:RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Template-Basic-Control-Rough-Distillation-GO2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_distillation_env_cfg:RoughDistillationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughDistillationRunnerCfg",
    },
)

gym.register(
    id="Template-Basic-Control-Rough-Distillation-Play-GO2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_distillation_env_cfg:RoughDistillationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughDistillationRunnerCfg",
    },
)

gym.register(
    id="Template-Basic-Control-Rough-Finetune-GO2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_fintune_env_cfg:RoughFintuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughStudentPPORunnerCfg",
    },
)

gym.register(
    id="Template-Basic-Control-Rough-Finetune-Play-GO2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_fintune_env_cfg:RoughFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughStudentPPORunnerCfg",
    },
)
