# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from re import L

from Robocon2026.utils.utils import print_yellow
import torch
import torch.nn as nn
from torch.distributions import Normal

from local_rsl_rl.networks import MLP, EmpiricalNormalization
from local_rsl_rl.networks.encodering import EncodingWrapper

from typing import List

from transformers import AutoModel


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.enable_resnet_encoder = kwargs.get("enable_resnet_encoder", False)
        self.img_obs_keys = kwargs.get("img_obs_keys", [])
        self.state_obs_keys = kwargs.get("state_obs_keys", [])
        self.img_bottleneck_dim = kwargs.get("img_bottleneck_dim", 256)
        self.state_latent_dim = kwargs.get("state_latent_dim", 64)

        print_yellow(f"{self.enable_resnet_encoder}")

        # get the observation dimensions
        self.obs_groups = obs_groups
        state_input_dim = 0

        num_actor_obs = 0

        actor_img_dim = 0
        actor_state_dim = 0
        actor_other_dim = 0
        for obs_group in obs_groups["policy"]:
            # assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            # num_actor_obs += obs[obs_group].shape[-1]
            if obs_group in self.img_obs_keys:
                num_actor_obs += self.img_bottleneck_dim
                actor_img_dim += self.img_bottleneck_dim
            elif obs_group in self.state_obs_keys:
                num_actor_obs += self.state_latent_dim
                actor_state_dim += self.state_latent_dim
                state_input_dim += obs[obs_group].shape[-1]
            else:
                num_actor_obs += obs[obs_group].shape[-1]
                actor_other_dim += obs[obs_group].shape[-1]

        num_critic_obs = 0

        critic_img_dim = 0
        critic_state_dim = 0
        critic_other_dim = 0
        for obs_group in obs_groups["critic"]:
            # assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            # num_critic_obs += obs[obs_group].shape[-1]
            if obs_group in self.img_obs_keys:
                num_critic_obs += self.img_bottleneck_dim
                critic_img_dim += self.img_bottleneck_dim
            elif obs_group in self.state_obs_keys:
                num_critic_obs += self.state_latent_dim
                critic_state_dim += self.state_latent_dim
            else:
                num_critic_obs += obs[obs_group].shape[-1]
                critic_other_dim += obs[obs_group].shape[-1]

        # actor
        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}\n")
        print(f"Actor Img Dims: {actor_img_dim}\n"
              f"Actor State Dims: {actor_state_dim}\n"
              f"Actor Other Dims: {actor_other_dim}\n")

        # critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}\n")
        print(f"Critic Img Dims: {critic_img_dim}\n"
              f"Critic State Dims: {critic_state_dim}\n"
              f"Critic Other Dims: {critic_other_dim}\n")

        if self.enable_resnet_encoder:
            # resnet_config = ResNetEncoder(
            #     stage_sizes=(1, 1, 1, 1),
            #     block_cls=ResNetBlock,
            #     pre_pooling=True
            # )

            resnet10 = AutoModel.from_pretrained(
                "helper2424/resnet10", trust_remote_code=True
            )

            # * 不同相机的数据用不同的编码器
            encoders = {}
            for image_obs_key in self.img_obs_keys:
                encoders[image_obs_key] = resnet10

            # * 共享编码器
            self.shared_encoder = EncodingWrapper(
                encoder=encoders,
                state_latent_dim=self.state_latent_dim,
                img_obs_keys=self.img_obs_keys,
                state_obs_keys=self.state_obs_keys,
                state_input_dim=state_input_dim,
            )
            print(f"Shared Encoder: {self.shared_encoder}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        # compute mean
        mean = self.actor(obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs, train=False)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs, train=self.training)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs, train: bool = True):
        # obs_list = []
        # for obs_group in self.obs_groups["policy"]:
        #     obs_list.append(obs[obs_group])
        # return torch.cat(obs_list, dim=-1)
        obs_dict = {}
        other_obs_list = []
        for obs_group in self.obs_groups["policy"]:
            if obs_group in self.img_obs_keys or obs_group in self.state_obs_keys:
                obs_dict[obs_group] = obs[obs_group]
            else:
                other_obs_list.append(obs[obs_group])

        if self.enable_resnet_encoder:
            encoded_obs = self.shared_encoder(obs_dict, train=train)

        if self.enable_resnet_encoder and len(other_obs_list) > 0:
            return torch.cat([encoded_obs, other_obs_list], dim=-1)
        elif self.enable_resnet_encoder and len(other_obs_list) == 0:
            return encoded_obs
        elif not self.enable_resnet_encoder and len(other_obs_list) > 0:
            return torch.cat(other_obs_list, dim=-1)

    def get_critic_obs(self, obs, train: bool = True):
        # obs_list = []
        # for obs_group in self.obs_groups["critic"]:
        #     obs_list.append(obs[obs_group])
        # return torch.cat(obs_list, dim=-1)
        obs_dict = {}
        other_obs_list = []
        for obs_group in self.obs_groups["critic"]:
            if obs_group in self.img_obs_keys or obs_group in self.state_obs_keys:
                obs_dict[obs_group] = obs[obs_group]
            else:
                other_obs_list.append(obs[obs_group])
        if self.enable_resnet_encoder:
            encoded_obs = self.shared_encoder(obs_dict, train=train)

        if self.enable_resnet_encoder and len(other_obs_list) > 0:
            return torch.cat([encoded_obs, other_obs_list], dim=-1)
        elif self.enable_resnet_encoder and len(other_obs_list) == 0:
            return encoded_obs
        elif not self.enable_resnet_encoder and len(other_obs_list) > 0:
            return torch.cat(other_obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs, train=False)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs, train=False)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes
