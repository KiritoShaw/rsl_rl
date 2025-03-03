# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class ActorCriticEncoder(nn.Module):
    is_recurrent = False
    def __init__(self,  env_cfg,
                        num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        is_teacher=True,
                        mlp_input_dim=None,
                        mlp_output_dim=None,
                        mlp_hidden_dims=[256, 256, 256],
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticEncoder, self).__init__()

        activation = get_activation(activation)  # 选择配置文件所指定的 activation function

        self.is_teacher = is_teacher
        self.num_base_obs = num_base_obs = env_cfg["env"]["num_base_obs"]
        self.num_height_obs = num_height_obs = env_cfg["env"]["num_height_obs"]
        self.num_extrinsic_obs = num_extrinsic_obs = env_cfg["env"]["num_extrinsic_obs"]
        
        self.get_base_obs = lambda obs: obs[:, :num_base_obs]
        self.get_height_obs = lambda obs: obs[:, num_base_obs:num_base_obs+num_height_obs]
        self.get_extrinsic_obs = lambda obs: obs[:, num_base_obs+num_height_obs:num_base_obs+num_height_obs+num_extrinsic_obs]

        # %%%%%% Encoder network: MLP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        encoder_layers = []
        encoder_layers.append(nn.Linear(mlp_input_dim, mlp_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(mlp_hidden_dims)):
            if l == len(mlp_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(mlp_hidden_dims[l], mlp_output_dim))
            else:
                encoder_layers.append(nn.Linear(mlp_hidden_dims[l], mlp_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)
        
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        mlp_input_dim_a = env_cfg["env"]["num_base_obs"] + env_cfg["env"]["num_latent"]
        mlp_input_dim_c = num_critic_obs

        # %%%%%% Policy network: MLP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))  # observation 的最后一维与 mlp_input_dim_a 相同即可
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        # 将 actor_layers 中的层按照顺序添加到 self.actor 中，构建了整个 Actor 网络的前向传播过程
        self.actor = nn.Sequential(*actor_layers)

        # %%%%%% Value function: MLP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Encoder MLP: {self.encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


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
        """ 计算熵 """
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """ 获取策略网络的输出动作的分布 """
        mean = self.actor(observations)  # 对 Actor 网络进行前向传播
        self.distribution = Normal(mean, mean*0. + self.std)

    def act_teacher(self, observations, **kwargs):
        """ 执行 teacher 网络输出的动作 """
        base_obs = self.get_base_obs(observations)
        height_obs = self.get_height_obs(observations)
        extrinsic_obs = self.get_extrinsic_obs(observations)
        latent = self.encoder(torch.cat((height_obs, extrinsic_obs), dim=-1))
        actor_input_obs = torch.cat((base_obs, latent), dim=-1)
        self.update_distribution(actor_input_obs)
        return self.distribution.sample()

    def act_student(self, observations, **kwargs):
        """ 执行 student 网络输出的动作 """
        base_obs = observations[:, -self.num_base_obs:]
        latent = self.encoder(observations)
        actor_input_obs = torch.cat((base_obs, latent), dim=-1)
        return self.actor(actor_input_obs), latent

    def act(self, observations, **kwargs):
        """ 从更新后的策略分布中采样动作得到具体的动作 """
        if self.is_teacher:
            return self.act_teacher(observations, **kwargs)
        else:
            return self.act_student(observations, **kwargs)
    
    def get_actions_log_prob(self, actions):
        """ 计算给定动作 actions 的对数概率 """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        base_obs = self.get_base_obs(observations)
        height_obs = self.get_height_obs(observations)
        extrinsic_obs = self.get_extrinsic_obs(observations)
        latent = self.encoder(torch.cat((height_obs, extrinsic_obs), dim=-1))
        actor_input_obs = torch.cat((base_obs, latent), dim=-1)
        actions_mean = self.actor(actor_input_obs)
        return actions_mean, latent

    def evaluate(self, critic_observations, **kwargs):
        """ 计算状态值函数 """
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
