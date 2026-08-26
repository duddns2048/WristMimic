# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0

class InterMimicBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)


            return

        def forward(self, obs_dict):
            obs = obs_dict['obs'] # 3198-dim observation
            weighting_obs = obs_dict.get('weighting_obs', None)
            states = obs_dict.get('rnn_states', None)

            # Compute object weights from weighting observation
            if weighting_obs is not None:
                obj_weights = self.eval_weighting_network(weighting_obs)
            else:
                obj_weights = self.eval_weighting_network(0)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output

        def eval_actor(self, obs):

            # obs: 3198-dim
            # self.actor_cnn: None (empty nn.Sequential)
            # obs1 = self.net1(obs)
            a_out = self.actor_cnn(obs)
            # None
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            # import pdb; pdb.set_trace()
            # 3-Layer MLP (with 3 activations)
            a_out = self.actor_mlp(a_out) 
            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                # self.mu_act: Identity
                # self.mu: 1-layer MLP (no activation)
                mu = self.mu_act(self.mu(a_out))
                # self.mu_expander: 1-layer mlp (with 1 activation, pre-activation)
                # mu = self.mu_expander(mu)
                if self.space_config['fixed_sigma']:
                    # self.sigma_act: Identity
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return

        def eval_critic(self, obs):
            # obs2 = self.net2(obs)
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))

            return value

        def eval_weighting_network(self, weighting_obs):
            """
            Compute object weights from weighting observation
            Args:
                weighting_obs: (batch, weighting_obs_size) or scalar 0 (fallback)
            Returns:
                obj_weights: (batch, num_objects) weights for each object
            """
            # TODO: Implement weighting network
            # For now, return None or uniform weights
            if isinstance(weighting_obs, int) and weighting_obs == 0:
                return None

            # TODO: Add actual network architecture here
            # Example:
            # weights = self.weighting_mlp(weighting_obs)
            # weights = torch.softmax(weights, dim=-1)
            # return weights

            return None

    def build(self, name, **kwargs):
        # Auto-adjust params based on policy type
        import copy
        params = copy.deepcopy(self.params)

        # Check if using VAE latent actions
        z_type = kwargs.get('z_type', None)

        if z_type == 'vae':
            # VAE latent action policy parameters
            # Network returns logstd, Model applies exp(logstd) = sigma
            # For sigma=0.36: need logstd = log(0.36) ≈ -1.02
            sigma_val = -1
            print(f"[InterMimicBuilder] Using VAE parameters: logstd={sigma_val:.2f} (sigma={0.36:.2f}), activation=silu")
            params['space']['continuous']['sigma_init']['val'] = sigma_val
            params['mlp']['activation'] = 'silu'
        else:
            # Regular MLP policy parameters (keep YAML defaults)
            print(f"[InterMimicBuilder] Using MLP parameters: sigma_init={params['space']['continuous']['sigma_init']['val']}, "
                  f"activation={params['mlp']['activation']}")

        net = InterMimicBuilder.Network(params, **kwargs)
        return net
