"""Асимметричный актор-критик: критик видит привилегированные признаки (истинные ветер, массу, моторы),
актор — только обычное наблюдение. Pinto et al., 2018 (arXiv:1710.06542).

Привилегированные признаки среда дописывает в КОНЕЦ наблюдения (EnvConfig.privileged = True),
поэтому актору достаточно взять первые `actor_dim` чисел. На действия при оценке
привилегированная часть не влияет (тест tests/test_asymmetric.py).
"""
from __future__ import annotations

import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn

from .fast_lstm import FastLstmPolicy


class SlicedMlpExtractor(nn.Module):
    """Сеть актора получает features[:, :actor_dim], сеть критика — все признаки."""

    def __init__(self, feature_dim: int, actor_dim: int, net_arch, activation_fn, device):
        super().__init__()
        self.actor_dim = actor_dim
        pi = MlpExtractor(actor_dim, net_arch=net_arch, activation_fn=activation_fn, device=device)
        vf = MlpExtractor(feature_dim, net_arch=net_arch, activation_fn=activation_fn, device=device)
        self.policy_net = pi.policy_net
        self.value_net = vf.value_net
        self.latent_dim_pi = pi.latent_dim_pi
        self.latent_dim_vf = vf.latent_dim_vf

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features[..., : self.actor_dim])

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class AsymmetricMlpPolicy(ActorCriticPolicy):
    def __init__(self, *args, actor_dim: int, **kwargs):
        self.actor_dim = actor_dim
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SlicedMlpExtractor(self.features_dim, self.actor_dim, self.net_arch,
                                                self.activation_fn, self.device)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data["actor_dim"] = self.actor_dim
        return data


class AsymmetricLstmPolicy(FastLstmPolicy):
    """LSTM актора получает только первые actor_dim признаков (FastLstmPolicy режет по lstm.input_size)."""

    def __init__(self, observation_space, action_space, lr_schedule, *args, actor_dim: int, **kwargs):
        self.actor_dim = actor_dim
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        hidden = self.lstm_actor.hidden_size
        self.lstm_actor = nn.LSTM(actor_dim, hidden, num_layers=self.lstm_actor.num_layers, **self.lstm_kwargs)
        self.lstm_actor.to(self.device)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data["actor_dim"] = self.actor_dim
        return data
