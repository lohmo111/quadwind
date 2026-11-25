"""Recurrent actor-critic and PPO update for continuous actions, implemented in PyTorch."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal

from .native_rollout import RecurrentBatch


LstmState = tuple[Tensor, Tensor]
PolicyState = tuple[Tensor, Tensor, Tensor, Tensor]


def _mlp(input_size: int, widths: Sequence[int], activation: type[nn.Module]) -> nn.Sequential:
    layers = []
    for width in widths:
        linear = nn.Linear(input_size, width)
        nn.init.orthogonal_(linear.weight, gain=sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, activation()))
        input_size = width
    return nn.Sequential(*layers)


def _run_lstm(lstm: nn.LSTM, inputs: Tensor, state: LstmState,
              episode_starts: Tensor) -> tuple[Tensor, LstmState]:
    """Run contiguous time spans together. Reset only the envs starting a new episode."""
    # Start a new LSTM call wherever at least one environment resets mid-sequence.
    reset_times = episode_starts[1:].any(dim=1).nonzero(as_tuple=True)[0] + 1
    boundaries = [0, *reset_times.tolist(), len(inputs)]
    pieces = []
    for start, end in zip(boundaries, boundaries[1:]):
        keep = (~episode_starts[start]).view(1, -1, 1)
        h, c = state
        output, state = lstm(inputs[start:end], (h * keep, c * keep))
        pieces.append(output)
    outputs = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
    return outputs, state


class NativeActorCritic(nn.Module):
    """Two independent LSTMs. Only the critic can see privileged observation columns."""

    def __init__(self, obs_dim: int, actor_dim: int, action_dim: int, hidden_size: int = 128,
                 n_layers: int = 1, actor_layers: Sequence[int] = (128, 128),
                 critic_layers: Sequence[int] = (128, 128), activation: str = "tanh",
                 log_std_init: float = -1.0):
        super().__init__()
        if not 0 < actor_dim <= obs_dim:
            raise ValueError("actor_dim must be between 1 and obs_dim")
        self.obs_dim, self.actor_dim, self.action_dim = obs_dim, actor_dim, action_dim
        self.hidden_size, self.n_layers = hidden_size, n_layers
        activation_type = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]
        self.actor_lstm = nn.LSTM(actor_dim, hidden_size, n_layers)
        self.critic_lstm = nn.LSTM(obs_dim, hidden_size, n_layers)
        self.actor_mlp = _mlp(hidden_size, actor_layers, activation_type)
        self.critic_mlp = _mlp(hidden_size, critic_layers, activation_type)
        self.actor_head = nn.Linear(actor_layers[-1] if actor_layers else hidden_size, action_dim)
        self.critic_head = nn.Linear(critic_layers[-1] if critic_layers else hidden_size, 1)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.zeros_(self.critic_head.bias)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    @property
    def lstm_actor(self) -> nn.LSTM:
        """Actor recurrence exposed to the project's existing memory-analysis tools."""
        return self.actor_lstm

    def initial_state(self, batch_size: int) -> PolicyState:
        shape = (self.n_layers, batch_size, self.hidden_size)
        h = self.log_std.new_zeros(shape)
        return h, torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)

    def actor_distribution(self, obs: Tensor, state: LstmState,
                           episode_starts: Tensor) -> tuple[Independent, LstmState]:
        features, next_state = _run_lstm(self.actor_lstm, obs[..., :self.actor_dim], state,
                                         episode_starts.bool())
        means = self.actor_head(self.actor_mlp(features))
        # Each action dimension is sampled independently; the joint log-density sums over them.
        return Independent(Normal(means, self.log_std.exp()), 1), next_state

    def forward(self, obs: Tensor, state: PolicyState,
                episode_starts: Tensor) -> tuple[Independent, Tensor, PolicyState]:
        """Accept [time, batch, features] observations and [time, batch] reset flags."""
        distribution, actor_state = self.actor_distribution(obs, state[:2], episode_starts)
        critic, critic_state = _run_lstm(self.critic_lstm, obs, state[2:], episode_starts.bool())
        values = self.critic_head(self.critic_mlp(critic)).squeeze(-1)
        return distribution, values, (*actor_state, *critic_state)


@dataclass
class PPOLoss:
    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    advantages: Tensor
    ratio: Tensor
    surrogate: Tensor
    clip_fraction: Tensor
    approx_kl: Tensor


def ppo_loss(policy: NativeActorCritic, batch: RecurrentBatch, *, clip_range: float = 0.2, vf_coef: float = 0.5,
             ent_coef: float = 0.0, normalize_advantage: bool = True) -> PPOLoss:
    """Replay recorded actions; all reductions and advantage normalization ignore right padding.

    Scalar transition data and masks have shape [time, batch]. The rollout's old log-density,
    advantages, targets and initial recurrent state are fixed throughout every training epoch.
    """
    valid = batch.mask.bool()
    scalars = (batch.episode_starts, batch.old_log_prob, batch.advantages, batch.returns)
    if valid.shape != batch.obs.shape[:2] or any(t.shape != valid.shape for t in scalars):
        raise ValueError("Expected transition tensors of shape [time, batch]")
    if batch.actions.shape != (*valid.shape, policy.action_dim):
        raise ValueError("Expected actions of shape [time, batch, action_dim]")
    if not valid.any():
        raise ValueError("A PPO minibatch requires at least one valid transition")

    # A sequence is differentiated through its real timesteps, starting with frozen rollout memory.
    state = tuple(part.detach() for part in batch.initial_state)
    obs = batch.obs.masked_fill(~valid[..., None], 0)
    actions = batch.actions.detach().masked_fill(~valid[..., None], 0)
    dist, values, _ = policy(obs, state, batch.episode_starts)
    log_ratio = (dist.log_prob(actions) - batch.old_log_prob.detach()).masked_fill(~valid, 0)
    ratio = log_ratio.exp()

    advantage = batch.advantages.detach()
    observed = advantage[valid]
    if normalize_advantage and observed.numel() > 1:
        advantage = (advantage - observed.mean()) / (observed.std() + 1e-8)
    advantage = advantage.masked_fill(~valid, 0)
    clipped_ratio = ratio.clamp(1 - clip_range, 1 + clip_range)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    policy_loss = -surrogate[valid].mean()
    value_loss = (values[valid] - batch.returns.detach()[valid]).square().mean()
    entropy = dist.entropy()[valid].mean()
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy
    with torch.no_grad():
        # Count ratios outside the interval, including both advantage signs.
        clip_fraction = (ratio[valid] != clipped_ratio[valid]).float().mean()
        approx_kl = ((ratio - 1) - log_ratio)[valid].mean()
    return PPOLoss(total, policy_loss, value_loss, entropy, advantage, ratio, surrogate,
                   clip_fraction, approx_kl)


def optimizer_step(policy: NativeActorCritic, optimizer: torch.optim.Optimizer,
                   loss: PPOLoss, max_grad_norm: float = 0.5) -> Tensor:
    """Apply one minibatch update. Return the norm before clipping; keep gradients inspectable."""
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return norm
