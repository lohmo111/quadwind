"""On-policy trajectories, terminal-aware GAE and time-ordered PPO minibatches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


LSTMState = tuple[Tensor, Tensor, Tensor, Tensor]


@torch.no_grad()
def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Return advantages and critic targets. Every input is time × environments.

    next_values at time limits must be evaluated on the final observation of
    that episode. Both types of episode end stop the backwards recursion.
    """
    if rewards.ndim != 2 or rewards.shape[0] == 0 or any(
        x.shape != rewards.shape for x in (values, next_values, terminated, truncated)
    ):
        raise ValueError("GAE expects nonempty, equally shaped [time, environments] tensors")
    bootstrap = torch.where(terminated, 0.0, next_values)
    deltas = rewards + gamma * bootstrap - values
    continues = ~(terminated | truncated)
    advantages = torch.empty_like(deltas)
    carry = torch.zeros_like(deltas[0])
    for t in reversed(range(len(deltas))):
        carry = deltas[t] + gamma * gae_lambda * torch.where(continues[t], carry, 0.0)
        advantages[t] = carry
    return advantages, advantages + values


@dataclass
class RecurrentBatch:
    """A right-padded batch, with the pre-observation memory of each sequence."""

    obs: Tensor
    actions: Tensor
    old_log_prob: Tensor
    advantages: Tensor
    returns: Tensor
    episode_starts: Tensor
    mask: Tensor
    initial_state: LSTMState


@dataclass
class Rollout:
    obs: Tensor
    actions: Tensor
    old_log_prob: Tensor
    rewards: Tensor
    values: Tensor
    next_values: Tensor
    terminated: Tensor
    truncated: Tensor
    episode_starts: Tensor
    states: LSTMState  # each [time, layers, environments, hidden]
    advantages: Tensor
    returns: Tensor

    def minibatches(
        self, batch_size: int, sequence_length: int, rng: np.random.Generator
    ) -> Iterator[RecurrentBatch]:
        """Visit each transition once, shuffling sequences but preserving time within each."""
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("batch_size and sequence_length must be positive")
        max_length = min(sequence_length, batch_size)
        starts = self.episode_starts.cpu().numpy()
        n_steps, n_envs = starts.shape
        chunks = []  # (environment, first time, length)
        for env in range(n_envs):
            boundaries = [0, *(np.flatnonzero(starts[1:, env]) + 1), n_steps]
            for begin, end in zip(boundaries, boundaries[1:]):
                for first in range(begin, end, max_length):
                    chunks.append((env, first, min(end - first, max_length)))
        rng.shuffle(chunks)
        group = []
        longest = 0
        for chunk in chunks:
            length = chunk[2]
            if group and max(longest, length) * (len(group) + 1) > batch_size:
                yield self._batch(group)
                group = []
                longest = 0
            group.append(chunk)
            longest = max(longest, length)
        if group:
            yield self._batch(group)

    def _batch(self, chunks: list[tuple[int, int, int]]) -> RecurrentBatch:
        def padded(data: Tensor) -> Tensor:
            sequences = [data[first:first + length, env] for env, first, length in chunks]
            return pad_sequence(sequences)

        mask = pad_sequence([
            torch.ones(length, dtype=torch.bool, device=self.obs.device) for _, _, length in chunks
        ])
        # Each column starts with the memory recorded before its first observation.
        state = tuple(
            torch.stack([component[first, :, env] for env, first, _ in chunks], dim=1)
            for component in self.states
        )
        return RecurrentBatch(
            obs=padded(self.obs), actions=padded(self.actions),
            old_log_prob=padded(self.old_log_prob), advantages=padded(self.advantages),
            returns=padded(self.returns), episode_starts=padded(self.episode_starts),
            mask=mask, initial_state=state,
        )


class RolloutCollector:
    """Keep vector-environment observations and policy memory across PPO updates."""

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.state = policy.initial_state(env.num_envs)
        self.device = self.state[0].device
        self.obs = self._observation(env.reset())
        self.episode_starts = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)

    def _observation(self, array) -> Tensor:
        # Environments may reuse their own observation buffers at the next step.
        return torch.tensor(array, dtype=self.state[0].dtype, device=self.device)

    def _timeout_values(self, infos, rows: np.ndarray) -> Tensor:
        """Evaluate the last observation before each timed-out environment auto-reset."""
        final_obs = self._observation(np.stack([infos[i]["terminal_observation"] for i in rows]))
        final_state = tuple(component[:, rows] for component in self.state)
        same_episode = torch.zeros((1, len(rows)), dtype=torch.bool, device=self.device)
        _, value, _ = self.policy(final_obs[None], final_state, same_episode)
        return value[0]

    @torch.no_grad()
    def collect(self, n_steps: int, gamma: float, gae_lambda: float) -> Rollout:
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        observations, actions, log_probs, values = [], [], [], []
        rewards, starts, terminations, truncations = [], [], [], []
        memories = []
        time_limits = []
        for t in range(n_steps):
            observations.append(self.obs)
            starts.append(self.episode_starts)
            memories.append(self.state)
            dist, value, self.state = self.policy(
                self.obs[None], self.state, self.episode_starts[None]
            )
            sampled = dist.sample()
            actions.append(sampled[0])
            log_probs.append(dist.log_prob(sampled)[0])
            values.append(value[0])

            # The Gaussian log probability belongs to the raw action. Only the
            # control sent to the simulator is clipped to the actuator bounds.
            control = np.clip(sampled[0].cpu().numpy(), self.env.action_space.low,
                              self.env.action_space.high)
            obs, reward, done, infos = self.env.step(control)
            done = np.asarray(done, dtype=bool)
            timeout = np.array([
                ended and info.get("TimeLimit.truncated", False) and not info.get("crashed", False)
                for ended, info in zip(done, infos)
            ], dtype=bool)
            timeout_rows = np.flatnonzero(timeout)
            if len(timeout_rows):
                time_limits.append((t, timeout_rows, self._timeout_values(infos, timeout_rows)))
            rewards.append(torch.tensor(reward, device=self.device, dtype=value.dtype))
            terminations.append(torch.as_tensor(done & ~timeout, device=self.device))
            truncations.append(torch.as_tensor(timeout, device=self.device))
            self.obs = self._observation(obs)
            self.episode_starts = torch.tensor(done, device=self.device)

        # For continuing trajectories, the next value is already on the tape,
        # except at the rollout boundary. For ended episodes, never use a reset
        # observation as the final observation of the preceding episode.
        _, last_value, _ = self.policy(self.obs[None], self.state, self.episode_starts[None])
        value_tensor = torch.stack(values)
        next_values = torch.cat([value_tensor[1:], last_value], dim=0)
        terminated = torch.stack(terminations)
        next_values[terminated] = 0
        for time, rows, final_value in time_limits:
            next_values[time, rows] = final_value
        reward_tensor = torch.stack(rewards)
        truncated = torch.stack(truncations)
        advantages, returns = compute_gae(
            reward_tensor, value_tensor, next_values, terminated, truncated, gamma, gae_lambda
        )
        return Rollout(
            obs=torch.stack(observations), actions=torch.stack(actions), old_log_prob=torch.stack(log_probs),
            rewards=reward_tensor, values=value_tensor, next_values=next_values,
            terminated=terminated, truncated=truncated, episode_starts=torch.stack(starts),
            states=tuple(torch.stack(component) for component in zip(*memories)),
            advantages=advantages, returns=returns,
        )
