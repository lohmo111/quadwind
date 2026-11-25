"""Numerical contracts for the recurrent PPO code used in the defense notebook."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quadwind.native_policy import NativeActorCritic, ppo_loss
from quadwind.native_rollout import RecurrentBatch, RolloutCollector, compute_gae


def small_policy():
    torch.manual_seed(23)
    return NativeActorCritic(3, 2, 1, hidden_size=5, actor_layers=(), critic_layers=())


def test_gae_bootstraps_timeouts_and_rollout_end_but_stops_both_episode_traces():
    rewards = torch.tensor([[1., 1.], [2., 2.], [8., 8.]])
    values = torch.tensor([[2., 2.], [3., 3.], [4., 4.]])
    next_values = torch.tensor([[3., 3.], [50., 8.], [7., 7.]])
    terminated = torch.tensor([[False, False], [True, False], [False, False]])
    truncated = torch.tensor([[False, False], [False, True], [False, False]])

    advantages, targets = compute_gae(rewards, values, next_values, terminated, truncated,
                                      gamma=0.5, gae_lambda=0.5)
    torch.testing.assert_close(advantages, torch.tensor([[.25, 1.25], [-1., 3.], [7.5, 7.5]]))
    torch.testing.assert_close(targets, torch.tensor([[2.25, 3.25], [2., 6.], [11.5, 11.5]]))


def test_actor_ignores_privileged_input_and_only_reset_stream_loses_memory():
    policy = small_policy()
    obs = torch.randn(4, 2, 3)
    initial = policy.initial_state(2)
    starts = torch.zeros(4, 2, dtype=torch.bool)
    starts[2, 0] = True

    dist, values, _ = policy(obs, initial, starts)
    changed = obs.clone()
    changed[:, :, 2] += 10
    dist_changed, values_changed, _ = policy(changed, initial, starts)
    torch.testing.assert_close(dist.mean, dist_changed.mean)
    assert not torch.allclose(values, values_changed)

    reset_dist, reset_values, _ = policy(obs[2:, :1], policy.initial_state(1), starts[2:, :1])
    torch.testing.assert_close(dist.mean[2:, :1], reset_dist.mean)
    torch.testing.assert_close(values[2:, :1], reset_values)
    other_dist, other_values, _ = policy(obs[:, 1:], policy.initial_state(1), starts[:, 1:])
    torch.testing.assert_close(dist.mean[:, 1:], other_dist.mean)
    torch.testing.assert_close(values[:, 1:], other_values)


def test_clipped_objective_both_advantage_signs_and_padding():
    policy = small_policy()
    obs = torch.randn(7, 1, 3)
    state = policy.initial_state(1)
    starts = torch.zeros(7, 1, dtype=torch.bool)
    valid = torch.tensor([True] * 6 + [False])[:, None]
    with torch.no_grad():
        dist, values, _ = policy(obs, state, starts)
        actions = dist.mean
        ratios = torch.tensor([1.5, .5, .5, 1.5, .8, 1.2, 9.])[:, None]
        old_log_prob = dist.log_prob(actions) - ratios.log()
        returns = values.clone()
        returns[-1] += 1000
    advantages = torch.tensor([1., -1., 1., -1., 0., 0., 1000.])[:, None]
    batch = RecurrentBatch(obs=obs, actions=actions, old_log_prob=old_log_prob,
                           advantages=advantages, returns=returns, episode_starts=starts,
                           mask=valid, initial_state=state)
    loss = ppo_loss(policy, batch, normalize_advantage=False)

    torch.testing.assert_close(loss.surrogate[valid], torch.tensor([1.2, -.8, .5, -1.5, 0., 0.]))
    # The two saturated terms have no policy gradient. The two unfavorable
    # excursions still contribute, even though all four ratios are outside the interval.
    slopes, = torch.autograd.grad(loss.surrogate.sum(), loss.ratio)
    torch.testing.assert_close(slopes[valid], torch.tensor([0., 0., 1., -1., 0., 0.]))
    assert loss.clip_fraction.item() == pytest.approx(4 / 6)
    assert loss.policy.item() == pytest.approx(.1)
    assert loss.value.item() == pytest.approx(0.)
    assert loss.total.item() == pytest.approx(.1)

    normalized = ppo_loss(policy, batch).total
    with torch.no_grad():
        old_log_prob[-1] += 5
        advantages[-1] *= -2
        returns[-1] += 500
    torch.testing.assert_close(ppo_loss(policy, batch).total, normalized)


class TwoStreamEnv:
    """Both streams auto-reset on the second step; only the second is a timeout."""

    num_envs = 2
    action_space = SimpleNamespace(low=np.array([-.1]), high=np.array([.1]))

    def obs(self):
        return np.array([[self.t, 0, .3], [self.t, 1, -.4]], dtype=np.float32)

    def reset(self):
        self.t = 0
        return self.obs()

    def step(self, controls):
        assert np.abs(controls).max() <= .1
        self.t += 1
        ends = np.array([self.t == 2] * self.num_envs)
        infos = [{}, {}]
        if ends[1]:
            # A crash on the time-limit step takes precedence over a timeout.
            infos[0] = {"TimeLimit.truncated": True, "crashed": True}
            infos[1] = {"TimeLimit.truncated": True, "terminal_observation": self.final_obs}
        return self.obs(), np.ones(2, dtype=np.float32), ends, infos

    final_obs = np.array([-5., 1., 4.], dtype=np.float32)


def test_collection_replays_old_log_probs_and_bootstraps_final_observation_with_memory():
    policy = small_policy()
    env = TwoStreamEnv()
    with torch.no_grad():
        policy.actor_head.bias.fill_(2.)
        policy.log_std.fill_(-5.)
    collector = RolloutCollector(env, policy)
    tape = collector.collect(5, .99, .95)
    assert (tape.actions > .1).all()  # rollout stores unbounded Gaussian samples
    torch.testing.assert_close(tape.episode_starts[:, 0],
                               torch.tensor([True, False, True, False, False]))
    assert tape.terminated[1, 0] and tape.truncated[1, 1]
    assert tape.next_values[1, 0].item() == 0
    with torch.no_grad():
        # Reconstruct timeout memory directly from the two actions' observations.
        _, _, state = policy(tape.obs[:2, 1:], policy.initial_state(1), tape.episode_starts[:2, 1:])
        terminal = torch.from_numpy(env.final_obs).reshape(1, 1, 3)
        _, terminal_value, _ = policy(terminal, state, torch.zeros(1, 1, dtype=torch.bool))
    torch.testing.assert_close(tape.next_values[1, 1], terminal_value.squeeze())
    assert not torch.isclose(tape.next_values[1, 1], tape.values[2, 1])

    seen = []
    padded = False
    for batch in tape.minibatches(batch_size=10, sequence_length=2, rng=np.random.default_rng(2)):
        dist, values, _ = policy(batch.obs, batch.initial_state, batch.episode_starts)
        torch.testing.assert_close(dist.log_prob(batch.actions)[batch.mask],
                                   batch.old_log_prob[batch.mask], atol=2e-4, rtol=1e-5)
        padded |= not batch.mask.all().item()
        for col in range(batch.mask.shape[1]):
            seq = batch.obs[batch.mask[:, col], col]
            assert len(seq) == 1 or torch.all(seq[1:, 0] == seq[:-1, 0] + 1)
            times, env_ids = seq[:, :2].long().T
            torch.testing.assert_close(values[batch.mask[:, col], col], tape.values[times, env_ids])
            seen.extend(tuple(row) for row in seq[:, :2].tolist())
    assert padded
    assert sorted(seen) == sorted((t, env_id) for t in range(5) for env_id in range(2))

    continuation = collector.collect(1, .99, .95)
    assert not continuation.episode_starts[0].any()
    torch.testing.assert_close(tape.next_values[-1], continuation.values[0])
