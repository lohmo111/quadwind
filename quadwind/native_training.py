"""Train, validate and reload the project's PyTorch implementation of recurrent PPO."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .conditions import make_suite
from .env import EnvConfig, PRIV_DIM
from .native_policy import NativeActorCritic, optimizer_step, ppo_loss
from .native_rollout import RolloutCollector
from .policies import ObsNormalizer
from .training import build_space, validation_metrics


class NativeModel:
    """Policy plus the inference/checkpoint interface shared by the project scripts."""

    def __init__(self, network_kwargs: dict, device: str):
        self.network_kwargs = network_kwargs
        self.device = torch.device(device)
        self.policy = NativeActorCritic(**network_kwargs).to(self.device)
        self.num_timesteps = 0

    @classmethod
    def from_config(cls, cfg: dict, env, seed: int, device: str):
        if cfg["algo"] != "rppo":
            raise ValueError("The native trainer implements recurrent PPO")
        torch.manual_seed(seed)
        pk = cfg["policy_kwargs"]
        if pk.get("shared_lstm") or not pk.get("enable_critic_lstm", True):
            raise ValueError("The native policy uses separate actor and critic LSTMs")
        obs_dim = env.observation_space.shape[0]
        actor_dim = obs_dim
        if cfg.get("asymmetric_critic"):
            if not EnvConfig.from_dict(cfg.get("env")).privileged:
                raise ValueError("An asymmetric critic requires env.privileged: true")
            actor_dim -= PRIV_DIM
        return cls(dict(obs_dim=obs_dim, actor_dim=actor_dim, action_dim=env.action_space.shape[0],
                        hidden_size=pk["lstm_hidden_size"], n_layers=pk["n_lstm_layers"],
                        actor_layers=pk["net_arch"]["pi"], critic_layers=pk["net_arch"]["vf"],
                        activation=pk["activation_fn"], log_std_init=pk["log_std_init"]), device)

    @torch.no_grad()
    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        inputs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        single = inputs.ndim == 1
        inputs = inputs.reshape(1, -1, self.policy.obs_dim)
        batch_size = inputs.shape[1]
        if state is None:
            state = self.policy.initial_state(batch_size)[:2]
        else:
            state = tuple(torch.as_tensor(x, device=self.device) for x in state)
        starts = torch.zeros((1, batch_size), device=self.device, dtype=torch.bool)
        if episode_start is not None:
            starts = torch.as_tensor(episode_start, device=self.device, dtype=torch.bool).reshape(1, batch_size)
        dist, next_state = self.policy.actor_distribution(inputs, state, starts)
        actions = dist.mean if deterministic else dist.sample()
        # QuadHover actions are bounded; PPO trains on the original Gaussian samples.
        actions = actions[0].clamp(-1.0, 1.0).cpu().numpy()
        return (actions[0] if single else actions), tuple(x.cpu().numpy() for x in next_state)

    def save(self, path):
        torch.save({"format": "quadwind-native-rppo-v1", "network": self.network_kwargs,
                    "policy": self.policy.state_dict(), "timesteps": self.num_timesteps}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("format") != "quadwind-native-rppo-v1":
            raise ValueError(f"Unrecognized native checkpoint format at {path}")
        model = cls(checkpoint["network"], device)
        model.policy.load_state_dict(checkpoint["policy"])
        model.policy.eval()
        model.num_timesteps = checkpoint["timesteps"]
        return model


def _checkpoint(model, env, folder, name):
    model.save(folder / f"{name}_model.zip")
    if env.norm_obs:
        ObsNormalizer.from_vecnormalize(env).save(folder / f"{name}_obs_norm.npz")


def train_native(model: NativeModel, env, cfg: dict, run_dir: Path, seed: int) -> float:
    """Run on-policy updates, then select the best policy on frozen validation episodes."""
    hp = cfg["algo_kwargs"]
    n_envs, total = env.num_envs, int(cfg["total_timesteps"])
    horizon, batch_size, epochs = hp["n_steps"], hp["batch_size"], hp["n_epochs"]
    sequence_length = hp.get("sequence_length", min(horizon, batch_size))
    freq = cfg["eval"]["freq"]
    if min(total, horizon, batch_size, epochs, sequence_length, freq) <= 0:
        raise ValueError("Training budget, rollout length, batch size, sequence length, epochs and eval freq must be positive")
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=hp["learning_rate"], eps=1e-5)
    rng = np.random.default_rng(seed)
    collector = RolloutCollector(env, model.policy)
    _checkpoint(model, env, run_dir, "initial")

    env_cfg = EnvConfig.from_dict(cfg.get("env"))
    conditions = make_suite(cfg["eval"]["suite"], cfg["eval"].get("n"), space=build_space(cfg))
    history = []
    next_eval, best = freq, float("inf")
    start = cfg["curriculum"]["start"]
    ramp_steps = max(1.0, total * cfg["curriculum"]["ramp_fraction"])
    metric_names = ("policy_loss", "value_loss", "approx_kl", "clip_fraction", "gradient_norm")

    with (run_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(("timesteps", "learning_rate", "difficulty", *metric_names))
        while model.num_timesteps < total:
            before = model.num_timesteps
            difficulty = min(1.0, start + (1 - start) * before / ramp_steps)
            env.set_attr("difficulty", difficulty)
            # Start the first update at the configured LR even on a one-rollout run.
            lr = hp["learning_rate"]
            if cfg.get("lr_schedule") == "linear":
                lr *= 1 - before / total
            for group in optimizer.param_groups:
                group["lr"] = lr

            steps = min(horizon, (total - before + n_envs - 1) // n_envs)
            rollout = collector.collect(steps, hp["gamma"], hp["gae_lambda"])
            model.num_timesteps += n_envs * steps
            metric_sums = np.zeros(len(metric_names))
            count = 0
            for _ in range(epochs):
                for batch in rollout.minibatches(batch_size, sequence_length, rng):
                    loss = ppo_loss(model.policy, batch, clip_range=hp["clip_range"],
                                    vf_coef=hp["vf_coef"], ent_coef=hp["ent_coef"],
                                    normalize_advantage=hp.get("normalize_advantage", True))
                    grad_norm = optimizer_step(model.policy, optimizer, loss, hp["max_grad_norm"])
                    n = int(batch.mask.sum().item())
                    metric_sums += n * np.array([loss.policy.item(), loss.value.item(), loss.approx_kl.item(),
                                                 loss.clip_fraction.item(), grad_norm.item()])
                    count += n
            writer.writerow((model.num_timesteps, lr, difficulty, *(metric_sums / count)))
            log.flush()
            if before == 0:
                _checkpoint(model, env, run_dir, "early")
            if model.num_timesteps >= next_eval or model.num_timesteps >= total:
                normalizer = ObsNormalizer.from_vecnormalize(env) if env.norm_obs else None
                row = validation_metrics(model, normalizer, env_cfg, conditions, model.num_timesteps)
                history.append(row)
                pd.DataFrame(history).to_csv(run_dir / "val_history.csv", index=False)
                if row["rmse_capped"] < best:
                    best = row["rmse_capped"]
                    _checkpoint(model, env, run_dir, "best")
                next_eval = (model.num_timesteps // freq + 1) * freq
    return best
