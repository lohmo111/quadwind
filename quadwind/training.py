"""Сборка среды и модели по конфигу, колбэки обучения (curriculum, валидация, логирование падений)."""
from __future__ import annotations

import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from .conditions import SCENARIO_SPACES, TRAIN_SPACE, ConditionSpace, make_suite, override_space
from .env import PRIV_DIM, EnvConfig, QuadHoverVecEnv
from .evaluation import rollout
from .metrics import episode_metrics
from .policies import ObsNormalizer, SB3Policy

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
TRAIN_SPACES = {"train": TRAIN_SPACE, "calm": SCENARIO_SPACES["calm"]}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overlays: tuple | list = ()) -> dict:
    """Конфиг метода = configs/base.yaml + файл метода (+ оверлеи вариантов среды, например configs/e7/*.yaml).

    Оверлей с ключом `variant` добавляет его к имени запуска: rppo__v2_persistent.
    """
    cfg = _load_one(path)
    for overlay in overlays:
        extra = yaml.safe_load(Path(overlay).read_text(encoding="utf-8")) or {}
        variant = extra.pop("variant", Path(overlay).stem)
        cfg = deep_merge(cfg, extra)
        cfg["name"] = f"{cfg['name']}__{variant}"
        cfg.setdefault("variants", []).append(variant)
    return cfg


def _load_one(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_name = cfg.pop("base", "base.yaml")
    base_path = path.parent / base_name if (path.parent / base_name).exists() else CONFIG_DIR / base_name
    if base_path.resolve() == path.resolve():
        return cfg
    return deep_merge(_load_one(base_path), cfg)


def build_space(cfg: dict) -> ConditionSpace:
    """Пространство условий обучения: train_space + space_overrides из конфига."""
    return override_space(TRAIN_SPACES[cfg.get("train_space", "train")], cfg.get("space_overrides"))


def make_env(cfg: dict, seed: int) -> VecNormalize:
    env_cfg = EnvConfig.from_dict(cfg.get("env"))
    venv = QuadHoverVecEnv(env_cfg, cfg["n_envs"], space=build_space(cfg),
                           seed=seed, difficulty=cfg["curriculum"]["start"])
    venv = VecMonitor(venv)
    norm = cfg["normalize"]
    return VecNormalize(venv, norm_obs=norm["obs"], norm_reward=norm["reward"], clip_obs=norm.get("clip_obs", 10.0),
                        gamma=cfg["algo_kwargs"].get("gamma", 0.99))


def _schedule(value: float, kind: str):
    if kind == "linear":
        # The final full rollout can extend beyond the requested step budget.
        return lambda progress_remaining: value * max(0.0, progress_remaining)
    return value


def make_model(cfg: dict, env, seed: int, device: str = "cpu", tensorboard_log: str | None = None):
    import torch.nn as nn

    kw = dict(cfg["algo_kwargs"])
    kw["learning_rate"] = _schedule(kw.pop("learning_rate"), cfg.get("lr_schedule", "constant"))
    pk = dict(cfg.get("policy_kwargs", {}))
    if "activation_fn" in pk:
        pk["activation_fn"] = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[pk["activation_fn"]]
    asym = bool(cfg.get("asymmetric_critic", False))
    if asym:                                            # критик видит признаки оракула, актор — нет
        env_cfg = EnvConfig.from_dict(cfg.get("env"))
        if not env_cfg.privileged:
            raise ValueError("asymmetric_critic требует env.privileged: true")
        pk["actor_dim"] = env_cfg.obs_dim - PRIV_DIM
    if cfg["algo"] == "ppo":
        pk.pop("lstm_hidden_size", None)
        pk.pop("n_lstm_layers", None)
        pk.pop("shared_lstm", None)
        pk.pop("enable_critic_lstm", None)
        from .asymmetric import AsymmetricMlpPolicy
        return PPO(AsymmetricMlpPolicy if asym else "MlpPolicy", env, policy_kwargs=pk, seed=seed, device=device,
                   tensorboard_log=tensorboard_log, verbose=0, **kw)
    if cfg["algo"] == "rppo":
        from sb3_contrib import RecurrentPPO

        from .asymmetric import AsymmetricLstmPolicy
        from .fast_lstm import FastLstmPolicy
        return RecurrentPPO(AsymmetricLstmPolicy if asym else FastLstmPolicy, env, policy_kwargs=pk, seed=seed,
                            device=device,
                            tensorboard_log=tensorboard_log, verbose=0, **kw)
    raise ValueError(f"unknown algo {cfg['algo']!r}")


def load_model(run_dir: Path, which: str = "best"):
    """Загружает модель и замороженную нормализацию из папки запуска."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    path = run_dir / f"{which}_model.zip"
    norm_path = run_dir / f"{which}_obs_norm.npz"
    if cfg.get("normalize", {}).get("obs") and not norm_path.is_file():
        raise FileNotFoundError(f"Checkpoint requires its matching observation statistics: {norm_path}")
    if cfg["algo"] == "rppo":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(path, device="cpu")
    else:
        model = PPO.load(path, device="cpu")
    normalizer = ObsNormalizer.load(norm_path) if norm_path.exists() else None
    return model, normalizer, cfg


class TrainingMonitor(BaseCallback):
    """Curriculum по difficulty + доля падений в обучающих эпизодах."""

    def __init__(self, start: float, ramp_fraction: float, total: int, log_every: int = 10_000):
        super().__init__()
        self.start, self.ramp, self.total, self.log_every = start, ramp_fraction, total, log_every
        self._ends, self._crashes, self._next = 0, 0, 0
        self.difficulty = start

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        if dones.any():
            for info, done in zip(self.locals["infos"], dones):
                if done:
                    self._ends += 1
                    self._crashes += int(info.get("crashed", False))
        if self.num_timesteps >= self._next:
            self._next = self.num_timesteps + self.log_every
            ramp_steps = max(self.ramp * self.total, 1.0)
            self.difficulty = float(min(1.0, self.start + (1.0 - self.start) * self.num_timesteps / ramp_steps))
            self.training_env.set_attr("difficulty", self.difficulty)
            self.logger.record("train/difficulty", self.difficulty)
            if self._ends:
                self.logger.record("train/crash_rate", self._crashes / self._ends)
                self._ends = self._crashes = 0
        return True


class ValidationCallback(BaseCallback):
    """Раз в eval_freq шагов: оценка на фиксированном наборе `val`; лучший по rmse_capped чекпоинт сохраняется."""

    def __init__(self, env_cfg: EnvConfig, run_dir: Path, eval_freq: int, suite: str = "val", n: int | None = None,
                 space: ConditionSpace | None = None):
        super().__init__()
        self.env_cfg, self.run_dir, self.eval_freq = env_cfg, Path(run_dir), eval_freq
        self.conditions = make_suite(suite, n, space=space)
        self.best = np.inf
        self.history: list[dict] = []
        self._next = eval_freq

    def evaluate(self) -> dict:
        vn = self.model.get_vec_normalize_env()
        normalizer = ObsNormalizer.from_vecnormalize(vn) if vn is not None and vn.norm_obs else None
        df = episode_metrics(rollout(SB3Policy(self.model, normalizer), self.env_cfg, self.conditions))
        return {"timesteps": self.num_timesteps, "rmse_capped": float(df.rmse_capped.mean()),
                "rmse_median": float(df.loc[~df.crashed, "rmse"].median()), "crash_rate": float(df.crashed.mean()),
                "ss_err": float(df.ss_err.median()), "hf_rms": float(df.hf_rms.median()),
                "return": float(df["return"].mean()), "normalizer": normalizer}

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.eval_freq
            self._run()
        return True

    def _run(self) -> None:
        res = self.evaluate()
        normalizer = res.pop("normalizer")
        for k, v in res.items():
            if k != "timesteps":
                self.logger.record(f"val/{k}", v)
        self.history.append(res)
        pd.DataFrame(self.history).to_csv(self.run_dir / "val_history.csv", index=False)
        if res["rmse_capped"] < self.best:
            self.best = res["rmse_capped"]
            self.model.save(self.run_dir / "best_model.zip")
            if normalizer is not None:
                normalizer.save(self.run_dir / "best_obs_norm.npz")

    def _on_training_end(self) -> None:
        if not self.history or self.history[-1]["timesteps"] != self.num_timesteps:
            self._run()


def environment_info() -> dict:
    import gymnasium
    import sb3_contrib
    import stable_baselines3
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "sb3": stable_baselines3.__version__, "sb3_contrib": sb3_contrib.__version__,
            "gymnasium": gymnasium.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def save_run_meta(run_dir: Path, cfg: dict, extra: dict) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps({**environment_info(), **extra}, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
