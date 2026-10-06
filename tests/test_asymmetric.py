"""Асимметричный критик: действия не зависят от привилегированных признаков, ценность — зависит."""
import numpy as np
import pytest
import torch as th

from quadwind.env import PRIV_DIM
from quadwind.training import load_config, make_env, make_model


@pytest.mark.parametrize("algo_cfg", ["configs/ppo_mlp.yaml", "configs/rppo.yaml"])
def test_actor_ignores_privileged_features_critic_uses_them(algo_cfg, tmp_path):
    cfg = load_config(algo_cfg)
    cfg.update(asymmetric_critic=True, n_envs=2)
    cfg["env"] = {"privileged": True}
    cfg["algo_kwargs"].update(n_steps=64, batch_size=64, n_epochs=1)
    env = make_env(cfg, 0)
    model = make_model(cfg, env, seed=0)
    model.learn(128)                                         # обучение запускается

    obs = np.random.default_rng(0).normal(size=(2, env.observation_space.shape[0])).astype(np.float32)
    obs2 = obs.copy()
    obs2[:, -PRIV_DIM:] += 3.0                              # меняем только привилегированную часть
    kw = {"state": None, "episode_start": np.ones(2, bool)} if cfg["algo"] == "rppo" else {}
    a1, _ = model.predict(obs, deterministic=True, **kw)
    a2, _ = model.predict(obs2, deterministic=True, **kw)
    assert np.allclose(a1, a2, atol=1e-6)

    t1, t2 = (th.as_tensor(o) for o in (obs, obs2))
    if cfg["algo"] == "rppo":
        states = tuple(th.zeros(1, 2, model.policy.lstm_critic.hidden_size) for _ in range(2))
        starts = th.ones(2)
        v1 = model.policy.predict_values(t1, states, starts)
        v2 = model.policy.predict_values(t2, states, starts)
    else:
        v1, v2 = model.policy.predict_values(t1), model.policy.predict_values(t2)
    assert not th.allclose(v1, v2)

    model.save(tmp_path / "m.zip")                           # сохранение и загрузка с actor_dim
    loaded = type(model).load(tmp_path / "m.zip", device="cpu")
    a3, _ = loaded.predict(obs2, deterministic=True, **kw)
    assert np.allclose(a1, a3, atol=1e-6)


def test_asymmetric_requires_privileged_env():
    cfg = load_config("configs/ppo_mlp.yaml")
    cfg.update(asymmetric_critic=True, n_envs=1)
    with pytest.raises(ValueError):
        make_model(cfg, make_env(cfg, 0), seed=0)
