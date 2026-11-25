"""Train through the CLI, then load native checkpoints through the demo's interface."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from quadwind.env import EnvConfig
from quadwind.policies import SB3Policy
from quadwind.training import load_model


ROOT = Path(__file__).resolve().parents[1]


def test_native_hover_train_validate_and_reload(tmp_path):
    overlay = tmp_path / "smoke.yaml"
    overlay.write_text(yaml.safe_dump({
        "variant": "smoke", "n_envs": 2, "total_timesteps": 32,
        "env": {"episode_seconds": 2.2},
        "eval": {"n": 2, "freq": 16},
        "algo_kwargs": {"n_steps": 8, "n_epochs": 2, "batch_size": 8},
        "policy_kwargs": {"lstm_hidden_size": 8, "net_arch": {"pi": [8], "vf": [8]}},
    }), encoding="utf-8")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/train.py"), "--config", "configs/rppo_hover.yaml",
        "--overlay", str(overlay), "--exp", str(tmp_path), "--seed", "17", "--threads", "1",
    ], cwd=ROOT, check=True)
    folder = tmp_path / "rppo_hover__smoke" / "seed_17"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "done" and meta["actual_timesteps"] == 32
    history = pd.read_csv(folder / "val_history.csv")
    assert history.timesteps.tolist() == [16, 32]
    assert np.isfinite(history.rmse_capped).all()

    initial, _, _ = load_model(folder, "initial")
    final, _, _ = load_model(folder, "final")
    assert not torch.equal(initial.policy.actor_head.weight, final.policy.actor_head.weight)
    model, normalizer, config = load_model(folder, "best")
    assert config["backend"] == "native" and normalizer is not None
    adapter = SB3Policy(model, normalizer)
    obs = np.ones((2, EnvConfig.from_dict(config["env"]).obs_dim), dtype=np.float32)
    start = np.ones(2, dtype=bool)
    adapter.reset(2)
    first = adapter.act(obs, start)
    assert first.shape == (2, 4) and np.isfinite(first).all()
    adapter.act(obs, ~start)
    assert adapter.hidden().shape == (2, 8)
    np.testing.assert_allclose(adapter.act(obs, start), first, atol=1e-6)
