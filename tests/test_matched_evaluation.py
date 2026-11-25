"""The CLI compares a trained policy and its baseline on the same plant and seeds."""
import sys

import pandas as pd

from scripts import evaluate
from scripts.summarize_hover import summarize
from quadwind.policies import PIDPolicy


def test_evaluation_matches_pid_to_saved_run(tmp_path, monkeypatch):
    run = tmp_path / "seed_0"
    run.mkdir()
    (run / "best_model.zip").touch()
    config = {"name": "rppo", "env": {"action_mode": "attitude", "rate_ki": [100, 100, 50],
                                       "privileged": True, "w_vel": 0.2}}
    model = type("Model", (), {"policy": object()})()
    monkeypatch.setattr(evaluate, "load_model", lambda *_: (model, None, config))
    calls = []

    def record(policy, env_cfg, suites, tags, namespace):
        calls.append((policy, env_cfg, suites, namespace))
        return pd.DataFrame([{**tags, "suite": "calm", "rmse_capped": 0.0, "crashed": False}])

    monkeypatch.setattr(evaluate, "evaluate_policy", record)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--runs", str(run), "--baselines", "pid",
                                      "--suites", "calm", "--namespace", "final", "--out", str(tmp_path / "out.csv")])
    evaluate.main()
    (_, agent_cfg, suites, ns), (pid, pid_cfg, pid_suites, pid_ns) = calls
    assert isinstance(pid, PIDPolicy)
    assert pid_cfg == evaluate.baseline_env_config(agent_cfg)
    assert pid_cfg.rate_ki == (100, 100, 50) and pid_cfg.action_mode == "attitude"
    assert pid.gains == evaluate.tuned_pid_gains(pid_cfg, evaluate.ROOT / "configs/pid_tuned_pi_att.json")
    assert suites == pid_suites == ["calm"] and ns == pid_ns == "final"
    out = pd.read_csv(tmp_path / "out.csv")
    assert list(zip(out.backend, out.checkpoint)) == [("sb3", "best"), ("baseline", "fixed")]


def test_summary_keeps_trainers_and_checkpoints_separate(tmp_path):
    legacy = tmp_path / "seed_0"
    legacy.mkdir()
    (legacy / "config.yaml").write_text("algo: rppo\n", encoding="utf-8")
    base = dict(namespace="final", environment_group=1, suite="test_id", method="rppo",
                seed_train=0, cond_seed="same", crashed=False)
    episodes = pd.DataFrame([
        {**base, "run": str(legacy), "rmse": 0.02, "rmse_capped": 0.02},
        {**base, "run": "new", "backend": "native", "checkpoint": "initial", "rmse": 1., "rmse_capped": 1.},
        {**base, "run": "new", "backend": "native", "checkpoint": "best", "rmse": 0.1, "rmse_capped": 0.1},
    ])
    results = summarize(episodes).set_index(["backend", "checkpoint"])
    assert results.median_surviving_rmse_cm.to_dict() == {
        ("sb3", "unrecorded"): 2., ("native", "initial"): 100., ("native", "best"): 10.,
    }
