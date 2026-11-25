"""Ветровая лаборатория без окна: все ползунки и кнопки срабатывают и меняют симуляцию."""
import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

SPEC = importlib.util.spec_from_file_location("wind_lab", Path(__file__).resolve().parent.parent / "scripts" / "wind_lab.py")
wind_lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wind_lab)


def click(button):
    button._observers.process("clicked", None)


@pytest.fixture()
def view():
    lab = wind_lab.Lab(wind_lab.build_controllers([]))
    v = wind_lab.LabView(lab, interactive=False)
    yield v
    v.plt.close(v.fig)


def test_widgets_drive_the_simulation(view):
    lab = view.lab
    view.s_speed.set_val(4.0)
    click(view.compass_buttons[[b.label.get_text() for b in view.compass_buttons].index("З")])
    assert lab.wind.speed == 4.0 and lab.wind.from_deg == 270.0
    lab.advance(200)                                       # 4 с симуляции
    view.update()
    view.tick()
    pid = lab.controllers[0]
    assert pid.err and pid.crashes == 0
    assert np.mean(np.array(pid.xy[-50:])[:, 0]) > 0       # западный ветер сносит на восток
    assert "западный" in view.wind_label.get_text()

    click(view.b_gust)
    assert lab.wind._gusts
    click(view.b_push)
    assert np.isfinite(pid.core.ev_push_t[0])
    view.s_mass.set_val(1.2)
    view.s_motor.set_val(80)
    view.s_delay.set_val(2)
    assert pid.core.dyn.mass[0] == pytest.approx(1.2 * wind_lab.CF2X.mass)
    assert pid.core.dyn.motor_eff[0, 0] == pytest.approx(0.8) and pid.core.act_delay[0] == 2
    view.s_noise.set_val(0.0)
    assert not pid.core.noise[0].any()
    lab.advance(50)
    view.update()

    click(view.b_pause)
    t = lab.t
    view.tick()
    assert lab.t == t and view.paused
    click(view.b_pause)
    before = lab.controllers[1].enabled
    view.checks.set_active(1)                              # переключить PD
    assert lab.controllers[1].enabled != before
    lab.advance(2)
    assert lab.controllers[1].core.step_count[0] == lab.controllers[0].core.step_count[0]
    view.s_seed.set_val(5)                                 # новый сид → сброс
    assert lab.seed == 5 and lab.t == 0.0
    click(view.b_calm)
    assert lab.wind.speed == 0.0
    view.tick()
    view.update()


def test_snapshot_cli(tmp_path, monkeypatch):
    out = tmp_path / "lab.png"
    monkeypatch.setattr(wind_lab.sys, "argv", ["wind_lab.py", "--snapshot", str(out), "--seconds", "3",
                                               "--from-deg", "90", "--speed", "2"])
    wind_lab.main()
    assert out.exists() and out.stat().st_size > 50_000


def test_replay_restores_time_noise_and_scheduled_gust():
    lab = wind_lab.Lab(wind_lab.build_controllers([]), seed=3, gust_at=0.2)
    lab.wind.turb_sigma = 0.2
    lab.advance(30)
    wind = np.array([w for _, w in lab.wind_log])
    error = np.array(lab.controllers[0].err)
    lab.controllers[0].crashes = 1
    lab.reset()
    lab.advance(30)
    np.testing.assert_array_equal(wind, [w for _, w in lab.wind_log])
    np.testing.assert_array_equal(error, lab.controllers[0].err)
    assert lab.controllers[0].crashes == 0


def test_demo_matches_rppo_plant_and_pid_gains(monkeypatch):
    from quadwind.training import load_model
    config = {"name": "rppo__att", "env": {"action_mode": "attitude", "rate_ki": [100, 100, 50],
                                            "privileged": True, "w_vel": 0.2, "w_tilt": 1.0}}
    fake_model = type("Model", (), {"policy": object()})()
    monkeypatch.setattr(load_model.__module__ + ".load_model", lambda *_: (fake_model, None, config))
    agent, pid = wind_lab.build_controllers(["example"], demo=True)
    assert agent.name == "RPPO" and pid.name == "PID"
    assert pid.cfg == wind_lab.baseline_env_config(agent.cfg)
    assert pid.cfg.action_mode == "attitude" and pid.cfg.rate_ki == (100, 100, 50)
    assert pid.policy.gains == wind_lab.tuned_pid_gains(pid.cfg, wind_lab.ROOT / "configs/pid_tuned_pi_att.json")
