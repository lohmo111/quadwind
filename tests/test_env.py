"""Среда: совместимость с gymnasium/SB3, автосброс, задержки, шум, история, оракул, события, детерминизм."""
from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from quadwind.conditions import NOMINAL_SPACE, TRAIN_SPACE, IntU, fixed, make_suite
from quadwind.env import PRIV_DIM, EnvConfig, HoverCore, QuadHoverEnv, QuadHoverVecEnv
from quadwind.params import CF2X

QUIET = replace(NOMINAL_SPACE, noise_scale=fixed(0.0), act_delay=IntU(0, 0))


def quiet_conditions(n, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [replace(QUIET, **kw).sample(rng) for _ in range(n)]


def test_gymnasium_and_sb3_checkers():
    gym_check_env(QuadHoverEnv(), skip_render_check=True)
    sb3_check_env(QuadHoverEnv(), warn=True)


def test_observation_dimensions():
    for history, priv in [(1, False), (4, False), (1, True), (16, True)]:
        cfg = EnvConfig(history=history, privileged=priv)
        env = QuadHoverVecEnv(cfg, 3)
        obs = env.reset()
        assert obs.shape == (3, history * 22 + (PRIV_DIM if priv else 0)) == (3, cfg.obs_dim)
        assert obs.dtype == np.float32
    assert EnvConfig(include_prev_action=False).obs_dim == 18


def test_noise_free_observation_equals_truth():
    core = HoverCore(EnvConfig(), 2)
    core.reset_envs([0, 1], quiet_conditions(2, init_radius=fixed(0.3), init_vel_std=fixed(0.2)))
    core.step(np.zeros((2, 4)))
    obs = core.observe()
    assert np.allclose(obs[:, 0:3], core.dyn.pos - core.target, atol=1e-6)
    assert np.allclose(obs[:, 3:6], core.dyn.vel, atol=1e-6)
    assert np.allclose(obs[:, 6:15], core.dyn.R.reshape(2, 9), atol=1e-6)


def test_observation_delay():
    cfg = EnvConfig()
    core = HoverCore(cfg, 1)
    core.reset_envs([0], quiet_conditions(1, obs_delay=IntU(2, 2), init_vel_std=fixed(0.5)))
    truth = []
    rng = np.random.default_rng(0)
    for k in range(10):
        truth.append(core.dyn.pos[0] - core.target)
        core.step(rng.uniform(-0.2, 0.2, (1, 4)))
        obs = core.observe()[0]
        expected = truth[max(k + 1 - 2, 0)]
        assert np.allclose(obs[0:3], expected, atol=1e-6), k


def test_action_delay_and_prev_action():
    core = HoverCore(EnvConfig(), 1)
    core.reset_envs([0], quiet_conditions(1, act_delay=IntU(2, 2)))
    actions = np.random.default_rng(1).uniform(-0.3, 0.3, (8, 1, 4))
    for k in range(8):
        core.step(actions[k])
        applied = core.applied_action[0]
        assert np.allclose(applied, actions[k - 2, 0] if k >= 2 else 0.0), k
        assert np.allclose(core.observe()[0, 18:22], actions[k, 0], atol=1e-6)   # a_{t−1} — отправленное


def test_history_stacks_previous_frames():
    cfg = EnvConfig(history=3)
    env = QuadHoverVecEnv(cfg, 2)
    prev = env.reset()
    for _ in range(5):
        obs, *_ = env.step(np.zeros((2, 4), dtype=np.float32))
        assert np.allclose(obs[:, 22:44], prev[:, 0:22])
        assert np.allclose(obs[:, 44:66], prev[:, 22:44])
        prev = obs


def test_privileged_features_contain_true_wind_and_mass():
    cfg = EnvConfig(privileged=True)
    conds = quiet_conditions(2, mass_ratio=fixed(1.2), wind_speed=fixed(2.0))
    core = HoverCore(cfg, 2)
    core.reset_envs([0, 1], conds)
    priv = core.observe()[:, -PRIV_DIM:]
    assert np.allclose(np.linalg.norm(priv[:, 0:3] * 5.0, axis=1), 2.0, atol=1e-5)
    assert np.allclose(priv[:, 3], 0.2, atol=1e-6)


def test_truncation_autoreset_and_timeout_flag():
    cfg = EnvConfig(episode_seconds=1.0)
    env = QuadHoverVecEnv(cfg, 2, space=QUIET)
    env.reset()
    for k in range(50):
        obs, rew, dones, infos = env.step(np.zeros((2, 4), dtype=np.float32))
        if k < 49:
            assert not dones.any()
    assert dones.all()
    for info in infos:
        assert info["TimeLimit.truncated"] and not info["crashed"]
        assert info["terminal_observation"].shape == (cfg.obs_dim,)
    assert (env.core.step_count == 0).all()          # автосброс выполнен
    assert (rew >= 0).all()


def test_crash_terminates_with_penalty():
    cfg = EnvConfig()
    env = QuadHoverVecEnv(cfg, 1, space=QUIET)
    env.reset()
    for _ in range(200):
        obs, rew, dones, infos = env.step(np.array([[-1.0, 0, 0, 0]], dtype=np.float32))  # моторы выключены
        if dones[0]:
            break
    assert dones[0] and infos[0]["crashed"] and not infos[0]["TimeLimit.truncated"]
    assert rew[0] == -cfg.crash_penalty


def test_reward_is_nonnegative_without_crash():
    env = QuadHoverVecEnv(EnvConfig(), 8, seed=3)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(300):
        _, rew, dones, infos = env.step(rng.uniform(-0.3, 0.3, (8, 4)).astype(np.float32))
        crashed = np.array([info.get("crashed", False) for info in infos])
        assert (rew[~crashed] >= 0).all() and (rew[~crashed] <= 2.0).all()


def test_seeding_is_reproducible_and_independent():
    def rollout(seed):
        env = QuadHoverVecEnv(EnvConfig(), 4)
        env.seed(seed)
        out = [env.reset()]
        rng = np.random.default_rng(0)
        for _ in range(30):
            out.append(env.step(rng.uniform(-0.2, 0.2, (4, 4)).astype(np.float32))[0])
        return np.array(out)
    a, b, c = rollout(7), rollout(7), rollout(8)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_fixed_conditions_do_not_depend_on_batch_composition():
    """Эпизод с условием c разворачивается одинаково, кто бы ни был с ним в батче (честные парные сравнения)."""
    conds = make_suite("test_id", 6)
    cfg = EnvConfig()

    def run(batch):
        core = HoverCore(cfg, len(batch))
        core.reset_envs(np.arange(len(batch)), batch)
        log = []
        for _ in range(100):
            core.step(np.zeros((len(batch), 4)))
            log.append(core.true_state()["wind"].copy())
        return np.array(log)

    full = run(conds)
    alone = run([conds[3]])
    swapped = run([conds[5], conds[3], conds[0]])
    assert np.array_equal(full[:, 3], alone[:, 0])
    assert np.array_equal(full[:, 3], swapped[:, 1])


def test_events_mass_motor_push():
    cfg = EnvConfig()
    core = HoverCore(cfg, 3)
    core.reset_envs([0, 1, 2], [make_suite("mass_pickup", 1)[0], make_suite("motor_degrade", 1)[0],
                                make_suite("push", 1)[0]])
    for k in range(int(5.0 * cfg.ctrl_freq)):
        core.step(np.zeros((3, 4)))
        if k == 0:
            assert core.dyn.mass[0] == pytest.approx(CF2X.mass) and core.dyn.motor_eff[1].min() == 1.0
    core.step(np.zeros((3, 4)))
    assert core.dyn.mass[0] == pytest.approx(1.2 * CF2X.mass)
    assert core.dyn.motor_eff[1].min() == pytest.approx(0.8)
    assert np.linalg.norm(core.dyn.ext_force[2]) == pytest.approx(0.3 * CF2X.weight)
    for _ in range(int(0.3 * cfg.ctrl_freq) + 1):
        core.step(np.zeros((3, 4)))
    assert not core.dyn.ext_force[2].any()


def test_difficulty_attribute_reaches_sampler():
    env = QuadHoverVecEnv(EnvConfig(), 4, space=TRAIN_SPACE, difficulty=0.0)
    env.reset()
    assert all(c.mass_ratio == 1.0 for c in env.core.conditions)
    env.set_attr("difficulty", 1.0)
    assert env.get_attr("difficulty") == [1.0] * 4
    env.reset()
    assert any(c.mass_ratio != 1.0 for c in env.core.conditions)


def test_attitude_mode_runs():
    env = QuadHoverVecEnv(EnvConfig(action_mode="attitude"), 2, space=QUIET)
    env.reset()
    for _ in range(50):
        obs, rew, dones, _ = env.step(np.zeros((2, 4), dtype=np.float32))
    assert not dones.any() and np.isfinite(obs).all()


@pytest.mark.parametrize("saturation", ["clip", "tanh"])
def test_reward_bounds_for_cost_saturation(saturation):
    cfg = EnvConfig(cost_saturation=saturation, w_tilt=0.3, w_dact=0.5)
    env = QuadHoverVecEnv(cfg, 8, seed=4)
    env.reset()
    rng = np.random.default_rng(1)
    for _ in range(200):
        _, rew, _, infos = env.step(rng.uniform(-1, 1, (8, 4)).astype(np.float32))
        ok = ~np.array([info.get("crashed", False) for info in infos])
        assert (rew[ok] >= 0).all() and (rew[ok] <= 2.0).all()


def test_integral_feature_accumulates_observed_error():
    cfg = EnvConfig(obs_integral=True)
    core = HoverCore(cfg, 1)
    core.reset_envs([0], quiet_conditions(1, init_radius=fixed(0.3)))
    assert np.allclose(core.observe()[0, -3:], 0.0)
    total = 0.0
    for _ in range(10):
        core.step(np.zeros((1, 4)))
        total += core.observe()[0, 0] * core.dt
    assert core.observe().shape[1] == cfg.obs_dim == 25
    assert core.observe()[0, -3] == pytest.approx(total, rel=1e-5)


def test_wind_override_replaces_episode_series():
    core = HoverCore(EnvConfig(privileged=True), 2)
    core.reset_envs([0, 1], quiet_conditions(2, wind_speed=fixed(1.0)))
    core.wind_override = np.array([0.0, -4.0, 0.0])                       # северный ветер 4 м/с
    for _ in range(30):
        core.step(np.zeros((2, 4)))
    assert np.allclose(core.true_state()["wind"], [0.0, -4.0, 0.0])
    assert np.allclose(core.observe()[:, -PRIV_DIM:-PRIV_DIM + 3] * 5.0, [0.0, -4.0, 0.0], atol=1e-5)
    assert (core.dyn.vel[:, 1] < 0).all()                                 # без управления сносит на юг
    core.wind_override = None
    core.step(np.zeros((2, 4)))
    assert np.allclose(np.linalg.norm(core.true_state()["wind"], axis=1), 1.0)


def test_residual_pid_zero_action_equals_pid():
    """Нулевая поправка агента — ровно PID по тем же наблюдениям."""
    import json
    from quadwind.control import PIDGains
    from quadwind.policies import PIDPolicy
    cfg = EnvConfig(residual_pid="configs/pid_tuned_pi.json")
    conds = make_suite("test_id", 3)
    res = HoverCore(cfg, 3)
    res.reset_envs([0, 1, 2], conds)
    ref = HoverCore(EnvConfig(), 3)
    ref.reset_envs([0, 1, 2], conds)
    pid = PIDPolicy(EnvConfig(), PIDGains(**json.load(open("configs/pid_tuned_pi.json"))))
    pid.reset(3)
    start = np.ones(3, bool)
    for _ in range(100):
        ref.step(pid.act(ref.observe(), start))
        start[:] = False
        res.step(np.zeros((3, 4)))
    assert np.allclose(res.dyn.pos, ref.dyn.pos, atol=1e-9)
