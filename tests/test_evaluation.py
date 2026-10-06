"""Прогон, метрики, статистика и адаптеры политик (в т.ч. передача LSTM-состояния)."""
import numpy as np
import pytest

from quadwind.conditions import make_suite
from quadwind.env import EnvConfig
from quadwind.evaluation import Rollout, rollout
from quadwind.metrics import RECOVER_RADIUS, _event_metrics, episode_metrics
from quadwind.policies import ConstantPolicy, ObsNormalizer, PIDPolicy, SB3Policy
from quadwind.stats import hierarchical_bootstrap, iqm, prob_improvement

CFG = EnvConfig()


def test_rollout_is_deterministic():
    conds = make_suite("test_id", 4)
    a, b = rollout(PIDPolicy(CFG), CFG, conds), rollout(PIDPolicy(CFG), CFG, conds)
    for k in a.data:
        assert np.array_equal(a.data[k], b.data[k]), k
    assert np.array_equal(a.reward, b.reward)


def test_crashes_are_recorded_and_masked():
    ro = rollout(ConstantPolicy((-1.0, 0, 0, 0)), CFG, make_suite("calm", 3))
    assert ro.crashed.all() and (ro.length < 100).all()
    for i in range(3):
        assert ro.alive[: ro.length[i], i].all() and not ro.alive[ro.length[i]:, i].any()
    df = episode_metrics(ro)
    assert (df.rmse_capped == 2.0).all() and df.ss_err.isna().all()


def test_event_metrics_on_synthetic_profile():
    t = np.arange(1, 751) * 0.02
    dist = np.full_like(t, 0.01)
    bump = (t >= 5.0) & (t < 7.0)
    dist[bump] = 0.2                                   # отклонение 20 см на 2 с после события
    m = _event_metrics(t, dist, 5.0)
    assert m["event_peak"] == pytest.approx(0.2)
    assert m["event_recovery_time"] == pytest.approx(2.0, abs=0.021)
    assert m["event_pre_err"] == pytest.approx(0.01)
    never = np.where(t >= 5.0, 0.2, 0.01)
    assert np.isnan(_event_metrics(t, never, 5.0)["event_recovery_time"])
    assert _event_metrics(t, np.full_like(t, RECOVER_RADIUS / 2), 5.0)["event_recovery_time"] == 0.0


class _FakeLSTMPolicy:
    lstm_actor = object()


class _FakeRecurrentModel:
    """Считает шаги в «скрытом состоянии»; сбрасывает счётчик по episode_start, как RecurrentPPO."""
    policy = _FakeLSTMPolicy()

    def __init__(self):
        self.calls = []

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        n = obs.shape[0]
        h = np.zeros((1, n, 2)) if state is None else state[0].copy()
        h[0, np.asarray(episode_start, dtype=bool)] = 0.0
        h[0, :, 0] += 1.0
        self.calls.append((state is None, np.asarray(episode_start).copy()))
        return np.zeros((n, 4), dtype=np.float32), (h, h.copy())


def test_recurrent_adapter_passes_state_and_resets_on_episode_start():
    model = _FakeRecurrentModel()
    pol = SB3Policy(model)
    assert pol.recurrent
    pol.reset(2)
    obs = np.zeros((2, 22), dtype=np.float32)
    for k in range(5):
        pol.act(obs, np.array([k == 0, k == 0]))
    assert np.allclose(pol.hidden()[:, 0], 5.0)               # состояние накапливалось, а не обнулялось
    pol.act(obs, np.array([True, False]))                     # новый эпизод только у первой среды
    assert pol.hidden()[0, 0] == 1.0 and pol.hidden()[1, 0] == 6.0
    assert model.calls[0][0] and not any(c[0] for c in model.calls[1:])


def test_recurrent_rollout_records_hidden():
    ro = rollout(SB3Policy(_FakeRecurrentModel()), CFG, make_suite("calm", 2), record_hidden=True)
    assert ro.hidden.shape[1:] == (2, 2)
    assert np.allclose(ro.hidden[:, 0, 0], np.arange(1, ro.hidden.shape[0] + 1))


def test_obs_normalizer_roundtrip(tmp_path):
    norm = ObsNormalizer(np.array([1.0, 2.0]), np.array([4.0, 9.0]), clip=5.0)
    assert np.allclose(norm(np.array([[3.0, 2.0]])), [[1.0, 0.0]], atol=1e-6)
    norm.save(tmp_path / "n.npz")
    back = ObsNormalizer.load(tmp_path / "n.npz")
    assert np.allclose(back.mean, norm.mean) and back.clip == 5.0


def test_stats_helpers():
    assert iqm([1, 2, 3, 4, 100, -100, 5, 6]) == pytest.approx(3.5)
    rng = np.random.default_rng(0)
    per_seed = [rng.normal(1.0, 0.1, 50) for _ in range(5)]
    point, lo, hi = hierarchical_bootstrap(per_seed, n_boot=500)
    assert lo <= point <= hi and hi - lo < 0.2
    assert prob_improvement([1, 1, 1], [2, 2, 2]) == 1.0
    assert prob_improvement([1, 2], [1, 2]) == 0.5


def test_rollout_type():
    ro = rollout(ConstantPolicy(), CFG, make_suite("calm", 1))
    assert isinstance(ro, Rollout) and ro.t[0] == pytest.approx(0.02)
