"""Инструменты анализа памяти: зонд восстанавливает линейную зависимость, хуки меняют состояние LSTM."""
import numpy as np

from quadwind.conditions import make_suite
from quadwind.env import EnvConfig
from quadwind.evaluation import rollout
from quadwind.memory import probe, r2, ridge_fit, ridge_predict, swap_state_hook, zero_state_hook
from quadwind.policies import SB3Policy
from tests.test_evaluation import _FakeRecurrentModel


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 8))
    W = rng.normal(size=(8, 2))
    Y = X @ W + 0.01 * rng.normal(size=(2000, 2))
    model = ridge_fit(X[:1500], Y[:1500], alpha=1e-3)
    assert r2(Y[1500:], ridge_predict(model, X[1500:])) > 0.99


def test_probe_on_features_that_contain_target():
    cfg = EnvConfig()
    ro = rollout(SB3Policy(_FakeRecurrentModel()), cfg, make_suite("val", 12), record_hidden=True)
    # «идеальные» признаки: сама медленная часть ветра + шум → R² близко к 1
    feats = ro.data["wind_slow"] + 0.01 * np.random.default_rng(0).normal(size=ro.data["wind_slow"].shape)
    res = probe(ro, feats, "wind_slow", time_bins=np.array([0, 5, 15.1]))
    assert res["r2"] > 0.95 and len(res["r2_by_time"]) == 2


def test_hooks_modify_lstm_state():
    pol = SB3Policy(_FakeRecurrentModel())
    pol.reset(2)
    obs = np.zeros((2, 22), dtype=np.float32)
    pol.act(obs, np.array([True, True]))
    pol.act(obs, np.array([False, False]))
    pol.state = (pol.state[0] * np.array([1.0, 3.0])[None, :, None], pol.state[1])
    swap_state_hook(5, np.array([1, 0]))(5, pol, None)
    assert pol.state[0][0, 0, 0] == 6.0 and pol.state[0][0, 1, 0] == 2.0
    zero_state_hook(5)(4, pol, None)                     # не тот шаг — без изменений
    assert pol.state[0][0, 0, 0] == 6.0
    zero_state_hook(5)(5, pol, None)
    assert not pol.state[0].any()


def test_intervention_affects_requested_decision():
    ro = rollout(SB3Policy(_FakeRecurrentModel()), EnvConfig(), make_suite("calm", 1),
                 record_hidden=True, hook=zero_state_hook(2))
    np.testing.assert_array_equal(ro.hidden[:4, 0, 0], [1, 2, 1, 2])


def test_intervention_stats_exclude_finished_flights():
    from types import SimpleNamespace
    from scripts.analyze_memory import intervention_stats

    errors = np.array([[1., 2.], [100., 4.]])
    r = SimpleNamespace(data={"pos_err": errors[:, :, None]}, alive=np.array([[1, 1], [0, 1]], dtype=bool),
                        crashed=np.array([True, False]))
    pairs = np.ones(2, dtype=bool)
    stats = intervention_stats(r, slice(0, 2), pairs)
    assert stats["mean_err_2s"] == 7 / 3 and stats["peak_err_2s"] == 4.0
    empty = intervention_stats(r, slice(7, 10), pairs)
    assert empty["samples_2s"] == 0 and empty["mean_err_2s"] is None and empty["peak_err_2s"] is None
