"""Быстрый путь LSTM совпадает с исходной реализацией sb3-contrib (выходы, состояния и градиенты)."""
import pytest
import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from quadwind.fast_lstm import FastLstmPolicy


def _inputs(starts_rows):
    th.manual_seed(0)
    n_seq, length, dim, hidden = 5, 7, 6, 4
    lstm = th.nn.LSTM(dim, hidden)
    features = th.randn(n_seq * length, dim, requires_grad=True)
    states = (th.randn(1, n_seq, hidden), th.randn(1, n_seq, hidden))
    starts = th.zeros(n_seq, length)
    for seq, pos in starts_rows:
        starts[seq, pos] = 1.0
    return features, states, starts.flatten(), lstm


@pytest.mark.parametrize("starts", [[(0, 0), (3, 0)], [], [(1, 0), (2, 3)]])
def test_fast_path_matches_reference(starts):
    f1, s1, e1, lstm = _inputs(starts)
    out_ref, st_ref = RecurrentActorCriticPolicy._process_sequence(f1, s1, e1, lstm)
    out_ref.sum().backward()
    g_ref = f1.grad.clone()

    f2 = f1.detach().clone().requires_grad_(True)
    out_fast, st_fast = FastLstmPolicy._process_sequence(f2, s1, e1, lstm)
    out_fast.sum().backward()
    assert th.allclose(out_ref, out_fast, atol=1e-6)
    assert th.allclose(st_ref[0], st_fast[0], atol=1e-6) and th.allclose(st_ref[1], st_fast[1], atol=1e-6)
    assert th.allclose(g_ref, f2.grad, atol=1e-6)
