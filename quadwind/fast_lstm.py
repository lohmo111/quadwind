"""Ускоренная LSTM-политика для RecurrentPPO (математически та же, что MlpLstmPolicy).

В sb3-contrib 2.x, если в минибатче есть хоть одно начало эпизода, LSTM прогоняется
поэлементным Python-циклом по времени. Обучающие последовательности там режутся ровно по
началам эпизодов, поэтому начало эпизода стоит только в позиции 0 и цикл включается почти
всегда. Если сбросить начальное состояние таких последовательностей один раз до вызова,
можно прогнать всю последовательность одним вызовом nn.LSTM — это то же самое вычисление.
Общий случай (начало эпизода внутри последовательности) уходит в исходную реализацию.
Эквивалентность проверяет tests/test_fast_lstm.py.
"""
from __future__ import annotations

import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


class FastLstmPolicy(RecurrentActorCriticPolicy):
    @staticmethod
    def _process_sequence(features, lstm_states, episode_starts, lstm):
        n_seq = lstm_states[0].shape[1]
        seq = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        if not th.all(starts[1:] == 0.0):
            return RecurrentActorCriticPolicy._process_sequence(features, lstm_states, episode_starts, lstm)
        keep = (1.0 - starts[0]).view(1, n_seq, 1)
        out, states = lstm(seq, (keep * lstm_states[0], keep * lstm_states[1]))
        return th.flatten(out.transpose(0, 1), start_dim=0, end_dim=1), states
