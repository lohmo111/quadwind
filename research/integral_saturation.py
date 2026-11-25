"""R7: почему политики с ∫ в attitude теряют интеграл при 6 м/с — насыщение признака (|∫| ≤ obs_integral_max) или действия.

    python research/integral_saturation.py runs/e7/ppo_mlp_integral__pi__safe__att/seed_0 ...
Печатает по скоростям wind_sweep: max |∫xy| и доля шагов на пределе за последние 2 с, |команда наклона| и доля насыщения.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quadwind.conditions import make_suite  # noqa: E402
from quadwind.env import EnvConfig  # noqa: E402
from quadwind.evaluation import rollout  # noqa: E402
from quadwind.policies import SB3Policy  # noqa: E402
from quadwind.training import load_model  # noqa: E402

torch.set_num_threads(2)
suite = make_suite("wind_sweep")
for run in sys.argv[1:]:
    m, n, c = load_model(Path(run), "best")
    ec = EnvConfig.from_dict(c.get("env"))
    pol = SB3Policy(m, n)
    acts, ints = [], []
    orig = pol.act

    def act(obs, start):
        a = orig(obs, start)
        acts.append(np.asarray(a).copy())
        return a
    pol.act = act
    ro = rollout(pol, ec, suite, hook=lambda k, p, core: ints.append(core.err_integral.copy()))
    A, I = np.asarray(acts), np.asarray(ints)       # (T, N, 4), (T, N, 3)
    last = slice(-100, None)                         # последние 2 с
    speeds = np.array([cc.wind.speed for cc in suite])
    print(run, ec.action_mode)
    for s in (3.0, 4.0, 5.0, 6.0):
        k = speeds == s
        ok = k & ~ro.crashed
        print(f"  {s} м/с: |∫xy| max {np.abs(I[last][:, ok, :2]).max(axis=(0, 2)).mean():.2f}"
              f"  доля |∫| на пределе: {(np.abs(I[last][:, ok, :2]) > 0.999 * ec.obs_integral_max).mean():.2f}"
              f"  |a_tilt| {np.abs(A[last][:, ok, 1:3]).max(axis=2).mean():.2f}"
              f"  доля |a|>0.99: {(np.abs(A[last][:, ok, 1:3]) > 0.99).mean():.2f}  падений {int((k & ro.crashed).sum())}/{k.sum()}")
