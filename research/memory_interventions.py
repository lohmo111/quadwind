"""Б7 п. 2: какая память несёт компенсацию ветра — ∫-признак, скрытое состояние LSTM или интеграторы PID.

    python research/memory_interventions.py
Постоянный ветер 2 / 3 / 4 м/с × 4 направления (wind_sweep), в t* = 7,5 с обнуляется память:
у RPPO — h и c, у политик с ∫ — признак ∫e_p, у RPPO + ∫ — по отдельности и вместе, у PID — интеграторы по позиции.
Ошибка, см: средняя и пиковая за 2 с после t*, средняя за последние 2 с эпизода (восстановилась ли).
При чтении исторических CSV учитывайте, что прежний хук срабатывал после выбора действия.
При 50 Гц первая команда с изменённой памятью тогда выбиралась в 7,52 с.
"""
import glob, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quadwind.conditions import make_suite  # noqa: E402
from quadwind.control import PIDGains  # noqa: E402
from quadwind.env import EnvConfig  # noqa: E402
from quadwind.evaluation import rollout  # noqa: E402
from quadwind.policies import PIDPolicy, SB3Policy  # noqa: E402
from quadwind.training import load_model  # noqa: E402

GROUPS = {
    "MLP+∫": "ppo_mlp_integral__pi__safe__att",
    "RPPO": "rppo__pi__safe__att__asym__hp_epochs5__hp_lstm128",
    "RPPO+∫": "rppo_integral__pi__safe__att__asym__hp_epochs5__hp_lstm128",
}
SPEEDS = (2.0, 3.0, 4.0)
T_STAR = 7.5


def hook(step, zero_h=False, zero_int=False, zero_pid=False):
    def h(k, policy, core):
        if k != step:
            return
        if zero_h and getattr(policy, "state", None) is not None:
            policy.state = tuple(np.zeros_like(s) for s in policy.state)
        if zero_int:
            core.err_integral[:] = 0.0
        if zero_pid:
            policy.pid.integral[:] = 0.0
    return h


def measure(policy, ec, conds, **kw):
    step = int(T_STAR * ec.ctrl_freq)
    ro = rollout(policy, ec, conds, hook=hook(step, **kw))
    err = 100 * np.linalg.norm(ro.data["pos_err"], axis=2)            # (T, N), см
    width = int(2 * ec.ctrl_freq)
    window = slice(step, step + width)
    valid = err[window][ro.alive[window]]
    peaks = [err[step:min(step + width, end), i].max() for i, end in enumerate(ro.length) if end > step]
    last = np.concatenate([err[max(0, end - width):end, i] for i, end in enumerate(ro.length)])
    return {"mean_2s": valid.mean() if valid.size else np.nan,
            "peak_2s": np.mean(peaks) if peaks else np.nan,
            "last_2s": last.mean(), "crashed": ro.crashed.mean()}


def main():
    torch.set_num_threads(2)
    conds = [c for c in make_suite("wind_sweep") if c.wind.speed in SPEEDS]
    rows = []
    ec = EnvConfig(rate_ki=(100.0, 100.0, 50.0), action_mode="attitude")
    gains = PIDGains(**json.load(open("configs/pid_tuned_pi_att.json")))
    for name, kw in (("—", {}), ("обнулить интеграторы PID", {"zero_pid": True})):
        rows.append({"method": "PID", "seed": "", "intervention": name, **measure(PIDPolicy(ec, gains), ec, conds, **kw)})
    for group, key in GROUPS.items():
        for run in sorted(glob.glob(f"runs/e7/{key}/seed_*")):
            model, norm, cfg = load_model(Path(run), "best")
            ec = EnvConfig.from_dict(cfg.get("env"))
            variants = [("—", {})]
            if ec.obs_integral:
                variants.append(("обнулить ∫", {"zero_int": True}))
            if group.startswith("RPPO"):
                variants.append(("обнулить h", {"zero_h": True}))
            if ec.obs_integral and group.startswith("RPPO"):
                variants.append(("обнулить ∫ и h", {"zero_int": True, "zero_h": True}))
            for name, kw in variants:
                rows.append({"method": group, "seed": Path(run).name, "intervention": name,
                             **measure(SB3Policy(model, norm), ec, conds, **kw)})
            print("done", run, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("research/memory_interventions_runs.csv", index=False)
    agg = t.groupby(["method", "intervention"], sort=False)[["mean_2s", "peak_2s", "last_2s", "crashed"]].agg(
        lambda x: f"{x.mean():.1f} [{x.min():.1f}–{x.max():.1f}]" if len(x) > 1 else f"{x.iloc[0]:.1f}")
    agg.to_csv("research/memory_interventions.csv")
    pd.set_option("display.width", 200)
    print(agg.to_string())


if __name__ == "__main__":
    main()
