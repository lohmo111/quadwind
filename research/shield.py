"""Экран безопасности: RL-политика, но при опасном состоянии (наклон/скорость) действует PID. Без переобучения."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
sys.path.insert(0, "research")
from evaluate_runs import summarize, episodes
from quadwind.control import PIDGains
from quadwind.env import EnvConfig
from quadwind.policies import PIDPolicy, SB3Policy
from quadwind.training import load_model


class Shield:
    def __init__(self, rl, pid, tilt_deg, speed, hold_steps=25):
        self.rl, self.pid, self.cos_t, self.speed, self.hold = rl, pid, np.cos(np.radians(tilt_deg)), speed, hold_steps
    def reset(self, n):
        self.rl.reset(n); self.pid.reset(n); self.left = np.zeros(n, int); self.share = []
    def act(self, obs, start):
        a_rl = self.rl.act(obs, start); a_pid = self.pid.act(obs, start)   # PID ведёт свои интеграторы всегда
        f = obs[:, :22]
        danger = (f[:, 14] < self.cos_t) | (np.linalg.norm(f[:, 3:6], axis=1) > self.speed)
        self.left = np.where(danger, self.hold, np.maximum(self.left - 1, 0))
        use = self.left > 0
        self.share.append(use.mean())
        return np.where(use[:, None], a_pid, a_rl).astype(np.float32)


g = PIDGains(**json.load(open("configs/pid_tuned_pi.json")))
rows = []
for run in ["runs/e7/rppo/seed_2", "runs/e7/ppo_mlp_integral/seed_1"]:
    m, n, c = load_model(run, "best")
    ec = EnvConfig.from_dict(c.get("env"))
    for tilt, spd in [(30, 1.0), (25, 0.7), (20, 0.5)]:
        sh = Shield(SB3Policy(m, n), PIDPolicy(ec, g), tilt, spd)
        name = f"shield_{tilt}_{spd}_{run.split('/')[2]}_{run[-1]}"
        r = summarize(name, episodes(name, sh, ec))
        r["pid_share%"] = 100 * np.mean(sh.share) if hasattr(sh, "share") else np.nan
        rows.append(r)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(2).to_string(index=False))
