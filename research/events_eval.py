"""S4b: отклик на события (пик, см / время восстановления, с) для кандидатов."""
import json, sys
import pandas as pd
sys.path.insert(0, ".")
from quadwind.conditions import make_suite
from quadwind.control import PIDGains
from quadwind.env import EnvConfig
from quadwind.evaluation import rollout
from quadwind.metrics import episode_metrics
from quadwind.policies import PIDPolicy, SB3Policy
from quadwind.training import load_model

runs = dict(a.split("=") for a in sys.argv[1:])
ec_pi = EnvConfig(rate_ki=(100.0, 100.0, 50.0))
pols = {"PID-PI": (PIDPolicy(ec_pi, PIDGains(**json.load(open("configs/pid_tuned_pi.json")))), ec_pi)}
for name, run in runs.items():
    m, n, c = load_model(run, "best")
    pols[name] = (SB3Policy(m, n), EnvConfig.from_dict(c.get("env")))
rows = []
for suite in ["step_gust", "gust_pulse", "mass_pickup", "mass_drop", "push", "motor_degrade"]:
    for name, (p, ec) in pols.items():
        df = episode_metrics(rollout(p, ec, make_suite(suite)))
        rows.append({"suite": suite, "method": name,
                     "c": f"{100*df.event_peak.median():.1f}/{df.event_recovery_time.median():.2f}"})
t = pd.DataFrame(rows).pivot(index="suite", columns="method", values="c")
t.to_csv("research/s4_events.csv"); print("пик, см / восстановление до 5 см, с (медианы по 32 эпизодам)"); print(t.to_string())
