"""S5: обобщение кандидатов на OOD-наборах (медиана выживших, см / доля падений)."""
import json, sys
import pandas as pd
sys.path.insert(0, ".")
from quadwind.conditions import make_suite, OOD_SPACES
from quadwind.control import PIDGains
from quadwind.env import EnvConfig
from quadwind.evaluation import evaluate_in_chunks
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
for suite in ["test_id"] + list(OOD_SPACES):
    for name, (p, ec) in pols.items():
        df = pd.concat([episode_metrics(r) for r in evaluate_in_chunks(p, ec, make_suite(suite))])
        ok = df[~df.crashed]
        rows.append({"suite": suite, "method": name, "med": 100 * ok.rmse.median(), "crash": 100 * df.crashed.mean()})
t = pd.DataFrame(rows)
t.to_csv("research/s5_ood_pi.csv", index=False)
cell = t.assign(c=lambda d: d.med.round(1).astype(str) + "/" + d.crash.round(0).astype(int).astype(str) + "%")
print(cell.pivot(index="suite", columns="method", values="c").to_string())
