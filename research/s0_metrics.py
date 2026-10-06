"""S0: надёжность оценки. Val (64) против test_id (200): ранговая согласованность, ширина интервалов, связь метрик."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, ".")
from quadwind.conditions import make_suite
from quadwind.control import PIDGains
from quadwind.env import EnvConfig
from quadwind.evaluation import evaluate_in_chunks
from quadwind.metrics import episode_metrics
from quadwind.policies import PIDPolicy, SB3Policy
from quadwind.training import load_model

runs = sorted(p.parent for p in Path("runs").glob("*/*/seed_*/best_model.zip")
              if p.parts[1] in ("pilot2", "e7", "reward_search"))
pols = {}
g = PIDGains(**json.load(open("configs/pid_tuned.json")))
pols["pid"] = (PIDPolicy(EnvConfig(), g), EnvConfig())
pols["pd"] = (PIDPolicy(EnvConfig(), g.without_integrators()), EnvConfig())
for r in runs:
    m, n, c = load_model(r, "best")
    pols[str(r)] = (SB3Policy(m, n), EnvConfig.from_dict(c.get("env")))
rows = []
for name, (pol, ec) in pols.items():
    for suite in ("val", "test_id"):
        for ro in evaluate_in_chunks(pol, ec, make_suite(suite)):
            df = episode_metrics(ro); df["suite"] = suite; df["model"] = name; rows.append(df)
df = pd.concat(rows); df.to_csv("research/s0_episodes.csv", index=False)
agg = df.groupby(["model", "suite"]).agg(rmse_capped=("rmse_capped", "mean"), crash=("crashed", "mean"),
      med=("rmse", "median"), ss=("ss_err", "median"), hf=("hf_rms", "median")).unstack("suite")
print(agg.round(4).to_string())
v, t = agg[("rmse_capped", "val")], agg[("rmse_capped", "test_id")]
print("Spearman val~test rmse_capped:", stats.spearmanr(v, t))
v, t = agg[("med", "val")], agg[("med", "test_id")]
print("Spearman val~test median rmse:", stats.spearmanr(v, t))
rng = np.random.default_rng(0)
for suite, n in (("val", 64), ("test_id", 200)):
    widths = []
    for name, g_ in df[df.suite == suite].groupby("model"):
        x = g_.rmse_capped.to_numpy()
        b = [rng.choice(x, x.size).mean() for _ in range(1000)]
        widths.append(np.subtract(*np.percentile(b, [97.5, 2.5])))
    print(f"{suite}: median 95% CI width of mean rmse_capped across models = {100*np.median(widths):.2f} cm")
ep = df[df.suite == "test_id"][["rmse", "ss_err", "max_dev", "hf_rms", "effort", "action_rate", "max_tilt_deg"]]
print(ep.corr(method="spearman").round(2).to_string())
