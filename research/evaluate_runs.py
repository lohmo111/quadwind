"""Оценка запусков по двум осям (S0): точность и безопасность + сценарии. Кэш на запуск в research/cache/.

    python research/evaluate_runs.py "runs/e7/*/seed_*" --out research/table.csv
Столбцы: med/mean rmse выживших (val 256), ss_err, падения на safety (1000) с 95% интервалом Уилсона,
статическая ошибка при 2–3 м/с, пик и восстановление после ступеньки ветра, пик после подхвата груза, дрожание.
"""
import argparse, glob, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quadwind.conditions import make_suite  # noqa: E402
from quadwind.control import PIDGains  # noqa: E402
from quadwind.env import EnvConfig  # noqa: E402
from quadwind.evaluation import evaluate_in_chunks  # noqa: E402
from quadwind.metrics import episode_metrics  # noqa: E402
from quadwind.policies import PIDPolicy, SB3Policy  # noqa: E402
from quadwind.training import load_model  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
SUITES = ("val", "safety", "wind_sweep", "step_gust", "mass_pickup")


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def episodes(name, policy, env_cfg):
    path = CACHE / f"{name.replace('/', '__').replace(chr(92), '__')}.csv"
    if path.exists():
        return pd.read_csv(path)
    frames = []
    for suite in SUITES:
        for ro in evaluate_in_chunks(policy, env_cfg, make_suite(suite)):
            df = episode_metrics(ro)
            df["suite"] = suite
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    CACHE.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


def summarize(name, df):
    val, saf = df[df.suite == "val"], df[df.suite == "safety"]
    ok = val[~val.crashed]
    k, n = int(saf.crashed.sum()), len(saf)
    lo, hi = wilson(k, n)
    sw = df[(df.suite == "wind_sweep") & df.cond_wind_speed.between(2, 3)]
    g, m = df[df.suite == "step_gust"], df[df.suite == "mass_pickup"]
    return {"run": name, "med_cm": 100 * ok.rmse.median(), "mean_cm": 100 * ok.rmse.mean(),
            "ss_cm": 100 * ok.ss_err.median(), "crash_val%": 100 * val.crashed.mean(),
            "crash_safety%": 100 * k / n, "crash_ci": f"{100 * lo:.1f}–{100 * hi:.1f}",
            "sweep_ss_cm": 100 * sw.ss_err.mean(), "gust_peak_cm": 100 * g.event_peak.median(),
            "gust_rec_s": g.event_recovery_time.median(), "mass_peak_cm": 100 * m.event_peak.median(),
            "hf": ok.hf_rms.median()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--which", default="best")
    ap.add_argument("--out", default="research/table.csv")
    ap.add_argument("--pid", action="store_true", help="добавить PID/PD (подобранные, P- и PI-контур)")
    args = ap.parse_args()
    rows = []
    if args.pid:
        g = PIDGains(**json.loads(Path("configs/pid_tuned_pi.json").read_text()))
        for loop, ki in (("P", (0, 0, 0)), ("PI", (100.0, 100.0, 50.0))):
            ec = EnvConfig(rate_ki=ki)
            for lab, gg in (("pid", g), ("pd", g.without_integrators())):
                # ключ кэша называет коэффициенты (pid_tuned_pi): старые pid_P / pid_PI посчитаны с pid_tuned.json
                rows.append(summarize(f"{lab}_{loop}", episodes(f"{lab}PI_on_{loop}", PIDPolicy(ec, gg), ec)))
    for run in sorted({p for pat in args.runs for p in glob.glob(pat)}):
        run = Path(run)
        if not (run / f"{args.which}_model.zip").exists() or not (run / "final_model.zip").exists():
            continue
        name = f"{run.parent.name}/{run.name}"
        model, norm, cfg = load_model(run, args.which)
        rows.append(summarize(name, episodes(f"{run.parts[-3]}_{name}_{args.which}", SB3Policy(model, norm),
                                             EnvConfig.from_dict(cfg.get("env")))))
        print("done", name, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
