"""Подбор коэффициентов PID на наборе val (случайный поиск + локальное уточнение).

PID получает такой же «бюджет внимания», что и RL: подбор только на val, оценка — на test.
Цель — средняя rmse_capped (падение = 2 м), как у критерия выбора чекпоинта RL.
Пример:  python scripts/tune_pid.py --trials 150 --refine 60 --out configs/pid_tuned.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quadwind.conditions import make_suite  # noqa: E402
from quadwind.control import PIDGains  # noqa: E402
from quadwind.env import EnvConfig  # noqa: E402
from quadwind.evaluation import rollout  # noqa: E402
from quadwind.metrics import episode_metrics  # noqa: E402
from quadwind.policies import PIDPolicy  # noqa: E402

TUNED = ("kp_xy", "kd_xy", "ki_xy", "kp_z", "kd_z", "ki_z", "k_att_rp", "k_att_yaw", "ki_att_rp", "ki_att_yaw")


def score(gains: PIDGains, cfg: EnvConfig, conds) -> tuple[float, dict]:
    df = episode_metrics(rollout(PIDPolicy(cfg, gains), cfg, conds))
    return float(df.rmse_capped.mean()), {"crash_rate": float(df.crashed.mean()), "rmse_median": float(df.rmse.median())}


def perturb(base: PIDGains, rng: np.random.Generator, spread: float) -> PIDGains:
    d = asdict(base)
    for k in TUNED:
        d[k] = float(d[k] * np.exp(rng.uniform(-spread, spread)))
    return PIDGains(**d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--refine", type=int, default=60)
    ap.add_argument("--suite", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="configs/pid_tuned.json")
    ap.add_argument("--rate-ki", type=float, nargs=3, default=None, help="I-член внутреннего контура (PI)")
    args = ap.parse_args()

    cfg = EnvConfig(rate_ki=tuple(args.rate_ki)) if args.rate_ki else EnvConfig()
    conds, rng = make_suite(args.suite), np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    best = PIDGains()
    best_score, info = score(best, cfg, conds)
    log = [{"phase": "default", "score": best_score, **info, **asdict(best)}]
    print(f"default: {best_score:.4f} {info}")
    for phase, n, spread in [("random", args.trials, np.log(2.0)), ("refine", args.refine, np.log(1.25))]:
        for _ in range(n):
            cand = perturb(PIDGains() if phase == "random" else best, rng, spread)
            s, info = score(cand, cfg, conds)
            log.append({"phase": phase, "score": s, **info, **asdict(cand)})
            if s < best_score:
                best, best_score = cand, s
                print(f"{phase}: new best {best_score:.4f} {info}")
    out = ROOT / args.out
    out.write_text(json.dumps(asdict(best), indent=2), encoding="utf-8")
    pd.DataFrame(log).to_csv(out.with_suffix(".trials.csv"), index=False)
    print(f"best {best_score:.4f} -> {out}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
