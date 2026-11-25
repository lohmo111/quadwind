"""Анализ памяти обученной RPPO: зонды и вмешательства в скрытое состояние.

Пример:  python scripts/analyze_memory.py --run runs/main/rppo/seed_0 --out results/memory/rppo_s0
Выход: summary.json, probe_r2_by_time.csv, intervention.csv и графики *.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quadwind.conditions import make_suite  # noqa: E402
from quadwind.env import EnvConfig  # noqa: E402
from quadwind.evaluation import rollout  # noqa: E402
from quadwind.memory import probe, swap_state_hook, zero_state_hook  # noqa: E402
from quadwind.policies import SB3Policy  # noqa: E402
from quadwind.training import load_model, make_env, make_model  # noqa: E402

TIME_BINS = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.01])


class ObsRecorder(SB3Policy):
    """SB3Policy, которая дополнительно отдаёт текущее (нормированное) наблюдение как «признаки»."""

    def act(self, obs, episode_start):
        normalized = self.normalizer(obs) if self.normalizer is not None else obs
        # Compare hidden state to the information actually available to the actor.
        self.last_obs = normalized[:, :self.model.policy.lstm_actor.input_size]
        return super().act(obs, episode_start)

    def hidden(self):
        return self.last_obs


def collect(policy, env_cfg, conds):
    return rollout(policy, env_cfg, conds, record_hidden=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--which", default="best")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=256, help="эпизодов из набора probe")
    args = ap.parse_args()
    run_dir = Path(args.run)
    out = Path(args.out) if args.out else ROOT / "results" / "memory" / "_".join(run_dir.parts[-3:])
    out.mkdir(parents=True, exist_ok=True)

    model, normalizer, cfg = load_model(run_dir, args.which)
    if not hasattr(model.policy, "lstm_actor"):
        ap.error("memory analysis requires a recurrent policy")
    env_cfg = EnvConfig.from_dict(cfg.get("env"))
    conds = make_suite("probe", args.n)

    # 1. зонды: обученная LSTM, текущее наблюдение, LSTM со случайными весами
    ro = collect(SB3Policy(model, normalizer), env_cfg, conds)
    ro_obs = collect(ObsRecorder(model, normalizer), env_cfg, conds)
    random_model = make_model(cfg, make_env(cfg, 0), seed=123)
    # случайная LSTM получает те же наблюдения, что и обученная (управляет обученная политика)
    feats_random = []

    class Twin(SB3Policy):
        def act(self, obs, episode_start):
            a = super().act(obs, episode_start)
            x = self.normalizer(obs) if self.normalizer is not None else obs
            _, self.rstate = random_model.predict(x, state=self.rstate, episode_start=episode_start, deterministic=True)
            h = self.rstate[0]
            feats_random.append(np.transpose(h, (1, 0, 2)).reshape(h.shape[1], -1))
            return a

        def reset(self, n):
            super().reset(n)
            self.rstate = None

    collect(Twin(model, normalizer), env_cfg, conds)
    feats = {"lstm_trained": ro.hidden, "observation": ro_obs.hidden, "lstm_random": np.asarray(feats_random)}

    summary, rows = {}, []
    for target in ("wind_slow", "wind", "mass_ratio"):
        for name, F in feats.items():
            res = probe(ro if name != "observation" else ro_obs, F, target, time_bins=TIME_BINS)
            summary[f"{target}/{name}"] = res["r2"]
            for (a, b), v in zip(zip(TIME_BINS[:-1], TIME_BINS[1:]), res["r2_by_time"]):
                rows.append({"target": target, "features": name, "t_from": a, "t_to": b, "r2": v})
    by_time = pd.DataFrame(rows)
    by_time.to_csv(out / "probe_r2_by_time.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, target in zip(axes, ("wind_slow", "mass_ratio")):
        for name, g in by_time[by_time.target == target].groupby("features"):
            ax.plot(0.5 * (g.t_from + g.t_to), g.r2, marker="o", label=name)
        ax.set_xscale("log")
        ax.set_title(f"R² зонда → {target}")
        ax.set_xlabel("время с начала эпизода, с")
        ax.axhline(0, color="grey", lw=0.5)
    axes[0].set_ylabel("R² (отложенные эпизоды)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "probe_r2_by_time.png", dpi=130)

    # 2. вмешательства: постоянный ветер 3 м/с в 4 направлениях, t* = 7.5 с
    sweep = [c for c in make_suite("wind_sweep") if c.wind.speed == 3.0]
    sweep = [replace(c, noise_scale=1.0) for c in sweep]
    t_star = int(7.5 / env_cfg_dt(env_cfg))
    base = rollout(SB3Policy(model, normalizer), env_cfg, sweep)
    zero = rollout(SB3Policy(model, normalizer), env_cfg, sweep, hook=zero_state_hook(t_star))
    opposite = np.array([2, 3, 0, 1])                      # направления 0↔π, π/2↔3π/2
    swap = rollout(SB3Policy(model, normalizer), env_cfg, sweep, hook=swap_state_hook(t_star, opposite))
    window = slice(t_star, t_star + int(2.0 / env_cfg_dt(env_cfg)))
    inter = []
    for name, r in (("none", base), ("zero_h", zero), ("swap_opposite_wind_h", swap)):
        err = np.linalg.norm(r.data["pos_err"][window], axis=2)
        inter.append({"intervention": name, "mean_err_2s": float(err.mean()), "peak_err_2s": float(err.max()),
                      "crashed": float(r.crashed.mean())})
    pd.DataFrame(inter).to_csv(out / "intervention.csv", index=False)

    t = base.t
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for name, r in (("без вмешательства", base), ("обнуление h", zero), ("h от противоположного ветра", swap)):
        ax.plot(t, np.linalg.norm(r.data["pos_err"][:, 0], axis=1), label=name)
    ax.axvline(t[t_star], color="grey", ls="--", lw=0.8)
    ax.set_xlabel("время, с")
    ax.set_ylabel("ошибка позиции, м")
    ax.set_title("Вмешательство в память при ветре 3 м/с")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "intervention.png", dpi=130)

    summary["intervention"] = inter
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"-> {out}")


def env_cfg_dt(env_cfg: EnvConfig) -> float:
    return 1.0 / env_cfg.ctrl_freq


if __name__ == "__main__":
    main()
