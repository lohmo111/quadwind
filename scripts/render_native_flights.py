"""Render paired early/best flights and the native trainer's validation histories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadwind.conditions import SUITE_SIZES, make_suite
from quadwind.env import EnvConfig, HoverCore
from quadwind.evaluation import rollout
from quadwind.policies import SB3Policy
from quadwind.training import build_space, load_model

COLORS = ("#ab4921", "#146b78", "#645698")


def paired_flights(run):
    early, norm, cfg = load_model(run, "early")
    if cfg.get("backend") != "native":
        raise ValueError(f"{run} must contain a native PPO-LSTM run")
    env_cfg = EnvConfig.from_dict(cfg["env"])
    # Fixed before inspecting the selected policy. The probe suite is independent
    # of validation, test_id and safety. Its seed reproduces the start and noise.
    conditions = make_suite("probe", 1, space=build_space(cfg), namespace="research")
    core = HoverCore(env_cfg, 1)
    core.reset_envs(np.array([0]), conditions)
    initial_error = core.true_state()["pos_err"][0]
    best, best_norm, _ = load_model(run, "best")
    flights = []
    rows = []
    seed = json.loads((run / "meta.json").read_text(encoding="utf8"))["seed"]
    for checkpoint, model, normalizer in (("early", early, norm), ("best", best, best_norm)):
        ro = rollout(SB3Policy(model, normalizer), env_cfg, conditions)
        error = np.vstack([initial_error, ro.data["pos_err"][:, 0]])
        dist = np.linalg.norm(error, axis=1)
        t = np.r_[0, ro.t]
        outcome = f"Crash ({ro.crash_reason[0]})" if ro.crashed[0] else "Completed"
        label = "Early" if checkpoint == "early" else "Validation best"
        print(f"{label}: {model.num_timesteps} steps, {outcome} at {t[-1]:.2f} s, "
              f"final 3D error {dist[-1]:.4f} m")
        flights.append((f"{label}   /   {model.num_timesteps:,} steps", t, error, outcome))
        rows.append(dict(checkpoint=checkpoint, seed_train=seed, timesteps=model.num_timesteps,
                         suite="research/probe", episode=0, condition_seed=conditions[0].to_record()["seed"],
                         length_s=t[-1], crashed=bool(ro.crashed[0]), crash_reason=ro.crash_reason[0],
                         peak_error_m=dist.max(), final_error_m=dist[-1]))
    return flights, rows, conditions[0], cfg


def write_video(flights, condition, seed, path):
    fig = plt.figure(figsize=(10, 6), facecolor="#fbfaf7")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.65, 1), left=.09, right=.97,
                            top=.83, bottom=.10, hspace=.63, wspace=.18)
    fig.text(.06, .956, "Early and selected policies on the same flight", fontsize=20, weight="bold", va="top")
    fig.text(.06, .895, f"Native PPO-LSTM   ·   training seed {seed}   ·   probe episode 0   ·   "
             f"base wind {condition.wind.speed:.2f} m/s", color="#425261", fontsize=11)
    duration = max(t[-1] for _, t, _, _ in flights)
    distance_ax = fig.add_subplot(grid[1, :])
    distance_ax.set(xlim=(0, duration), xlabel="Flight time (s)", ylabel="3D distance to target (m)")
    xy_limit = max(.1, max(abs(e[:, :2]).max() for _, _, e, _ in flights) * 1.12)
    artists = []
    for col, (name, _, error, _) in enumerate(flights):
        color = COLORS[col]
        ax = fig.add_subplot(grid[0, col], aspect="equal")
        ax.set(xlim=(-xy_limit, xy_limit), ylim=(-xy_limit, xy_limit),
               xlabel="x offset (m)", ylabel="y offset (m)")
        ax.set_title(name, color=color, weight="bold")
        ax.plot(0, 0, marker="+", markersize=13, markeredgewidth=2, color="#20232d", label="Target")
        ax.plot(*error[0, :2], "s", ms=5, color="#999999", label="Start")
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        trail, = ax.plot([], [], color=color, lw=1.8)
        dot, = ax.plot([], [], "o", color=color, ms=6)
        status = ax.text(.5, -.32, "", transform=ax.transAxes, ha="center", fontsize=10, color=color)
        distance_line, = distance_ax.plot([], [], color=color, lw=2, label=name.split("   /")[0])
        artists.append((trail, dot, status, distance_line))
    distance_ax.set_ylim(0, max(.1, max(np.linalg.norm(e, axis=1).max() for _, _, e, _ in flights) * 1.12))
    distance_ax.legend(frameon=False, loc="upper right", fontsize=9)
    for ax in fig.axes:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fps = 20
    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, str(path), dpi=120):
        # Hold the final frame for two seconds so both outcomes can be read.
        for frame in range(int(np.ceil(duration * fps)) + 1 + 2 * fps):
            clock = min(frame / fps, duration)
            for (_, t, error, outcome), (trail, dot, status, line) in zip(flights, artists):
                k = t.searchsorted(clock + 1e-9, side="right") - 1
                trail.set_data(error[:k+1, 0], error[:k+1, 1])
                dot.set_data([error[k, 0]], [error[k, 1]])
                dist = np.linalg.norm(error[:k+1], axis=1)
                line.set_data(t[:k+1], dist)
                state = outcome if k == len(t) - 1 else "Flying"
                status.set_text(f"{state}   ·   {t[k]:.2f} s   ·   {dist[-1]:.2f} m")
            writer.grab_frame()
    plt.close(fig)


def write_validation(run, cfg, path):
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.8), sharex=True, layout="constrained")
    fig.suptitle("Native PPO-LSTM on fixed validation flights", weight="bold")
    peak_crash_pct = 0.0
    for i, csv in enumerate(sorted(run.parent.glob("seed_*/val_history.csv"))):
        h = pd.read_csv(csv)
        peak_crash_pct = max(peak_crash_pct, 100 * h.crash_rate.max())
        best = h.rmse_capped.idxmin()
        x = h.timesteps / 1e6
        for ax, y in zip(axes, (h.rmse_capped * 100, h.crash_rate * 100)):
            color = COLORS[i % len(COLORS)]
            ax.plot(x, y, "o-", color=color, ms=4, label=csv.parent.name.replace("_", " "))
            ax.plot(x[best], y[best], "*", color=color, ms=15, mec="black", mew=.5)
    n = cfg["eval"].get("n") or SUITE_SIZES[cfg["eval"]["suite"]]
    axes[0].set_title(f"{n} episodes per check   ·   star marks the selected checkpoint for each seed", fontsize=10)
    axes[0].set(ylabel="Mean capped RMSE (cm, log scale)", yscale="log")
    axes[0].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].legend()
    top = max(1.0, peak_crash_pct)
    axes[1].set(xlabel="Training transitions (millions)", ylabel="Crashed episodes (%)",
                ylim=(-0.08 * top, 1.10 * top))
    axes[0].text(.99, .96, f"Crash penalty: {EnvConfig.from_dict(cfg['env']).max_dist:g} m",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=9)
    for ax in axes:
        ax.grid(alpha=.22, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=ROOT / "runs/native_defense/rppo_hover/seed_0")
    ap.add_argument("--out", type=Path, default=ROOT / "results/native_defense")
    args = ap.parse_args()
    torch.set_num_threads(1)
    flights, rows, condition, cfg = paired_flights(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    write_video(flights, condition, rows[0]["seed_train"], args.out / "before_after.mp4")
    write_validation(args.run, cfg, args.out / "validation.png")
    pd.DataFrame(rows).to_csv(args.out / "paired_flights.csv", index=False)
    print(f"Condition seed {condition.seed}; outputs in {args.out}")


if __name__ == "__main__":
    main()
