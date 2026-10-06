"""Обучение одного метода с одним сидом.

Пример:
    python scripts/train.py --config configs/rppo.yaml --seed 0
    python scripts/train.py --config configs/ppo_mlp.yaml --seed 1 --total-steps 2000000 --exp pilot
Результат: runs/<exp>/<name>/seed_<k>/ (config.yaml, meta.json, best/final модели и нормализация,
val_history.csv, логи TensorBoard в tb/).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from quadwind.env import EnvConfig  # noqa: E402
from quadwind.policies import ObsNormalizer  # noqa: E402
from quadwind.training import (TrainingMonitor, ValidationCallback, build_space, load_config, make_env,  # noqa: E402
                               make_model, save_run_meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overlay", action="append", default=[], help="оверлей варианта среды (можно несколько)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", default="main", help="имя серии экспериментов (папка в runs/)")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--eval-freq", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2, help="потоки torch на процесс (для параллельных запусков)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name-suffix", default="", help="добавить к имени запуска (например, __10M)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overlay)
    cfg["name"] += args.name_suffix
    if args.total_steps:
        cfg["total_timesteps"] = args.total_steps
    if args.eval_freq:
        cfg["eval"]["freq"] = args.eval_freq
    torch.set_num_threads(args.threads)

    run_dir = ROOT / "runs" / args.exp / cfg["name"] / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_meta(run_dir, cfg, {"seed": args.seed, "status": "running"})

    env = make_env(cfg, args.seed)
    model = make_model(cfg, env, args.seed, device=args.device, tensorboard_log=str(run_dir / "tb"))
    total = int(cfg["total_timesteps"])
    callbacks = [
        TrainingMonitor(cfg["curriculum"]["start"], cfg["curriculum"]["ramp_fraction"], total),
        ValidationCallback(EnvConfig.from_dict(cfg.get("env")), run_dir, cfg["eval"]["freq"], cfg["eval"]["suite"],
                           cfg["eval"].get("n"), space=build_space(cfg)),
    ]
    t0 = time.perf_counter()
    model.learn(total_timesteps=total, callback=callbacks, tb_log_name="run", progress_bar=False)
    elapsed = time.perf_counter() - t0

    model.save(run_dir / "final_model.zip")
    env.save(str(run_dir / "vecnormalize.pkl"))
    if env.norm_obs:
        ObsNormalizer.from_vecnormalize(env).save(run_dir / "final_obs_norm.npz")
    best = callbacks[1].best
    save_run_meta(run_dir, cfg, {"seed": args.seed, "status": "done", "train_seconds": round(elapsed, 1),
                                 "steps_per_second": round(total / elapsed, 1), "best_val_rmse_capped": best})
    print(f"[{cfg['name']} seed {args.seed}] {total:,} steps in {elapsed / 60:.1f} min "
          f"({total / elapsed:,.0f} steps/s); best val rmse_capped = {best:.4f}; -> {run_dir}")


if __name__ == "__main__":
    main()
