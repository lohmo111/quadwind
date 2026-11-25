"""Обучение одного метода с одним сидом.

Пример:
    python scripts/train.py --config configs/rppo.yaml --seed 0
    python scripts/train.py --config configs/ppo_mlp.yaml --seed 1 --total-steps 2000000 --exp pilot
Результаты сохраняются в runs/<exp>/<name>/seed_<k>/. У собственного RPPO есть
снимки initial/early/best/final, журнал обновлений train_history.csv и оценки val_history.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from quadwind.env import EnvConfig  # noqa: E402
from quadwind.policies import ObsNormalizer  # noqa: E402
from quadwind.training import (TrainingMonitor, ValidationCallback, build_space, load_config, make_env,  # noqa: E402
                               make_model, save_run_meta)


def train(cfg: dict, run_dir: Path, seed: int, device: str) -> tuple[int, float, float]:
    env = make_env(cfg, seed)
    try:
        model = make_model(cfg, env, seed, device=device, tensorboard_log=str(run_dir / "tb"))
        t0 = time.perf_counter()
        if cfg["backend"] == "native":
            from quadwind.native_training import train_native
            best = train_native(model, env, cfg, run_dir, seed)
        else:
            total = int(cfg["total_timesteps"])
            validation = ValidationCallback(
                EnvConfig.from_dict(cfg.get("env")), run_dir, cfg["eval"]["freq"], cfg["eval"]["suite"],
                cfg["eval"].get("n"), space=build_space(cfg),
            )
            curriculum = TrainingMonitor(cfg["curriculum"]["start"], cfg["curriculum"]["ramp_fraction"], total)
            model.learn(total_timesteps=total, callback=[curriculum, validation], tb_log_name="run", progress_bar=False)
            best = validation.best
        elapsed = time.perf_counter() - t0
        model.save(run_dir / "final_model.zip")
        env.save(str(run_dir / "vecnormalize.pkl"))
        if env.norm_obs:
            ObsNormalizer.from_vecnormalize(env).save(run_dir / "final_obs_norm.npz")
        return model.num_timesteps, elapsed, best
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "rppo_hover.yaml"))
    ap.add_argument("--overlay", action="append", default=[], help="оверлей варианта среды (можно несколько)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", default="main", help="имя серии экспериментов (папка в runs/)")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--eval-freq", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2, help="потоки torch на процесс (для параллельных запусков)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", choices=("native", "sb3"),
                    help="Defaults to the project trainer for RPPO and to SB3 for feedforward PPO.")
    ap.add_argument("--name-suffix", default="", help="добавить к имени запуска (например, __10M)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overlay)
    cfg["name"] += args.name_suffix
    cfg["backend"] = args.backend or cfg.get("backend") or ("native" if cfg["algo"] == "rppo" else "sb3")
    if cfg["backend"] == "native" and cfg["algo"] != "rppo":
        ap.error("The native backend implements recurrent PPO; use an rppo configuration")
    if args.total_steps is not None:
        cfg["total_timesteps"] = args.total_steps
    if args.eval_freq is not None:
        cfg["eval"]["freq"] = args.eval_freq
    if min(args.threads, int(cfg["total_timesteps"]), cfg["eval"]["freq"]) < 1:
        ap.error("CPU threads, training steps and validation frequency must be positive")
    torch.set_num_threads(args.threads)

    run_dir = ROOT / "runs" / args.exp / cfg["name"] / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    saved_cfg = run_dir / "config.yaml"
    if saved_cfg.is_file():
        old_backend = yaml.safe_load(saved_cfg.read_text(encoding="utf-8")).get("backend", "sb3")
        if old_backend != cfg["backend"]:
            ap.error(f"{run_dir} belongs to the {old_backend} trainer. Use a different --exp for {cfg['backend']}.")
    run_info = {"seed": args.seed, "backend": cfg["backend"], "device": args.device, "torch_threads": args.threads}
    save_run_meta(run_dir, cfg, {**run_info, "status": "running"})

    try:
        actual, elapsed, best = train(cfg, run_dir, args.seed, args.device)
    except (Exception, KeyboardInterrupt) as exc:
        save_run_meta(run_dir, cfg, {**run_info, "status": "failed", "error": repr(exc)})
        raise
    save_run_meta(run_dir, cfg, {**run_info, "status": "done", "train_seconds": round(elapsed, 1),
                                 "actual_timesteps": actual, "steps_per_second": round(actual / elapsed, 1),
                                 "best_val_rmse_capped": best})
    print(f"[{cfg['name']} seed {args.seed}] {actual:,} steps in {elapsed / 60:.1f} min "
          f"({actual / elapsed:,.0f} steps/s); best val rmse_capped = {best:.4f}; -> {run_dir}")


if __name__ == "__main__":
    main()
