"""Очередь запусков обучения: конфиги × сиды, не больше --parallel процессов одновременно.

Завершённые запуски с чекпоинтом пропускаются, включая архивные серии с сохранённым best.
Пример (основная серия на ночь):
    python scripts/run_series.py --exp main --seeds 0 1 2 3 4 --parallel 4 --threads 3 \
        --configs configs/ppo_mlp.yaml configs/ppo_stack16.yaml configs/ppo_oracle.yaml configs/rppo.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadwind.training import load_config  # noqa: E402


def finished_run(folder: Path, backend: str) -> bool:
    config_path = folder / "config.yaml"
    if not config_path.is_file():
        return False
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    saved_backend = cfg.get("backend", "sb3")
    if saved_backend != backend:
        raise ValueError(f"{folder} contains a {saved_backend} run. Choose another --exp for {backend}.")
    meta_path = folder / "meta.json"
    if meta_path.is_file() and json.loads(meta_path.read_text(encoding="utf-8")).get("status") != "done":
        return False
    # The released e7 runs retain their selected weights and normalization.
    names = ("final", "best") if meta_path.is_file() else ("final",)
    return any((folder / f"{name}_model.zip").is_file() and (
        not cfg.get("normalize", {}).get("obs") or (folder / f"{name}_obs_norm.npz").is_file()
    ) for name in names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main")
    ap.add_argument("--configs", nargs="+", default=[])
    ap.add_argument("--plan", default=None, help="файл очереди: «конфиг | оверлеи | сид» на строку")
    ap.add_argument("--overlays", nargs="*", default=["-"],
                    help="варианты среды; '-' — без оверлея; 'a.yaml+b.yaml' — два оверлея в одном запуске")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--backend", choices=("sb3", "native"), default="sb3",
                    help="The original experiment queues used SB3 and SB3-Contrib.")
    ap.add_argument("--total-steps", type=int, default=None)
    args = ap.parse_args()
    if args.parallel < 1 or args.threads < 1:
        ap.error("--parallel and --threads must both be positive")

    queue = []
    if args.plan:                                # строки «конфиг | оверлеи через + или - | сид»
        for line in Path(args.plan).read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split("|")]
            config, overlay, seed = parts[:3]
            steps = int(float(parts[3])) if len(parts) > 3 else None     # необязательный 4-й столбец — шаги
            overlays = [] if overlay == "-" else overlay.split("+")
            name = load_config(config, overlays)["name"] + (f"__{steps // 1_000_000}M" if steps else "")
            if finished_run(ROOT / "runs" / args.exp / name / f"seed_{seed}", args.backend):
                print(f"skip {name} seed {seed}: done")
                continue
            queue.append((config, overlays, name, int(seed), steps))
        args.seeds = []
    for seed in args.seeds:                      # сначала все методы с сидом 0, потом с сидом 1, …
        for overlay in args.overlays:
            for config in args.configs:
                overlays = [] if overlay == "-" else overlay.split("+")      # «a.yaml+b.yaml» — несколько оверлеев
                name = load_config(config, overlays)["name"]
                if finished_run(ROOT / "runs" / args.exp / name / f"seed_{seed}", args.backend):
                    print(f"skip {name} seed {seed}: done")
                    continue
                queue.append((config, overlays, name, seed, None))
    (ROOT / "logs").mkdir(exist_ok=True)
    running: list[tuple[subprocess.Popen, str]] = []
    failed = []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < args.parallel:
            config, overlays, name, seed, steps = queue.pop(0)
            cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", config, "--seed", str(seed),
                   "--exp", args.exp, "--threads", str(args.threads), "--backend", args.backend]
            for overlay in overlays:
                cmd += ["--overlay", overlay]
            if steps or args.total_steps:
                cmd += ["--total-steps", str(steps or args.total_steps)]
            if steps:
                cmd += ["--name-suffix", f"__{steps // 1_000_000}M"]
            with open(ROOT / "logs" / f"{args.exp}_{name}_s{seed}.log", "w") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            running.append((proc, f"{name} s{seed}"))
            print(f"[{(time.time() - t0) / 60:6.1f} min] start {name} seed {seed}  (queue: {len(queue)})")
        time.sleep(10)
        for proc, label in list(running):
            if proc.poll() is not None:
                running.remove((proc, label))
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                if proc.returncode:
                    failed.append(label)
                print(f"[{(time.time() - t0) / 60:6.1f} min] {status}: {label}")
    if failed:
        raise SystemExit("Training failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
