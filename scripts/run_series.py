"""Очередь запусков обучения: конфиги × сиды, не больше --parallel процессов одновременно.

Готовые запуски (есть final_model.zip) пропускаются, поэтому серию можно перезапускать после сбоя.
Пример (основная серия на ночь):
    python scripts/run_series.py --exp main --seeds 0 1 2 3 4 --parallel 4 --threads 3 \
        --configs configs/ppo_mlp.yaml configs/ppo_stack16.yaml configs/ppo_oracle.yaml configs/rppo.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadwind.training import load_config  # noqa: E402


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
            if (ROOT / "runs" / args.exp / name / f"seed_{seed}" / "final_model.zip").exists():
                print(f"skip {name} seed {seed}: done")
                continue
            queue.append((config, overlays, name, int(seed), steps))
        args.seeds = []
    for seed in args.seeds:                      # сначала все методы с сидом 0, потом с сидом 1, …
        for overlay in args.overlays:
            for config in args.configs:
                overlays = [] if overlay == "-" else overlay.split("+")      # «a.yaml+b.yaml» — несколько оверлеев
                name = load_config(config, overlays)["name"]
                if (ROOT / "runs" / args.exp / name / f"seed_{seed}" / "final_model.zip").exists():
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
                   "--exp", args.exp, "--threads", str(args.threads)]
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
