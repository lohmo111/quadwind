"""Build the Moodle ZIP from the tested project files and the team's chosen video."""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path("runs/e7/rppo__pi__safe__att__asym__hp_epochs5__hp_lstm128")


def project_files():
    yield from map(Path, (
        "README.md", "requirements.txt", "requirements-notebook.txt", "pytest.ini", "start_demo.cmd",
        "presentation/RPPO_final_EN.pptx", "presentation/defense_ru.md",
        "notebook/rppo_one_update.ipynb", "docs/rppo_one_update_ru.html", "docs/TEAM_DEFENSE_PLAN.md",
        "results/defense/episodes.csv", "results/defense/summary.csv",
        "configs/base.yaml", "configs/rppo_hover.yaml", "configs/rppo.yaml",
        "configs/ppo_mlp.yaml", "configs/ppo_stack16.yaml", "configs/ppo_oracle.yaml",
        "configs/e7/v4_all.yaml", "configs/pid_tuned.json", "configs/pid_tuned_pi.json",
        "configs/pid_tuned_pi_att.json",
    ))
    for pattern in ("quadwind/*.py", "scripts/*.py", "tests/*.py"):
        for file in sorted(ROOT.glob(pattern)):
            yield file.relative_to(ROOT)
    for seed in range(3):
        for name in ("best_model.zip", "best_obs_norm.npz", "config.yaml", "meta.json", "val_history.csv"):
            yield MODELS / f"seed_{seed}" / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "demo/hover.mp4")
    parser.add_argument("--out", type=Path, default=ROOT / "submission/Quadwind_RPPO.zip")
    args = parser.parse_args()
    video = args.video.resolve()
    if not video.is_file() or video.stat().st_size == 0 or video.suffix.lower() != ".mp4":
        parser.error("--video needs an existing, nonempty MP4 file")
    entries = [(ROOT / path, path.as_posix()) for path in project_files()]
    entries.append((video, "demo/hover.mp4"))
    for source, _ in entries:
        if not source.is_file():
            parser.error(f"missing submission file: {source}")
        if source.resolve() == args.out.resolve():
            parser.error("output overlaps a source file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.out.parent, suffix=".zip", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for source, name in entries:
                archive.write(source, name)
        with ZipFile(temporary) as archive:
            damaged = archive.testzip()
            if damaged:
                raise ValueError(f"CRC verification failed for {damaged}")
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{args.out} ({len(entries)} files, {args.out.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
