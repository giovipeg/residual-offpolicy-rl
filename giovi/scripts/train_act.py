#!/usr/bin/env python3
"""BC/ACT training on CubeToContainer, with the README's training parameters.

Same launch path as `smoke_train_act.py` -- `_launch.run` puts the trainer in a
subprocess with `MUJOCO_GL` and `simple_env` on `PYTHONPATH` -- but with the
"BC policy training" block of README.md instead of the 20-step smoke values, so
this is a real run (200k steps, batch 256, 100 eval episodes) rather than a
sanity check.

The README's command is for TwoArmCoffee; four of its flags are task-specific
and take CubeToContainer's values here, from giovi/configs/residual_td3_tasks.json
and the dataset's own metadata: `--dataset`, `--eval_env`, `--eval_video_key`
and `--eval_camera_size` (84, the size the cube-to-container images are stored
at; the trainer's default happens to agree, but it is the one flag it asks you
to match to the dataset, so it is spelled out).

The run id this prints is what `train_td3.py` takes as its base policy. Unlike
the smoke pair, that one asks W&B for the `_best` artifact, which is only
written once a rollout beats 0.0 success -- so let this run long enough to
actually land an episode.

Usage:
    python giovi/scripts/train_act.py
    python giovi/scripts/train_act.py --dry-run
    python giovi/scripts/train_act.py --steps 50000 --eval_num_envs 8
"""

from __future__ import annotations

import argparse
import os
import sys

import _launch

TRAINER = "resfit.lerobot.scripts.train_bc_dexmg"

# The README's "BC policy training" command, with the task-specific flags noted
# in the module docstring pointed at CubeToContainer.
ARGS = [
    "--dataset", "giovipeg/cube-to-container",
    "--policy", "act",
    "--steps", "200000",
    "--batch_size", "256",
    "--wandb_project", "dexmg-bc",
    "--eval_env", "CubeToContainer",
    "--rollout_freq", "5000",
    "--eval_video_key", "observation.images.agentview",
    "--eval_camera_size", "84",
    "--eval_render_size", "224",
    "--eval_num_envs", "16",
    "--eval_num_episodes", "100",
    "--wandb_enable",
    # BC training on this dataset is dataloader-bound, not GPU-bound: stock
    # LeRobot decodes an mp4 frame per camera on every __getitem__, so a 200k
    # step run at batch 256 decodes ~100M frames out of a dataset that is only
    # 705 MiB once decoded. `--cache_in_ram` decodes it once into shared memory
    # instead and serves bit-identical items (giovi/ram_cache.py).
    "--cache_in_ram",
]

# Worker count is the one setting that does not transfer between machines, so
# derive it rather than baking in a number: leave a few cores for the main
# process and the eval envs. Override with `--num_workers N` -- worth tuning
# once on the training server with `tune_num_workers.py`, since too many
# workers oversubscribes the decode/collate threads and gets *slower*.
_cpus = os.cpu_count() or 4
ARGS += ["--num_workers", str(max(2, min(16, _cpus - 4)))]


def drop_overridden(base: list[str], extra: list[str]) -> list[str]:
    """Remove flags from `base` that the caller also passed in `extra`.

    argparse takes the last occurrence, so the override already won without
    this; dropping the default just keeps the printed command readable rather
    than showing `--num_workers 12 --num_workers 8` and leaving the reader to
    work out which one applies.
    """
    overridden = {a.split("=", 1)[0] for a in extra if a.startswith("--")}
    kept: list[str] = []
    i = 0
    while i < len(base):
        # A flag owns every following token until the next `--flag`, which
        # covers both `--steps 200000` and bare switches like `--wandb_enable`.
        j = i + 1
        while j < len(base) and not base[j].startswith("--"):
            j += 1
        if base[i] not in overridden:
            kept.extend(base[i:j])
        i = j
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the trainer command without running it")
    ap.epilog = "Any other flag is forwarded to the trainer, where it overrides the README default."
    # parse_known_args so trainer flags can be passed straight through, in any
    # position, without a `--` separator.
    args, extra = ap.parse_known_args()

    return _launch.run(TRAINER, [*drop_overridden(ARGS, extra), *extra], dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
