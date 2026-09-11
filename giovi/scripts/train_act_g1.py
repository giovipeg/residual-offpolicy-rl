#!/usr/bin/env python3
"""BC/ACT training on CubeToContainerG1, with the README's training parameters.

The G1 counterpart of `train_act.py`. Same trainer, same launch path, same
README block -- what differs is the robot: `CubeToContainerG1` is the same scene
driven by a bimanual Unitree G1 with Inspire hands instead of a Panda, so the
observation and action schemas are not the same shape:

              Panda (train_act.py)        G1 (this script)
    action    7                           24  (R arm 6, L arm 6, R hand 6, L hand 6)
    state     9                           38  (per arm: eef pos 3 + quat 4 + hand qpos 12)
    cameras   agentview, eye_in_hand      agentview, eye_in_left_hand, eye_in_right_hand

None of that is configured here -- ACT takes its input/output shapes from the
dataset's own metadata, and the rollout env takes its observation contract from
the task name. Those two agreeing is what `patch_resfit.py` sets up, and what
`verify_dataset_g1.py` checks; if the `CubeToContainerG1` registration is
missing or partial, training still runs and only the success rate tells you.

`--eval_camera_size 84` matches the size the images are stored at, same as the
Panda dataset. `--eval_video_key` stays on `agentview`; the in-hand cameras are
in `rl_camera` for the residual stage but make poor rollout video.

The run id this prints is what `train_td3_g1.py` takes as its base policy. That
one asks W&B for the `_best` artifact, which is only written once a rollout
beats 0.0 success -- so let this run long enough to actually land an episode.

Usage:
    python giovi/scripts/train_act_g1.py
    python giovi/scripts/train_act_g1.py --dry-run
    python giovi/scripts/train_act_g1.py --steps 50000 --eval_num_envs 4
"""

from __future__ import annotations

import argparse
import os
import sys

import _launch
from train_act import drop_overridden

TRAINER = "resfit.lerobot.scripts.train_bc_dexmg"

ARGS = [
    "--dataset", "giovipeg/cube-to-container-g1",
    "--policy", "act",
    "--steps", "200000",
    "--batch_size", "256",
    "--wandb_project", "dexmg-bc",
    "--eval_env", "CubeToContainerG1",
    "--rollout_freq", "5000",
    "--eval_video_key", "observation.images.agentview",
    "--eval_camera_size", "84",
    "--eval_render_size", "224",
    # Half the Panda run's 16. Each eval env renders three 84x84 cameras instead
    # of two and steps a 39-DOF model instead of a 7-DOF one, so the rollouts
    # cost roughly twice as much per env.
    "--eval_num_envs", "8",
    "--eval_num_episodes", "100",
    "--wandb_enable",
    # As on the Panda run, BC here is dataloader-bound rather than GPU-bound:
    # stock LeRobot decodes an mp4 frame per camera on every __getitem__, and
    # this dataset has three cameras. `--cache_in_ram` decodes it once into
    # shared memory instead (giovi/ram_cache.py, which keys off meta.video_keys
    # and so needs no change for the extra camera).
    "--cache_in_ram",
]

# Same derivation as train_act.py: leave a few cores for the main process and
# the eval envs, and tune once per machine with `tune_num_workers.py`.
_cpus = os.cpu_count() or 4
ARGS += ["--num_workers", str(max(2, min(16, _cpus - 4)))]


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
