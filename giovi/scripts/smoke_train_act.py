#!/usr/bin/env python3
"""Smoke test for BC/ACT training on CubeToContainer.

Trains an ACT policy from scratch for 20 steps and logs it to the `dexmg-bc`
W&B project. The run id it prints is what `smoke_train_td3.py` takes as its
base policy.

Usage:
    python giovi/scripts/smoke_train_act.py
    python giovi/scripts/smoke_train_act.py --dry-run
    python giovi/scripts/smoke_train_act.py --steps 100 --batch_size 8
"""

from __future__ import annotations

import argparse
import sys

import _launch

TRAINER = "resfit.lerobot.scripts.train_bc_dexmg"

ARGS = [
    "--dataset", "giovipeg/cube-to-container",
    "--policy", "act",
    "--steps", "20",
    "--batch_size", "4",
    "--wandb_project", "dexmg-bc",
    "--eval_env", "CubeToContainer",
    "--rollout_freq", "10",
    "--eval_video_key", "observation.images.agentview",
    "--eval_camera_size", "84",
    "--eval_render_size", "84",
    "--eval_num_envs", "1",
    "--eval_num_episodes", "1",
    "--wandb_enable",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the trainer command without running it")
    ap.epilog = "Any other flag is forwarded to the trainer, where it overrides the smoke default."
    # parse_known_args so trainer flags can be passed straight through, in any
    # position, without a `--` separator.
    args, extra = ap.parse_known_args()

    return _launch.run(TRAINER, [*ARGS, *extra], dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
