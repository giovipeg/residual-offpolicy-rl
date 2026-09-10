#!/usr/bin/env python3
"""Residual TD3 on CubeToContainer, with the README's training parameters.

Same launch path as `smoke_train_td3.py` -- `train_residual_td3` has no CLI flag
for the base policy, so `ensure_task_config.ensure()` generates the missing
CubeToContainer config class from giovi/configs/residual_td3_tasks.json, writes
the BC run id into it, and hands back the Hydra config name. What differs is the
overrides: this script passes the "Residual RL training" block of README.md
rather than the cut-down smoke values, so it is a real run (300k steps, 300k
buffer, 10k warmup) and not a 200-step one.

Two of the README's overrides are deliberately absent:

  * `--config-name`: comes from `ensure_task_config`, which derives it from the
    task name (the README's is the ResFiT-shipped coffee one).
  * `wandb.project`: the generated task config already carries the project from
    the task table, so setting it here would be a second place to keep in sync.

The base policy is unchanged from the smoke script's contract -- the *BC* run,
in project `dexmg-bc`, here the one `train_act.py` produces -- but this one uses
the config's default `wt_type=best`, so it needs a BC run that actually beat 0.0
success at an eval rollout and logged a `_best` artifact.

Usage:
    python giovi/scripts/train_td3.py dexmg-bc/<run_id>
    python giovi/scripts/train_td3.py dexmg-bc/<run_id> --dry-run
    python giovi/scripts/train_td3.py dexmg-bc/<run_id> algo.total_timesteps=500_000

For bc policies with a low num_steps use:
python giovi/scripts/train_td3.py dexmg-bc/09anv9tx base_policy.wt_type=latest
"""

from __future__ import annotations

import argparse
import re
import sys

import _launch
import ensure_task_config

TRAINER = "resfit.rl_finetuning.scripts.train_residual_td3"
TASK = "CubeToContainer"

# The README's "Residual RL training" command, verbatim apart from the two
# task-specific flags noted in the module docstring.
OVERRIDES = [
    "algo.prefetch_batches=4",
    "algo.n_step=5",
    "algo.gamma=0.995",
    "algo.learning_starts=10_000",
    "algo.critic_warmup_steps=10_000",
    "algo.num_updates_per_iteration=4",
    "algo.stddev_max=0.025",
    "algo.stddev_min=0.025",
    "algo.buffer_size=300_000",
    "agent.actor.action_scale=0.2",
    "agent.actor_lr=1e-6",
    "wandb.name=resfit",
    "wandb.group=resfit",
    "debug=false",
]


def base_policy_id(value: str) -> str:
    """`download_policy_from_wandb` does `run_id.split("/")` -- reject anything else here."""
    if not re.fullmatch(r"[^/]+/[^/]+", value):
        raise argparse.ArgumentTypeError(f"expected '<wandb_project>/<run_id>', got '{value}'")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "base_policy",
        type=base_policy_id,
        metavar="PROJECT/RUN_ID",
        help="W&B id of the BC run to use as the base policy, e.g. dexmg-bc/xihhienp",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the trainer command without running it")
    ap.epilog = "Any other argument is forwarded as a Hydra override, where it wins over the README defaults."
    # parse_known_args so Hydra overrides can be passed straight through, in any
    # position, without a `--` separator.
    args, extra = ap.parse_known_args()

    # Create the task config if ResFiT has none, and point it at this BC run.
    # Idempotent: rerunning with the same id changes nothing.
    config_name = ensure_task_config.ensure(TASK, base_policy=args.base_policy)

    trainer_args = [f"--config-name={config_name}", *OVERRIDES, *extra]
    return _launch.run(TRAINER, trainer_args, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
