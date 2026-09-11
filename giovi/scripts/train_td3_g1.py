#!/usr/bin/env python3
"""Residual TD3 on CubeToContainerG1, with the README's training parameters.

The G1 counterpart of `train_td3.py`, and structurally identical to it: the
trainer has no CLI flag for the base policy, so `ensure_task_config.ensure()`
generates the `CubeToContainerG1` config class from
giovi/configs/residual_td3_tasks.json, writes the BC run id into it, and hands
back the Hydra config name.

The task table entry is what carries everything G1-specific -- the dataset, and
the three `rl_camera` keys the residual's encoder reads. The action and state
dimensions are not configured anywhere: the residual actor is sized from the
live env (24-D here, against the Panda task's 7-D), and both normalisers are
rebuilt from the dataset's own statistics.

The base policy is the *BC* run in project `dexmg-bc`, here the one
`train_act_g1.py` produces. This uses the config's default `wt_type=best`, so it
needs a BC run that actually beat 0.0 success at an eval rollout and logged a
`_best` artifact.

One thing worth watching: the G1 expert parks the left arm at zero deltas, so 12
of the 24 action dimensions are constant across the whole dataset. The base
policy learns to output ~0 there, but the residual explores every dimension, and
at `action_scale=0.2` it will actively drive the left arm and hand. The
`min_action_range` floor keeps the normalisation from dividing by zero, so this
will not crash -- but if residual success stalls below the base policy, masking
the residual to the right-arm dimensions is the first thing to try.

Usage:
    python giovi/scripts/train_td3_g1.py dexmg-bc/<run_id>
    python giovi/scripts/train_td3_g1.py dexmg-bc/<run_id> --dry-run
    python giovi/scripts/train_td3_g1.py dexmg-bc/<run_id> algo.total_timesteps=500_000

For bc policies with a low num_steps use:
python giovi/scripts/train_td3_g1.py dexmg-bc/<run_id> base_policy.wt_type=latest
"""

from __future__ import annotations

import argparse
import sys

import _launch
import ensure_task_config
from train_td3 import base_policy_id

TRAINER = "resfit.rl_finetuning.scripts.train_residual_td3"
TASK = "CubeToContainerG1"

# The README's "Residual RL training" block, unchanged from the Panda run --
# these are the paper's reported optima for sparse-reward tasks and are not
# robot-specific. Kept as its own list rather than imported from train_td3 so
# that tuning one task cannot silently retune the other.
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
