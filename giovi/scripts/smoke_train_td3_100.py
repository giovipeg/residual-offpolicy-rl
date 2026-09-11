#!/usr/bin/env python3
"""Smoke test for residual TD3 on CubeToContainer -- 100-step variant.

Identical to `smoke_train_td3.py` except `algo.total_timesteps=100` instead of
200. With `eval_interval_every_steps=100` the run still evaluates (and writes
`last.pt`) exactly once, at its final step.

`train_residual_td3` has no CLI flag for the base policy: per the README, the
`<wandb_project>/<run_id>` of the BC run belongs in the task's config class in
resfit/rl_finetuning/config/residual_td3.py -- and ResFiT ships no class for
CubeToContainer at all. This script calls `ensure_task_config.ensure()`, which
generates the missing class from giovi/configs/residual_td3_tasks.json, writes
the run id into it, and hands back the Hydra config name to launch with. So the
whole launch is one command and nothing in ResFiT is edited by hand.

The argument is the *BC* run -- the one `smoke_train_act.py` produced, whose
project is `dexmg-bc`.

Usage:
    python giovi/scripts/smoke_train_td3_100.py dexmg-bc/<run_id>
    python giovi/scripts/smoke_train_td3_100.py dexmg-bc/<run_id> --dry-run
    python giovi/scripts/smoke_train_td3_100.py dexmg-bc/<run_id> algo.total_timesteps=1000
"""

from __future__ import annotations

import argparse
import re
import sys

import _launch
import ensure_task_config

TRAINER = "resfit.rl_finetuning.scripts.train_residual_td3"
TASK = "CubeToContainer"

# `base_policy.wt_type=latest`: the config default is `best`, but train_bc_dexmg
# only writes a `_best` artifact when a rollout beats `best_success_rate`, which
# starts at 0.0 and is compared with a strict `>`. A 20-step smoke ACT scores
# 0.0, so no `_best` artifact is ever logged and asking for one dies with
# `artifact membership 'run_<id>_best:latest' not found`. The `_latest` artifact
# is written on every checkpoint and has the same `policy/` layout, so it is
# what a smoke run has to use. Drop this once the base policy comes from a real
# BC run that actually succeeds at an eval episode.
OVERRIDES = [
    "base_policy.wt_type=latest",
    "algo.total_timesteps=800",
    "algo.batch_size=4",
    "algo.buffer_size=1_000",
    "algo.learning_starts=50",
    "algo.critic_warmup_steps=50",
    "algo.num_updates_per_iteration=1",
    "algo.prefetch_batches=0",
    "offline_data.num_episodes=2",
    "eval_interval_every_steps=100",
    "eval_num_envs=1",
    "eval_num_episodes=1",
    "wandb.project=dexmg-cube-to-container-td3",
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
    ap.epilog = "Any other argument is forwarded as a Hydra override, where it wins over the smoke defaults."
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
