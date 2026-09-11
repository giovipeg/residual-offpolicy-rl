#!/usr/bin/env python3
"""A short residual-TD3 run on CubeToContainer -- between the smoke test and the real one.

`smoke_train_td3.py` exists to prove the launch path works: 200 steps, batch of
4, two offline episodes. It finishes in minutes and the policy it produces is
noise. `train_td3.py` is the README's real run: 300k steps, and hours upon hours.
This script sits in between, at 5k steps, so it returns an actual
trained-a-bit policy without costing a full training budget.

The split it makes is between the two kinds of override:

  * the ones that *shape* the policy -- n_step, gamma, the stddev schedule,
    action_scale, actor_lr, updates per iteration -- are taken verbatim from
    `train_td3.py`. A run at different exploration noise or a different residual
    action scale is not a preview of the real one, it is a different experiment.
    One of them is spelled differently here: `train_td3.py` sets the exploration
    noise with `algo.stddev_max/min`, which the trainer never reads, so its runs
    silently explore at the 0.05 default rather than the 0.025 they ask for. See
    the comment on the override below -- this script sets the schedule that is
    actually read, so its noise really is the intended 0.025.
  * the ones that are just *budget* -- total_timesteps, buffer_size,
    learning_starts, critic_warmup_steps, eval cadence -- are scaled down.

So the numbers that differ from the real run are only the ones the run length
forces, and everything a shorter run can afford to keep is kept: the default
batch size of 256 (the smoke's 4 is a memory concession, not a choice) and the
task config's full 200 offline episodes, which half of every batch is drawn from
at the default `offline_fraction=0.5`.

                          smoke      short (here)   train_td3.py
    total_timesteps          200           5_000         300_000
    learning_starts           50           2_000          10_000
    critic_warmup_steps       50           2_000          10_000
    buffer_size            1_000          10_000         300_000
    batch_size                 4             256             256
    offline episodes           2             200             200
    eval every               100           1_000          10_000

Note what the first three rows mean together: the two warmup numbers do not
scale with the run, because they are a prefix the step counter never sees. A 5k
run still pays 2k random env steps and 2k critic-only updates up front.

The launch path is unchanged from the other two: `train_residual_td3` has no CLI
flag for the base policy, so `ensure_task_config.ensure()` generates the
CubeToContainer config class from giovi/configs/residual_td3_tasks.json, writes
the BC run id into it, and hands back the Hydra config name.

The argument is the *BC* run, in project `dexmg-bc` -- the one `train_act.py`
produces. Use a real one: with the shape parameters held at the real run's
values, the quality of this run is now the base policy's to lose.

Usage:
    python giovi/scripts/short_train_td3.py dexmg-bc/<run_id>
    python giovi/scripts/short_train_td3.py dexmg-bc/<run_id> --dry-run
    # dial the length without touching the shape -- keep the buffer above
    # learning_starts + total_timesteps so nothing is evicted:
    python giovi/scripts/short_train_td3.py dexmg-bc/<run_id> \
        algo.total_timesteps=30_000 algo.buffer_size=50_000
"""

from __future__ import annotations

import argparse
import re
import sys

import _launch
import ensure_task_config

TRAINER = "resfit.rl_finetuning.scripts.train_residual_td3"
TASK = "CubeToContainer"

OVERRIDES = [
    # --- base policy ----------------------------------------------------
    # Same reason as in the smoke script: `train_bc_dexmg` only logs a `_best`
    # artifact once a rollout beats `best_success_rate`, which starts at 0.0 and
    # is compared with a strict `>`, so a BC run that never scored a success has
    # no `_best` to ask for. `_latest` is written at every checkpoint. Pass
    # `base_policy.wt_type=best` when the BC run did log one -- it wins over
    # this, and it is the better weight to start from.
    "base_policy.wt_type=latest",
    # --- shape: verbatim from train_td3.py ------------------------------
    "algo.prefetch_batches=4",
    "algo.n_step=5",
    "algo.gamma=0.995",
    "algo.num_updates_per_iteration=4",
    # The README's `algo.stddev_max=0.025 algo.stddev_min=0.025` does NOT work,
    # here or in train_td3.py. The trainer reads `algo.stddev_schedule`, and that
    # is a `field(init=False)` built by `RLPDAlgoConfig.__post_init__` -- which
    # runs when the dataclass is instantiated, i.e. before any CLI override is
    # applied, and never again. Setting the endpoints from the command line
    # leaves the schedule at the 0.05 default and is silently ignored: the run
    # explores at double the noise it claims to. Set the schedule itself. The
    # inner quotes are required, or Hydra's override grammar tries to call a
    # function named `linear`.
    'algo.stddev_schedule="linear(0.025,0.025,300000)"',
    "agent.actor.action_scale=0.2",
    "agent.actor_lr=1e-6",
    # --- budget ---------------------------------------------------------
    "algo.total_timesteps=5_000",
    # Deliberately NOT scaled with total_timesteps. Neither of these spends the
    # training budget -- both run to completion before the step counter starts,
    # so they are a fixed prefix, not a fraction. They are also the two that
    # decide whether the actor starts on a critic worth trusting, which is how a
    # residual run diverges early. 2k is the floor worth using; raise it before
    # lowering it. The cost is that the prefix is now about as long as the run
    # itself, so this is proportionally dearer per training step than a longer
    # one -- that is the price of a 5k run, not a reason to cut the warmup.
    "algo.learning_starts=2_000",
    "algo.critic_warmup_steps=2_000",
    # Online buffer only -- the offline episodes live in their own. Only needs
    # to clear learning_starts + total_timesteps (7k) for this run to never
    # evict, as the real one does not. At ~84x84x2 cameras the buffer is most of
    # this script's host RAM, so there is no reason to hold the slack.
    "algo.buffer_size=10_000",
    # --- evaluation -----------------------------------------------------
    # The eval pass is also what writes `last.pt` (and `best.pt` on an
    # improvement), so this interval is the checkpoint interval too. Still six
    # evals over the run, on enough episodes that the success rate means
    # something -- below ~20 the number is too noisy to read. At 5k steps those
    # six cost a visible share of wall clock, since eval is pure env stepping
    # with no gradient work to hide behind; raise the interval before cutting
    # the episode count, or you lose the signal rather than the cost.
    "eval_interval_every_steps=1_000",
    "eval_num_envs=4",
    "eval_num_episodes=20",
    # --- logging --------------------------------------------------------
    # No `wandb.project`: the generated task config carries it, and logging
    # beside the real runs is the point -- the curves are only worth anything
    # next to theirs. The group is what tells them apart.
    "wandb.name=short",
    "wandb.group=short",
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
    ap.epilog = "Any other argument is forwarded as a Hydra override, where it wins over the defaults above."
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
