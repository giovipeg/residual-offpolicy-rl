#!/bin/bash
#
# Smoke test for residual TD3 on CubeToContainer.
#
# Make executable with: chmod +x smoke_train_td3.sh
# Then run with:        ./smoke_train_td3.sh <wandb_project>/<run_id>
#
# `train_residual_td3` has no CLI flag for the base policy: per the README, the
# `<wandb_project>/<run_id>` of the BC run belongs in the task's config class in
# resfit/rl_finetuning/config/residual_td3.py -- and ResFiT ships no class for
# CubeToContainer at all. `ensure_task_config.py` handles both: it generates the
# missing class from the task table in giovi/configs/residual_td3_tasks.json and
# writes the run id into it. Nothing in ResFiT has to be edited by hand.
#
# The argument is the *BC* run -- the one `smoke_train_act.sh` produced, whose
# project is `dexmg-bc`.
#
# Note the `base_policy.wt_type=latest` override below. The config default is
# `best`, but train_bc_dexmg only writes a `_best` artifact when a rollout beats
# `best_success_rate`, which starts at 0.0 and is compared with a strict `>`. A
# 20-step smoke ACT scores 0.0, so no `_best` artifact is ever logged and asking
# for one dies with `artifact membership 'run_<id>_best:latest' not found`. The
# `_latest` artifact is written on every checkpoint and has the same `policy/`
# layout, so it is what a smoke run has to use. Drop the override once the base
# policy comes from a real BC run that actually succeeds at an eval episode.

set -euo pipefail

# `import simple_env` (registers the CubeToContainer robosuite task) must resolve.
# Offscreen MuJoCo rendering for the eval rollouts.
export MUJOCO_GL=egl

export PYTHONPATH=/home/giovi/giovi/talos/simple_robosuite_env:$PYTHONPATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK=CubeToContainer

if [ $# -ne 1 ]; then
    echo "usage: $(basename "$0") <wandb_project>/<run_id>" >&2
    echo "       e.g. $(basename "$0") dexmg-bc/xihhienp" >&2
    exit 1
fi

BASE_POLICY="$1"

# `download_policy_from_wandb` does `run_id.split("/")`, so exactly one slash
# and two non-empty halves -- fail here rather than deep inside the W&B API.
if [[ ! "$BASE_POLICY" =~ ^[^/]+/[^/]+$ ]]; then
    echo "error: expected '<wandb_project>/<run_id>', got '$BASE_POLICY'" >&2
    exit 1
fi

# Create the task config if ResFiT has none, and point it at this BC run.
# Idempotent: rerunning with the same id changes nothing.
python "$SCRIPT_DIR/ensure_task_config.py" "$TASK" --base-policy "$BASE_POLICY"

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_cube_to_container_config \
    base_policy.wt_type=latest \
    algo.total_timesteps=200 \
    algo.batch_size=4 \
    algo.buffer_size=1_000 \
    algo.learning_starts=50 \
    algo.critic_warmup_steps=50 \
    algo.num_updates_per_iteration=1 \
    algo.prefetch_batches=0 \
    offline_data.num_episodes=2 \
    eval_interval_every_steps=100 \
    eval_num_envs=1 \
    eval_num_episodes=1 \
    wandb.project=dexmg-cube-to-container-td3 \
    debug=false
