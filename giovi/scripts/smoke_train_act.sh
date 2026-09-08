#!/bin/bash

# Make executable with: chmod +x run_thing.sh
# Then run with: ./run_thing.sh

# `import simple_env` (registers the CubeToContainer robosuite task) must resolve.
# Offscreen MuJoCo rendering for the eval rollouts.
export MUJOCO_GL=egl

export PYTHONPATH=/home/giovi/giovi/talos/simple_robosuite_env:$PYTHONPATH

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset giovipeg/cube-to-container \
    --policy act \
    --steps 20 \
    --batch_size 4 \
    --wandb_project dexmg-bc \
    --eval_env CubeToContainer \
    --rollout_freq 10 \
    --eval_video_key observation.images.agentview \
    --eval_camera_size 84 \
    --eval_render_size 84 \
    --eval_num_envs 1 \
    --eval_num_episodes 1 \
    --wandb_enable
