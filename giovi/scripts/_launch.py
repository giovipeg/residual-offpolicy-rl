"""Shared plumbing for the smoke-test launchers.

Both smoke scripts run a ResFiT trainer in a subprocess rather than importing
it, for two reasons: `MUJOCO_GL` has to be set before MuJoCo is first imported,
and the trainers own `sys.argv` (argparse in one, Hydra in the other). A child
process gets both right without any monkey-patching.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

# giovi/scripts/_launch.py -> the ResFiT repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
# The checkout that provides `import simple_env`, which registers CubeToContainer.
SIMPLE_ENV_ROOT = REPO_ROOT.parent / "simple_robosuite_env"


def trainer_env() -> dict[str, str]:
    """A copy of the environment with what the ResFiT trainers need."""
    env = os.environ.copy()
    # Offscreen MuJoCo rendering for the eval rollouts.
    env["MUJOCO_GL"] = "egl"
    # `import simple_env` (registers the CubeToContainer robosuite task) and
    # `import giovi.ram_cache` (the --cache_in_ram dataset) must both resolve.
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(SIMPLE_ENV_ROOT), str(REPO_ROOT), env.get("PYTHONPATH", "")])
    )
    return env


def run(module: str, args: list[str], *, dry_run: bool = False) -> int:
    """Run `python -m <module> <args>` from the repo root with the trainer env."""
    if not SIMPLE_ENV_ROOT.exists():
        print(f"warning: {SIMPLE_ENV_ROOT} does not exist; `import simple_env` will fail", file=sys.stderr)

    cmd = [sys.executable, "-m", module, *args]
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=REPO_ROOT, env=trainer_env())
