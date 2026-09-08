#!/usr/bin/env python3
"""Patch a ResFiT checkout: register `CubeToContainer`, and unbreak replay buffers.

ResFiT hard-codes its task table in two places, and neither is extensible from
the outside:

  * `resfit/dexmg/environments/dexmg.py` -- robot list, episode horizon, the
    module import that registers the robosuite env, and the expected camera /
    low-dim observation keys.
  * `resfit/lerobot/scripts/train_bc_dexmg.py` -- a separate allow-list of
    `--eval_env` names, which rejects anything it does not know about with
    `ValueError: Unknown environment: ...` *before* `dexmg.py` is ever reached.

It also works around a torch/torchrl ABI break in
`resfit/rl_finetuning/scripts/train_residual_td3.py`: torch 2.14 dropped
`torch::autograd::deleteNode`, which no released torchrl wheel (up to 0.13.3)
can link against, so torchrl's compiled extension never loads and every run
dies with `NameError: name 'SumSegmentTreeFp32' is not defined` while building
the replay buffers. See the `_make_replay_buffer` edit below.

This script applies every edit needed to make `--eval_env CubeToContainer`
work, plus that fixup, and nothing else. It is idempotent -- rerunning it is a
no-op -- and writes a `.bak` next to each file the first time it changes it.

Environment-level fixes (torchcodec, FFmpeg) live in `patch_residual.py`.

This script lives inside the ResFiT checkout it patches, so `--resfit-root`
defaults to the repo two directories up from here.

Usage:
    python giovi/scripts/patch_resfit.py
    python giovi/scripts/patch_resfit.py --check
    python giovi/scripts/patch_resfit.py --revert
    python giovi/scripts/patch_resfit.py --resfit-root ~/another-checkout
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ENV_NAME = "CubeToContainer"
HORIZON = 220

# giovi/scripts/patch_resfit.py -> the ResFiT repo root.
DEFAULT_RESFIT_ROOT = Path(__file__).resolve().parents[2]
# The checkout that provides `import simple_env`; only used for the hint printed
# at the end, and overridable with --simple-env-root.
DEFAULT_SIMPLE_ENV_ROOT = DEFAULT_RESFIT_ROOT.parent / "simple_robosuite_env"

# Both key helpers in dexmg.py gate the single-arm Panda observation contract on
# the same set literal, so the same edit lands twice in that file.
SINGLE_ARM_SET_OLD = (
    '        if env_lower in {"lift", "can", "pickplacecan", "square", "nutassemblysquare", "threading"}:'
)
SINGLE_ARM_SET_NEW = (
    '        if env_lower in {"lift", "can", "pickplacecan", "square", "nutassemblysquare", '
    '"threading", "cubetocontainer"}:'
)


@dataclass
class Edit:
    """One textual edit against a ResFiT source file.

    `marker` is the text that is present iff the edit has already been applied,
    which is what makes the whole script idempotent.
    """

    name: str
    op: str  # "insert_after" | "replace"
    anchor: str  # text to find (the inserted-after anchor, or the text to replace)
    text: str = ""  # inserted after `anchor`, or the replacement for it
    occurrences: int = 1  # how many times `anchor` is expected to appear
    marker: str = field(default="")

    def __post_init__(self):
        if not self.marker:
            self.marker = self.text.strip()

    def applied(self, source: str) -> bool:
        return self.marker in source

    def apply(self, source: str) -> str:
        found = source.count(self.anchor)
        if found != self.occurrences:
            raise LookupError(
                f"anchor for '{self.name}' found {found}x, expected {self.occurrences}x"
            )
        if self.op == "insert_after":
            return source.replace(self.anchor, self.anchor + self.text, 1)
        return source.replace(self.anchor, self.text)


EDITS: dict[str, list[Edit]] = {
    "resfit/dexmg/environments/dexmg.py": [
        Edit(
            name="ENV_ROBOTS entry",
            op="insert_after",
            anchor='    # Square task -- implemented in robosuite as `NutAssemblySquare`\n'
            '    "NutAssemblySquare": ["Panda"],\n',
            text="    # ------------------------------------------------------------------\n"
            "    # External tasks (registered by importing their own package)\n"
            "    # ------------------------------------------------------------------\n"
            "    # Cube-to-container pick-and-place -- single Panda arm, from `simple_env`\n"
            f'    "{ENV_NAME}": ["Panda"],\n',
            marker=f'    "{ENV_NAME}": ["Panda"],',
        ),
        Edit(
            name="episode horizon",
            op="insert_after",
            anchor='            "TwoArmCanSortRandom": 400,\n',
            text=f'            "{ENV_NAME}": {HORIZON},\n',
        ),
        Edit(
            name="env module import",
            op="insert_after",
            anchor="            import dexmimicgen  # noqa: F401, PLC0415\n",
            text=f'\n        if env_name == "{ENV_NAME}":\n'
            f"            import simple_env  # noqa: F401, PLC0415  (registers {ENV_NAME})\n",
        ),
        Edit(
            name="image + low-dim keys (single-arm Panda contract, 2 sites)",
            op="replace",
            anchor=SINGLE_ARM_SET_OLD,
            text=SINGLE_ARM_SET_NEW,
            occurrences=2,
            marker='"threading", "cubetocontainer"}',
        ),
    ],
    "resfit/lerobot/scripts/train_bc_dexmg.py": [
        Edit(
            name="--eval_env allow-list",
            op="replace",
            anchor='        mimicgen_envs = [\n'
            '            "Threading",  # Single-arm threading task from MimicGen\n'
            "        ]\n"
            "\n"
            "        envs = dexmimicgen_envs + robomimic_envs + mimicgen_envs\n",
            text='        mimicgen_envs = [\n'
            '            "Threading",  # Single-arm threading task from MimicGen\n'
            "        ]\n"
            "        external_envs = [\n"
            f'            "{ENV_NAME}",  # Single-arm pick-and-place registered by the `simple_env` package\n'
            "        ]\n"
            "\n"
            "        envs = dexmimicgen_envs + robomimic_envs + mimicgen_envs + external_envs\n",
            marker="envs = dexmimicgen_envs + robomimic_envs + mimicgen_envs + external_envs",
        ),
        Edit(
            name="--video_backend flag",
            op="insert_after",
            anchor='parser.add_argument("--num_workers", type=int, default=4)\n',
            text="parser.add_argument(\n"
            '    "--video_backend",\n'
            "    type=str,\n"
            "    default=None,\n"
            '    choices=["torchcodec", "pyav", "video_reader"],\n'
            "    help=\"Video decoder for the LeRobot dataset. Defaults to LeRobot's choice (torchcodec when \"\n"
            "    \"installed); use 'pyav' if libtorchcodec fails to load against the local FFmpeg.\",\n"
            ")\n",
            marker='"--video_backend",',
        ),
        Edit(
            name="pass video_backend to LeRobotDataset",
            op="insert_after",
            anchor="        download_videos=True,\n        image_transforms=image_transforms,\n",
            text="        video_backend=cfg.video_backend,\n",
        ),
    ],
    "resfit/rl_finetuning/scripts/train_residual_td3.py": [
        Edit(
            name="import the non-prioritized buffer",
            op="replace",
            anchor="from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictPrioritizedReplayBuffer\n",
            text="from torchrl.data import (\n"
            "    LazyTensorStorage,\n"
            "    ReplayBuffer,\n"
            "    TensorDictPrioritizedReplayBuffer,\n"
            "    TensorDictReplayBuffer,\n"
            ")\n",
            marker="    TensorDictReplayBuffer,",
        ),
        Edit(
            name="_make_replay_buffer helper (torchrl extension workaround)",
            op="insert_after",
            anchor='if "MUJOCO_EGL_DEVICE_ID" in os.environ:\n'
            '    del os.environ["MUJOCO_EGL_DEVICE_ID"]\n',
            text="\n\n"
            "def _make_replay_buffer(sampling_strategy: str, *, alpha, beta, eps, priority_key, **kwargs):\n"
            "    \"\"\"Build a replay buffer that only needs torchrl's C++ extension for real PER.\n"
            "\n"
            "    torch 2.14 removed `torch::autograd::deleteNode`, a symbol every released\n"
            "    torchrl wheel (through 0.13.3) still links against, so `torchrl._torchrl`\n"
            "    fails to load and `PrioritizedSampler`'s segment trees are missing --\n"
            "    `NameError: name 'SumSegmentTreeFp32' is not defined`.\n"
            "\n"
            "    Under `sampling_strategy=\"uniform\"` this file already passes alpha=beta=0\n"
            "    and skips every `update_tensordict_priority` call, which is plain uniform\n"
            "    sampling, so the non-prioritized buffer is an exact substitute and needs no\n"
            "    extension. `prioritized_replay` genuinely needs the segment trees and still\n"
            "    requires a torchrl built against the installed torch.\n"
            "    \"\"\"\n"
            "    if sampling_strategy == \"prioritized_replay\":\n"
            "        return TensorDictPrioritizedReplayBuffer(\n"
            "            alpha=alpha, beta=beta, eps=eps, priority_key=priority_key, **kwargs\n"
            "        )\n"
            "    return TensorDictReplayBuffer(priority_key=priority_key, **kwargs)\n",
            marker="def _make_replay_buffer(",
        ),
        Edit(
            name="online buffer uses the helper",
            op="replace",
            anchor="    online_rb = TensorDictPrioritizedReplayBuffer(\n",
            text="    online_rb = _make_replay_buffer(\n        cfg.algo.sampling_strategy,\n",
            marker="    online_rb = _make_replay_buffer(",
        ),
        Edit(
            name="offline buffer uses the helper",
            op="replace",
            anchor="    offline_rb = TensorDictPrioritizedReplayBuffer(\n",
            text="    offline_rb = _make_replay_buffer(\n        cfg.algo.sampling_strategy,\n",
            marker="    offline_rb = _make_replay_buffer(",
        ),
    ],
}


def resolve(root: Path, rel: str) -> Path:
    target = root / rel
    if not target.exists():
        sys.exit(f"not found: {target}\nIs --resfit-root really a ResFiT checkout?")
    return target


def do_check(root: Path) -> int:
    missing = 0
    for rel, edits in EDITS.items():
        source = resolve(root, rel).read_text()
        print(rel)
        for edit in edits:
            ok = edit.applied(source)
            missing += not ok
            print(f"  {'+' if ok else '-'} {edit.name}")
    if missing:
        print(f"\n{missing} edit(s) missing -- run without --check to apply.")
        return 1
    print("\nall edits present.")
    return 0


def do_revert(root: Path) -> int:
    reverted = 0
    for rel in EDITS:
        target = resolve(root, rel)
        backup = target.with_suffix(target.suffix + ".bak")
        if not backup.exists():
            print(f"= {rel} (no backup, left alone)")
            continue
        shutil.copy2(backup, target)
        backup.unlink()
        reverted += 1
        print(f"reverted {rel} from {backup.name}")
    if not reverted:
        print("nothing to revert")
    return 0


def do_apply(root: Path, simple_env_root: Path) -> int:
    total_applied = 0

    for rel, edits in EDITS.items():
        target = resolve(root, rel)
        source = target.read_text()
        applied, skipped = [], []

        for edit in edits:
            if edit.applied(source):
                skipped.append(edit.name)
                continue
            try:
                source = edit.apply(source)
            except LookupError as exc:
                sys.exit(
                    f"{exc} in {target}.\n"
                    "The ResFiT file has diverged from what this script expects; "
                    "apply that edit by hand."
                )
            applied.append(edit.name)

        if not applied:
            print(f"= {rel} (already patched)")
            continue

        backup = target.with_suffix(target.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(target, backup)
            print(f"backup -> {backup}")
        target.write_text(source)
        print(f"patched {rel}")
        for name in applied:
            print(f"  + {name}")
        for name in skipped:
            print(f"  = {name} (already present)")
        total_applied += len(applied)

    if not total_applied:
        print("\nnothing to do")
        return 0

    hint = simple_env_root if simple_env_root.exists() else Path("/path/to/simple_robosuite_env")
    print(
        f"\n{ENV_NAME} is now a ResFiT task. Put the simple_robosuite_env checkout on "
        "PYTHONPATH so `import simple_env` resolves, and use EGL for the offscreen "
        "rollout renders:\n"
        f"  export PYTHONPATH={hint}:$PYTHONPATH\n"
        "  export MUJOCO_GL=egl\n"
        "Then fix up the conda env if you have not already:\n"
        "  python giovi/scripts/patch_residual.py"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--resfit-root",
        type=Path,
        default=DEFAULT_RESFIT_ROOT,
        help="ResFiT checkout to patch (default: %(default)s)",
    )
    ap.add_argument(
        "--simple-env-root",
        type=Path,
        default=DEFAULT_SIMPLE_ENV_ROOT,
        help="Checkout providing `import simple_env`, used for the PYTHONPATH hint "
        "(default: %(default)s)",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report which edits are present, change nothing")
    mode.add_argument("--revert", action="store_true", help="restore the .bak files")
    args = ap.parse_args()

    root = args.resfit_root.expanduser()
    if args.check:
        return do_check(root)
    if args.revert:
        return do_revert(root)
    return do_apply(root, args.simple_env_root.expanduser())


if __name__ == "__main__":
    sys.exit(main())
