#!/usr/bin/env python3
"""Make sure a residual-TD3 task config exists in ResFiT, and point it at a BC run.

`train_residual_td3` has no CLI flag for the base policy: the README requires the
`<wandb_project>/<run_id>` of the BC run to live in the task's config class in
`resfit/rl_finetuning/config/residual_td3.py`. That file also has no entry for
tasks ResFiT does not ship, so a new task needs a whole dataclass written by hand.

This script does both, from the task table in `giovi/configs/residual_td3_tasks.json`:

  * if the task's config class is missing, it renders one from the table and
    inserts it, along with its `cs.store(...)` registration;
  * if `--base-policy` is given, it writes that id into the class's
    `BasePolicyConfig`.

Both steps are idempotent, and the first change writes a `.bak` next to the file
(same convention as `patch_resfit.py`).

Class and Hydra names are derived from the task name, so `CubeToContainer` gives
`ResidualTD3CubeToContainerConfig` / `residual_td3_cube_to_container_config`.

Usage:
    python giovi/scripts/ensure_task_config.py CubeToContainer --base-policy dexmg-bc/abc123
    python giovi/scripts/ensure_task_config.py CubeToContainer --check
    python giovi/scripts/ensure_task_config.py --list
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# giovi/scripts/ensure_task_config.py -> the ResFiT repo root.
DEFAULT_RESFIT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_FILE = Path(__file__).resolve().parents[1] / "configs" / "residual_td3_tasks.json"
CONFIG_REL = "resfit/rl_finetuning/config/residual_td3.py"

# The generated classes are inserted immediately above this banner, and their
# cs.store(...) lines immediately after the last existing one.
HYDRA_BANNER = "# Register with Hydra"

BASE_CLASS = "ResidualTD3DexmgConfig"


def class_name(task: str) -> str:
    return f"ResidualTD3{task}Config"


def config_name(task: str) -> str:
    """CubeToContainer -> residual_td3_cube_to_container_config."""
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", task).lower()
    return f"residual_td3_{snake}_config"


def load_tasks(tasks_file: Path) -> dict[str, dict]:
    if not tasks_file.exists():
        sys.exit(f"task table not found: {tasks_file}")
    with tasks_file.open() as fh:
        table = json.load(fh)
    return {k: v for k, v in table.items() if not k.startswith("_")}


def render(task: str, spec: dict, tasks_file: Path) -> str:
    """Render the dataclass for `task`, in the style of the classes around it."""

    if "dataset" not in spec:
        sys.exit(f"task '{task}' has no 'dataset' in the task table")

    description = spec.get("description", f"{task} residual TD3 task.")
    try:
        provenance = tasks_file.relative_to(DEFAULT_RESFIT_ROOT)
    except ValueError:
        provenance = tasks_file

    lines = [
        "@dataclass",
        f"class {class_name(task)}({BASE_CLASS}):",
        f'    """{description}',
        "",
        f"    Generated from {provenance} by giovi/scripts/ensure_task_config.py --",
        "    edit the task table and rerun rather than editing this class.",
        '    """',
        "",
        f'    task: str = "{task}"',
    ]

    if "video_key" in spec:
        lines += ["", f'    video_key: str = "{spec["video_key"]}"']

    if "rl_camera" in spec:
        cameras = "\n".join(f'            "{cam}",' for cam in spec["rl_camera"])
        lines += [
            "",
            "    rl_camera: list[str] = field(",
            "        default_factory=lambda: [",
            cameras,
            "        ]",
            "    )",
        ]

    lines += [
        "",
        "    offline_data: OfflineDataConfig = field(",
        "        default_factory=lambda: OfflineDataConfig(",
        f'            name="{spec["dataset"]}",',
    ]
    if "num_episodes" in spec:
        lines.append(f"            num_episodes={spec['num_episodes']},")
    lines += [
        "        )",
        "    )",
        "",
        "    base_policy: BasePolicyConfig = field(",
        "        default_factory=lambda: BasePolicyConfig(",
        '            wandb_id="TODO",',
        "        )",
        "    )",
    ]

    if "wandb_project" in spec:
        lines += [
            "",
            "    wandb: WandBConfig = field(",
            f'        default_factory=lambda: WandBConfig(project="{spec["wandb_project"]}")',
            "    )",
        ]

    return "\n".join(lines)


def find_class_block(source: str, cls: str) -> tuple[int, int] | None:
    """Span of `cls`'s definition, from its @dataclass line to the next top-level form."""
    match = re.search(rf"^@dataclass\nclass {re.escape(cls)}\(", source, re.MULTILINE)
    if match is None:
        return None
    start = match.start()
    rest = source[match.end() :]
    nxt = re.search(r"^(@dataclass|# -{10,}|cs\.store\()", rest, re.MULTILINE)
    end = match.end() + nxt.start() if nxt else len(source)
    return start, end


def insert_class(source: str, task: str, spec: dict, tasks_file: Path) -> str:
    banner = source.index(HYDRA_BANNER)
    # Back up to the top of the separator comment block that precedes the banner.
    block_start = source.rindex("# ---", 0, banner)

    prefix = source[:block_start].rstrip("\n")
    suffix = source[block_start:]
    body = render(task, spec, tasks_file)

    source = f"{prefix}\n\n\n{body}\n\n\n{suffix}"

    stores = list(re.finditer(r"^cs\.store\(.*\)$", source, re.MULTILINE))
    if not stores:
        sys.exit(f"no cs.store(...) calls found in {CONFIG_REL}; cannot register the task")
    last = stores[-1].end()
    registration = f'\ncs.store(name="{config_name(task)}", node={class_name(task)})'
    return source[:last] + registration + source[last:]


def set_base_policy(source: str, cls: str, base_policy: str) -> str:
    span = find_class_block(source, cls)
    if span is None:
        sys.exit(f"class {cls} not found; cannot set its base policy")
    start, end = span
    block = source[start:end]

    found = len(re.findall(r'wandb_id="[^"]*"', block))
    if found != 1:
        sys.exit(f"expected exactly one wandb_id in {cls}, found {found}")

    patched = re.sub(r'wandb_id="[^"]*"', f'wandb_id="{base_policy}"', block, count=1)
    return source[:start] + patched + source[end:]


def current_base_policy(source: str, cls: str) -> str | None:
    span = find_class_block(source, cls)
    if span is None:
        return None
    match = re.search(r'wandb_id="([^"]*)"', source[span[0] : span[1]])
    return match.group(1) if match else None


def write(target: Path, source: str) -> None:
    backup = target.with_suffix(target.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"backup -> {backup}")
    target.write_text(source)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", help="task name, as keyed in the task table (e.g. CubeToContainer)")
    ap.add_argument(
        "--base-policy",
        metavar="PROJECT/RUN_ID",
        help="W&B `<project>/<run_id>` of the BC run to use as the base policy",
    )
    ap.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE, help="task table (default: %(default)s)")
    ap.add_argument("--resfit-root", type=Path, default=DEFAULT_RESFIT_ROOT, help="ResFiT checkout to patch")
    ap.add_argument("--list", action="store_true", help="list the tasks in the table and exit")
    ap.add_argument("--check", action="store_true", help="report the task's status, change nothing")
    ap.add_argument("--print", action="store_true", help="print the rendered class and exit, changing nothing")
    args = ap.parse_args()

    tasks = load_tasks(args.tasks_file.expanduser())

    if args.list:
        for name, spec in tasks.items():
            print(f"{name}  ({spec.get('dataset', 'no dataset')})")
        return 0

    if args.task is None:
        ap.error("a task name is required unless --list is given")

    if args.task not in tasks:
        sys.exit(f"task '{args.task}' is not in {args.tasks_file}. Known: {', '.join(tasks) or 'none'}")
    spec = tasks[args.task]
    cls = class_name(args.task)

    if args.print:
        print(render(args.task, spec, args.tasks_file.expanduser()))
        return 0

    target = args.resfit_root.expanduser() / CONFIG_REL
    if not target.exists():
        sys.exit(f"not found: {target}\nIs --resfit-root really a ResFiT checkout?")
    source = original = target.read_text()

    exists = find_class_block(source, cls) is not None

    if args.check:
        print(f"{cls}: {'present' if exists else 'MISSING'}")
        if exists:
            print(f"  base policy: {current_base_policy(source, cls)}")
        print(f"  --config-name={config_name(args.task)}")
        return 0 if exists else 1

    if not exists:
        source = insert_class(source, args.task, spec, args.tasks_file.expanduser())
        print(f"+ {cls} generated from {args.tasks_file}")
    else:
        print(f"= {cls} already present")

    if args.base_policy:
        source = set_base_policy(source, cls, args.base_policy)
        print(f"  base policy: {args.base_policy}")

    if source == original:
        print(f"= {CONFIG_REL} unchanged")
        return 0

    write(target, source)
    print(f"patched {CONFIG_REL}  (--config-name={config_name(args.task)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
