#!/usr/bin/env python3
"""Fix up the `residual` conda env so the ResFiT training scripts actually run.

`patch_resfit.py` patches ResFiT's *source*; this script patches the *env*
around it. Every fixup here is something that breaks a real run:

  * **torchcodec** -- the env ships torchcodec 0.4.0, which is built against an
    older libtorch than the installed torch, so every decoder library fails to
    load (`undefined symbol: _ZN3c104impl3cow23materialize_cow_storage...`) and
    `LeRobotDataset.__getitem__` dies on the first video frame. LeRobot's
    documented fallback (`--video_backend pyav`) does *not* rescue this: that
    path goes through `torchvision.io.VideoReader`, which torchvision removed
    in 0.22, so it raises `AttributeError` instead. Upgrading torchcodec is the
    only working route. `--no-deps` keeps pip from touching torch/torchvision.

Each fixup is verified by actually exercising it, not by comparing version
strings, so a fixup that is already unnecessary is skipped.

Usage:
    python giovi/scripts/patch_residual.py            # apply what is needed
    python giovi/scripts/patch_residual.py --check    # report only, change nothing
    python giovi/scripts/patch_residual.py --dry-run  # print the commands
    python giovi/scripts/patch_residual.py --revert   # restore the original versions
"""

from __future__ import annotations

import argparse
import ctypes.util
import importlib.metadata
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EXPECTED_ENV = "residual"

# The torchcodec release that loads against torch >= 2.14 and the FFmpeg 4
# shipped by Ubuntu 22.04, and the one the env originally had.
TORCHCODEC_WANTED = "0.16.0"
TORCHCODEC_ORIGINAL = "0.4.0"

# Probe run in a subprocess: torchcodec dlopen()s its decoder libs at import
# time, so a plain import is the whole test.
TORCHCODEC_PROBE = "import torch; from torchcodec.decoders import VideoDecoder"


def run(cmd: list[str], dry_run: bool = False) -> int:
    print("  $ " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def probe(code: str) -> tuple[bool, str]:
    """Run `code` in a fresh interpreter; return (ok, last line of stderr)."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True, ""
    lines = [line.strip() for line in proc.stderr.splitlines() if line.strip()]
    for line in reversed(lines):
        if re.match(r"^[\w.]*(Error|Exception)\b", line):
            return False, line
    return False, lines[-1] if lines else "failed"


def pip(*args: str) -> list[str]:
    return [sys.executable, "-m", "pip", *args]


@dataclass
class Fixup:
    name: str
    why: str
    probe_code: str
    apply_cmd: list[str]
    revert_cmd: list[str]

    def ok(self) -> tuple[bool, str]:
        return probe(self.probe_code)


FIXUPS = [
    Fixup(
        name="torchcodec (LeRobot video decoding)",
        why=f"torchcodec {TORCHCODEC_ORIGINAL} is ABI-incompatible with the installed torch, "
        "and LeRobot's pyav fallback is dead on torchvision >= 0.22",
        probe_code=TORCHCODEC_PROBE,
        apply_cmd=pip("install", "--no-deps", "-U", f"torchcodec=={TORCHCODEC_WANTED}"),
        revert_cmd=pip("install", "--no-deps", f"torchcodec=={TORCHCODEC_ORIGINAL}"),
    ),
]


def report_env() -> None:
    print(f"interpreter: {sys.executable}")
    versions = []
    for dist in ("torch", "torchvision", "torchcodec", "av", "robosuite", "lerobot"):
        try:
            versions.append(f"{dist}={importlib.metadata.version(dist)}")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"{dist}=absent")
    print("packages:   " + "  ".join(versions))

    ffmpeg = shutil.which("ffmpeg") or "not on PATH"
    libav = ctypes.util.find_library("avutil") or "not found"
    print(f"ffmpeg:     {ffmpeg} (libavutil: {libav})")
    print()


def do_check() -> int:
    report_env()
    missing = 0
    for fix in FIXUPS:
        ok, err = fix.ok()
        missing += not ok
        print(f"  {'+' if ok else '-'} {fix.name}")
        if not ok:
            print(f"      {err}")
            print(f"      why: {fix.why}")
    if missing:
        print(f"\n{missing} fixup(s) needed -- run without --check to apply.")
        return 1
    print("\nenv is good.")
    return 0


def do_apply(dry_run: bool) -> int:
    report_env()
    applied = 0
    for fix in FIXUPS:
        ok, err = fix.ok()
        if ok:
            print(f"= {fix.name} (already working)")
            continue
        print(f"+ {fix.name}")
        print(f"      {err}")
        print(f"      why: {fix.why}")
        if run(fix.apply_cmd, dry_run) != 0:
            sys.exit(f"command failed while fixing '{fix.name}'")
        applied += 1
        if dry_run:
            continue
        ok, err = fix.ok()
        if not ok:
            sys.exit(f"'{fix.name}' still broken after the fix:\n  {err}")
        print("      verified")

    if dry_run:
        print("\ndry run -- nothing was installed")
        return 0
    if not applied:
        print("\nnothing to do")
        return 0

    simple_env_root = Path(__file__).resolve().parents[3] / "simple_robosuite_env"
    hint = simple_env_root if simple_env_root.exists() else Path("/path/to/simple_robosuite_env")
    print(
        f"\n{applied} fixup(s) applied. Reminder for the ResFiT side:\n"
        f"  export PYTHONPATH={hint}:$PYTHONPATH   # so `import simple_env` resolves\n"
        "  export MUJOCO_GL=egl   # offscreen rollout rendering\n"
        "  python giovi/scripts/patch_resfit.py"
    )
    return 0


def do_revert(dry_run: bool) -> int:
    for fix in FIXUPS:
        print(f"- {fix.name}")
        run(fix.revert_cmd, dry_run)
    if not dry_run:
        print("\nreverted to the env's original versions (runs will break again)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report what is broken, change nothing")
    mode.add_argument("--revert", action="store_true", help="reinstall the original package versions")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--force", action="store_true", help=f"proceed even if the env is not '{EXPECTED_ENV}'")
    args = ap.parse_args()

    # Keep our prints interleaved correctly with pip's output when piped.
    sys.stdout.reconfigure(line_buffering=True)

    env_name = Path(sys.prefix).name
    if env_name != EXPECTED_ENV and not (args.check or args.force):
        sys.exit(
            f"this interpreter lives in '{env_name}', not '{EXPECTED_ENV}' ({sys.executable}).\n"
            f"Activate the right env (conda activate {EXPECTED_ENV}) or pass --force."
        )

    if args.check:
        return do_check()
    if args.revert:
        return do_revert(args.dry_run)
    return do_apply(args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
