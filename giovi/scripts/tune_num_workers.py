#!/usr/bin/env python3
"""Find the best `--num_workers` for BC training on this machine.

The worker count is the one training setting that does not transfer between
machines: too few and the GPU starves, too many and the decode/collate threads
oversubscribe the cores and throughput *drops*. It has to be measured where
training actually runs, which is what this script is for -- run it once on the
training server and put the winner in `train_act.py` (or pass `--num_workers`).

It builds the dataset, policy and dataloader exactly the way
`train_bc_dexmg.py` does, then for each candidate worker count reports:

  data    -- ms/batch the dataloader alone can sustain
  update  -- ms/batch for forward + backward + optimizer step (worker-independent)
  loop    -- ms/step for the real training loop, which is the number that matters

`loop` is roughly max(data, update) when prefetching keeps up, so once `loop`
approaches `update` you are GPU-bound and more workers cannot help.

Usage:
    python giovi/scripts/tune_num_workers.py
    python giovi/scripts/tune_num_workers.py --workers 4 8 12 16 --batches 50
    python giovi/scripts/tune_num_workers.py --compare-uncached
    python giovi/scripts/tune_num_workers.py --skip-update   # no GPU work at all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# Repo root, so `giovi.ram_cache` and `resfit` both import when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build(cfg):
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.common.datasets.transforms import ImageTransforms, ImageTransformsConfig
    from resfit.lerobot.policies.factory import make_policy, make_policy_config

    ds_meta = LeRobotDatasetMetadata(cfg.dataset)
    policy_cfg = make_policy_config(cfg.policy)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    image_transforms = ImageTransforms(ImageTransformsConfig(enable=True))

    def make_dataset(cached: bool):
        cls = LeRobotDataset
        if cached:
            from giovi.ram_cache import RamCachedLeRobotDataset

            cls = RamCachedLeRobotDataset
        return cls(
            cfg.dataset,
            delta_timestamps=delta_timestamps,
            download_videos=True,
            image_transforms=image_transforms,
        )

    policy = None
    if not cfg.skip_update:
        policy = make_policy(policy_cfg, ds_meta=ds_meta)
        policy.train()
    return make_dataset, policy


def loader(dataset, cfg, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg.device != "cpu",
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def time_data(dl, device, batches, warmup):
    """ms/batch the dataloader sustains, worker spin-up excluded."""
    it = iter(dl)
    for _ in range(warmup):
        to_device(next(it), device)
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(batches):
        to_device(next(it), device)
    if device != "cpu":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / batches * 1000


def time_update(policy, optimizer, batch, cfg, batches, warmup):
    """ms/batch for forward + backward + step on a fixed batch."""

    def one():
        loss, _ = policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(batches):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / batches * 1000


def time_loop(dl, policy, optimizer, cfg, batches, warmup):
    """ms/step for the real training loop -- the number that actually matters."""
    it = iter(dl)

    def one():
        batch = to_device(next(it), cfg.device)
        loss, _ = policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(batches):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / batches * 1000


def sweep(dataset, policy, cfg, label):
    print(f"\n=== {label} ===")
    optimizer = None
    update_ms = None
    if policy is not None:
        policy.to(cfg.device)
        optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5)
        probe = loader(dataset, cfg, min(cfg.workers))
        batch = to_device(next(iter(probe)), cfg.device)
        update_ms = time_update(policy, optimizer, batch, cfg, cfg.batches, cfg.warmup)
        del probe, batch
        print(f"update (GPU floor, worker-independent): {update_ms:7.1f} ms/step")

    header = f"{'workers':>8} {'data ms':>10} {'loop ms':>10} {'steps/s':>9} {'verdict':>14}"
    if update_ms is None:
        header = f"{'workers':>8} {'data ms':>10}"
    print(header)

    rows = []
    for nw in cfg.workers:
        dl = loader(dataset, cfg, nw)
        data_ms = time_data(dl, cfg.device, cfg.batches, cfg.warmup)
        if update_ms is None:
            print(f"{nw:>8} {data_ms:>10.1f}")
            rows.append((nw, data_ms, data_ms))
        else:
            loop_ms = time_loop(dl, policy, optimizer, cfg, cfg.batches, cfg.warmup)
            verdict = "GPU-bound" if data_ms <= update_ms * 1.1 else "data-bound"
            print(f"{nw:>8} {data_ms:>10.1f} {loop_ms:>10.1f} {1000 / loop_ms:>9.1f} {verdict:>14}")
            rows.append((nw, data_ms, loop_ms))
        # Tear the workers down before the next config so they do not overlap.
        del dl

    best = min(rows, key=lambda r: r[2])
    print(f"\nbest: --num_workers {best[0]}  ({best[2]:.1f} ms/step, "
          f"{best[2] * cfg.project / 3.6e6:.1f} h for {cfg.project:,} steps)")
    return best


def main() -> int:
    cpus = os.cpu_count() or 4
    default_workers = sorted({w for w in (2, 4, 6, 8, 12, 16, 24, 32) if w <= cpus} | {min(cpus, 32)})

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="giovipeg/cube-to-container")
    ap.add_argument("--policy", default="act")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, nargs="+", default=default_workers)
    ap.add_argument("--batches", type=int, default=30, help="timed batches per configuration")
    ap.add_argument("--warmup", type=int, default=5, help="untimed batches first (worker spin-up)")
    ap.add_argument("--grad_clip_norm", type=float, default=10.0)
    ap.add_argument("--project", type=int, default=200_000, help="step count for the wall-clock projection")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-cache", action="store_true", help="sweep the plain video-decoding dataset")
    ap.add_argument("--compare-uncached", action="store_true", help="sweep both, to size what the cache buys")
    ap.add_argument("--skip-update", action="store_true", help="dataloader only; no policy, no GPU work")
    cfg = ap.parse_args()

    print(f"host: {cpus} CPUs | device: {cfg.device} | batch {cfg.batch_size} | sweeping {cfg.workers}")
    make_dataset, policy = build(cfg)

    if cfg.compare_uncached or cfg.no_cache:
        sweep(make_dataset(cached=False), policy, cfg, "video-decoding dataset (no cache)")
    if not cfg.no_cache:
        sweep(make_dataset(cached=True), policy, cfg, "RAM-cached dataset (--cache_in_ram)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
