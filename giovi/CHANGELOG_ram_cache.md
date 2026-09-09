# BC dataloader RAM cache

**Date:** 2026-09-09
**Scope:** `train_act.py` (ACT/BC training on `giovipeg/cube-to-container`)

## Problem

BC training was dataloader-bound, not GPU-bound. Stock `LeRobotDataset.__getitem__`
re-decodes video frames on every access: each sample opens one mp4 per camera, seeks,
and decodes a single frame. A 200k-step run at batch 256 therefore decodes ~100 million
frames out of a dataset that is only **705 MiB once decoded** (200 episodes / 17,462
frames at 84x84, 2 cameras) and takes **3.6 s** to decode in full.

Profiled per-item cost (dev laptop; absolute numbers will differ on the training server,
the ranking is what carries over):

| component | cost | note |
|---|---|---|
| arrow `select` of the 100-step action chunk | 6.1 ms | 100 scattered rows -> Python list -> `torch.stack` |
| mp4 decode, 2 cameras | 8.2 ms | one file open + seek *per sample* |
| color-jitter transforms | 2.9 ms | the actual augmentation is the cheap part |

Note the `data:` figures in the training log are a single instantaneous sample per
`log_freq` steps, so they swing wildly (0.5 ms -> 1600 ms). They report whether the
prefetch queue happened to be full at that step, not the average cost.

## Changes

### Added: `giovi/ram_cache.py`

`RamCachedLeRobotDataset(LeRobotDataset)`. On construction it materialises, once:

* every video, decoded into one contiguous **uint8** tensor per camera;
* every delta-indexed non-video column, stacked into one tensor.

Both go in **shared memory** (`share_memory_()`), so N dataloader workers map the same
705 MiB rather than each holding a private copy. It overrides only `_query_videos` and
`_query_hf_dataset`; image transforms are untouched and frames still come back as
float32 in [0, 1].

It refuses to build (rather than silently serving misaligned frames) if episodes are not
laid out contiguously in episode order, or if a video's frame count does not match its
episode length.

Also exports `estimated_cache_bytes(repo_id)` to check the decoded size against available
RAM before committing.

### Modified: `resfit/lerobot/scripts/train_bc_dexmg.py`

Adds a `--cache_in_ram` flag and swaps the dataset class behind it. Applied through the
repo's existing patch convention, **not** edited by hand -- see below.

### Modified: `giovi/scripts/patch_resfit.py`

Two new idempotent `Edit`s:

* `--cache_in_ram flag` -- the argparse flag.
* `RAM-cached dataset class` -- selects `RamCachedLeRobotDataset` when the flag is set.

Apply / check / revert as usual:

```
python giovi/scripts/patch_resfit.py
python giovi/scripts/patch_resfit.py --check
python giovi/scripts/patch_resfit.py --revert
```

### Modified: `giovi/scripts/train_act.py`

* `--cache_in_ram` added to `ARGS` (on by default for this run).
* `--num_workers` is **derived from `os.cpu_count()`**, not hard-coded, since it is the
  one setting that does not transfer between machines.
* Flags you override are now dropped from `ARGS` (`drop_overridden`), so the printed
  command shows `--num_workers 8` once instead of `--num_workers 12 --num_workers 8`.
  This is cosmetic: argparse took the last occurrence before, so the override already
  won -- it just was not obvious which value applied.

### Added: `giovi/scripts/tune_num_workers.py`

Sweeps `--num_workers` on the machine training actually runs on, since that setting does
not transfer between machines. See "Tuning on the training server" below.

### Modified: `giovi/scripts/_launch.py`

Repo root added to `PYTHONPATH` so `import giovi.ram_cache` resolves in the trainer
subprocess regardless of cwd.

## Storage format (`simple_robosuite_env`)

The mp4 container is a poor fit for this access pattern: it buys ~8x smaller storage at
the cost of a container open + seek per sample. Measured alternatives, per frame:

| storage | on disk (200 eps) | image read cost | action-chunk cost |
|---|---|---|---|
| mp4 (current) | 25 MiB | 4.1 ms/frame | 6.1 ms |
| image / PNG | 214 MiB | 0.17 ms/frame | 6.1 ms |
| RAM cache over mp4 | 25 MiB | ~0 | ~0 |

`scripts/collect_dataset.py` in the `simple_robosuite_env` checkout gained two flags
(both default to the previous behaviour, so nothing changes unless asked):

* `--image-frames` -- write `dtype: "image"` and `use_videos=False`. Note LeRobot embeds
  these as PNG bytes *inside the parquet* (5.5 KiB/frame), not as loose files, so this
  does not create 35k small files.
* `--keep-shards` -- keep the raw `.npz` rollout shards instead of deleting them, so the
  storage format can be changed later by re-running only the write stage rather than the
  whole simulation.

**Not done:** the published `giovipeg/cube-to-container` was left as mp4. Re-collecting
is cheap (~2-3 min for 200 episodes) but would produce *different pixels* -- pristine
renders rather than h264-decoded ones -- which would not be comparable against BC runs
trained on the current dataset.

## Verification

Verified, and machine-independent:

* **Output is bit-identical to the uncached dataset.** 324 items x 15 keys, including
  both endpoints of every 17th episode (where action-chunk clamping and the `*_is_pad`
  flags apply): same dtype, same shape, `torch.equal` exact.
* **Index-mapping assumptions all hold:** `timestamp == frame_index / fps`; episodes
  contiguous and in order; `frame_index` restarts at 0 per episode; video frame count ==
  episode length for all 400 videos; cached frame == decoded frame at 0.0 max abs diff.
* **Builds in the real trainer:** `RAM cache: 34924 frames across 2 camera(s) +
  1 column(s), 705 MiB, built in 3.8 s`, reaching step 0 at a sane loss.
* **Patch is idempotent** (re-run reports "nothing to do") and writes a `.bak`.
* **Safe on image-mode datasets.** The cache skips *all* camera keys when building the
  column cache, not just video keys. Without that, an image-mode dataset would stack the
  PNG column into float32 CHW -- ~2.8 GiB for cube-to-container -- to cache something PNG
  already serves in 0.17 ms. Uncached keys fall through to LeRobot. Verified against a
  2-episode image-mode dataset: identical over all 164 items, 0 frame tensors cached.

**Not verified: the end-to-end speedup.** It was only ever measured on a dev laptop,
which is not where training runs, so no throughput number here should be quoted as the
expected result -- measure on the server with `tune_num_workers.py`. (An early ad-hoc
benchmark reporting 988 -> 207 ms/batch was itself unreliable: it swept configurations in
one process without tearing down the previous sweep's workers, so they competed. The
tuner isolates each configuration.) The structural win does not depend on the machine;
on a faster GPU it should be *larger*, since the update time shrinks while decode stays
CPU-bound.

## Tuning on the training server

1. Confirm the decoded dataset fits comfortably in RAM (`estimated_cache_bytes`); it is
   705 MiB for cube-to-container.
2. Run the sweep and put the winner in `train_act.py`:

   ```
   python giovi/scripts/tune_num_workers.py                     # cached, with GPU floor
   python giovi/scripts/tune_num_workers.py --compare-uncached  # size what the cache buys
   python giovi/scripts/tune_num_workers.py --skip-update       # dataloader only, no GPU
   ```

   It builds the dataset, policy and dataloader the way `train_bc_dexmg.py` does and
   reports, per worker count, dataloader ms/batch, real training-loop ms/step, and
   whether that configuration is data-bound or GPU-bound. More workers is *not*
   monotonically better -- oversubscribing the decode/collate threads makes it slower --
   so expect a peak below core count.
3. Once the tuner says GPU-bound, more workers cannot help; further gains have to come
   from the model side (bf16 autocast, `torch.compile`) or from the eval cadence.
4. If it is still data-bound at every worker count, the next lever is the float32
   worker->main IPC: `_query_videos` returns float32, so a batch of 256 moves ~43 MB
   through shared memory instead of ~10.8 MB as uint8. Passing uint8 through the loader
   and normalising on GPU would cut that 4x, at the cost of augmenting in 8-bit. Beyond
   that, a GPU-resident pipeline removes the dataloader entirely.

## Known pre-existing bug (not fixed)

The trainer crashes at the first log step unless `--wandb_enable` is passed: the guard is
`if wandb is not None` rather than `if cfg.wandb_enable`, so an importable module is
enough to reach `wandb.log()` with no `wandb.init()`. Workaround for local runs:
`WANDB_MODE=disabled` plus `--wandb_enable --wandb_project <name>`.
