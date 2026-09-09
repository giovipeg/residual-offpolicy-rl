"""A `LeRobotDataset` that keeps the whole dataset decoded in RAM.

Why
===
`LeRobotDataset.__getitem__` re-decodes video frames on every access: each
sample opens one mp4 per camera, seeks, and decodes a single frame. For a small
imitation-learning dataset that is enormously wasteful -- CubeToContainer is
200 episodes / 17,462 frames at 84x84, which is 705 MiB once decoded and 3.6 s
to decode in full, yet a 200k-step run at batch 256 decodes 100 *million*
frames.

Profiled on `giovipeg/cube-to-container`, the per-item cost splits roughly as
below. The absolute numbers are from a dev laptop and will differ on the
training server -- what carries over is the ranking: decoding and the arrow
gather dominate, and the actual augmentation is the cheap part.

    arrow `select` of the 100-step action chunk   6.1 ms
    mp4 decode, 2 cameras                         8.2 ms
    color-jitter transforms                       2.9 ms

This class removes the first two by materialising both up front:

  * every video is decoded once into one contiguous uint8 tensor per camera,
  * every delta-indexed non-video column is stacked once into one tensor,

both in shared memory, so the dataloader workers map the same pages instead of
each holding a private copy. `__getitem__` then reduces to tensor indexing plus
the image transforms, which are left exactly as they were -- frames come back
as float32 in [0, 1], bit-identical to what the decoder returned.

On an image-mode dataset (`use_videos=False`, one PNG per frame) there are no videos
to cache, and PNG random reads are already ~25x cheaper than mp4 seeks (~0.17 ms vs
~4.1 ms per frame), so only the column cache applies. That is fine: the arrow gather
is the dominant cost for those datasets.

The cache is only worth building when the decoded dataset fits in RAM
comfortably; `estimated_cache_bytes` is there to check before committing. Note
that the cost is paid once per run, not once per worker: the tensors live in
shared memory, so N dataloader workers map the same pages.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


class RamCachedLeRobotDataset(LeRobotDataset):
    """`LeRobotDataset` with all frames and delta-indexed columns held in RAM.

    Takes the same arguments as `LeRobotDataset`; the cache is built during
    `__init__`, after the parent has set up metadata and the arrow table.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_cache()

    # ------------------------------------------------------------------
    # Cache construction
    # ------------------------------------------------------------------
    def _build_cache(self) -> None:
        from torchcodec.decoders import VideoDecoder

        t0 = time.perf_counter()

        starts = np.asarray(self.episode_data_index["from"])
        ends = np.asarray(self.episode_data_index["to"])
        lengths = ends - starts
        # Offset of each episode inside the concatenated frame cache.
        self._ep_offset = np.concatenate([[0], np.cumsum(lengths)])[:-1]

        # The cache is indexed by `offset[ep] + frame_in_ep`. That only lines up
        # with the arrow row order if episodes are contiguous and stored in
        # order, which is what `episode_data_index` should already guarantee --
        # assert it rather than silently serving the wrong frames.
        if not np.array_equal(self._ep_offset, starts):
            raise RuntimeError(
                "episodes are not laid out contiguously in episode order; the RAM "
                "cache cannot map arrow rows to frames. Use the uncached dataset."
            )

        self._frames: dict[str, torch.Tensor] = {}
        for key in self.meta.video_keys:
            per_ep = []
            for ep in range(self.meta.total_episodes):
                path = self.root / self.meta.get_video_file_path(ep, key)
                frames = VideoDecoder(str(path))[:]  # (T, 3, H, W) uint8
                if frames.shape[0] != lengths[ep]:
                    raise RuntimeError(
                        f"{path} has {frames.shape[0]} frames but episode {ep} has "
                        f"{lengths[ep]} rows; refusing to build a misaligned cache."
                    )
                per_ep.append(frames)
            tensor = torch.cat(per_ep, dim=0).contiguous()
            tensor.share_memory_()  # map, don't copy, into each dataloader worker
            self._frames[key] = tensor

        self._columns: dict[str, torch.Tensor] = {}
        for key in self.delta_indices or {}:
            # Skip every camera key, not just the video ones. On an image-mode
            # dataset (`use_videos=False`) the frames live in the arrow table as
            # PNGs that LeRobot decodes to float32 CHW on access, so stacking
            # that column would materialise ~4x the raw pixel size -- 2.8 GiB for
            # cube-to-container -- to cache what PNG already serves in ~0.17 ms
            # per frame. Those datasets get the column cache only, which is where
            # the win is for them anyway.
            if key in self.meta.camera_keys:
                continue
            column = torch.stack(self.hf_dataset[key]).contiguous()
            column.share_memory_()
            self._columns[key] = column

        nbytes = sum(t.numel() * t.element_size() for t in self._frames.values())
        nbytes += sum(t.numel() * t.element_size() for t in self._columns.values())
        logger.info(
            "RAM cache: %d frames across %d camera(s) + %d column(s), %.0f MiB, built in %.1f s",
            sum(t.shape[0] for t in self._frames.values()),
            len(self._frames),
            len(self._columns),
            nbytes / 2**20,
            time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # Overrides -- same contracts as the parent, served from the cache
    # ------------------------------------------------------------------
    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        item = {
            key: self._columns[key][torch.as_tensor(q_idx)]
            for key, q_idx in query_indices.items()
            if key in self._columns
        }
        # Anything not cached (image-mode camera keys) falls through to LeRobot,
        # which drops video keys itself.
        uncached = {k: v for k, v in query_indices.items() if k not in self._columns}
        if uncached:
            item.update(super()._query_hf_dataset(uncached))
        return item

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        offset = int(self._ep_offset[ep_idx])
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            # The decoder maps a timestamp to `round(ts * fps)` within the
            # episode's own video, so the cache does the same.
            rows = [offset + round(ts * self.meta.fps) for ts in query_ts]
            frames = self._frames[vid_key][rows].float().div_(255)
            item[vid_key] = frames.squeeze(0)
        return item


def estimated_cache_bytes(repo_id: str, root=None) -> int:
    """Decoded size of `repo_id`, to sanity-check RAM before building the cache."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    total = 0
    for key in meta.video_keys:
        shape = meta.features[key]["shape"]
        total += meta.total_frames * int(np.prod(shape))
    return total
