"""Standalone reference-video helpers (MiniMax H3: up to 3 → <Video 1>…<Video 3>)."""

from __future__ import annotations

from typing import Any

import torch

# Official MiniMaxH3ReferenceToVideo Autogrow max=3.
MAX_REFERENCE_VIDEOS = 3
REF_VIDEO_KEY_PREFIX = "ref_video_"

#: Sanity bound on a *stored* id, not the model slot count: the timeline stores
#: materials by absolute id, and the run renumbers the referenced ones to
#: ``ref_video_0..N-1`` before the model call (batch_prepare).
_MAX_SLOT_ID = 64


def ref_videos_dict(items: list[tuple[int, torch.Tensor]]) -> dict[str, torch.Tensor] | None:
    """Build official ``ref_video_N`` mapping from (index, frames) pairs.

    The keys keep the caller's ids; :func:`_renumber` in the prepare step turns
    the referenced ones into gap-free slots, so an id above
    :data:`MAX_REFERENCE_VIDEOS` must not be dropped here.
    """
    out: dict[str, torch.Tensor] = {}
    for index, frames in items:
        idx = int(index)
        if idx < 0 or idx >= _MAX_SLOT_ID:
            continue
        if not isinstance(frames, torch.Tensor) or frames.numel() <= 0:
            continue
        out[f"{REF_VIDEO_KEY_PREFIX}{idx}"] = frames
    return out or None
