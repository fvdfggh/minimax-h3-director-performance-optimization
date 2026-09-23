"""Standalone reference-video helpers (MiniMax H3: up to 3 → <Video 1>…<Video 3>)."""

from __future__ import annotations

from typing import Any

import torch

# Official MiniMaxH3ReferenceToVideo Autogrow max=3.
MAX_REFERENCE_VIDEOS = 3
REF_VIDEO_KEY_PREFIX = "ref_video_"


def ref_videos_dict(items: list[tuple[int, torch.Tensor]]) -> dict[str, torch.Tensor] | None:
    """Build official ``ref_video_N`` mapping from (index, frames) pairs."""
    out: dict[str, torch.Tensor] = {}
    for index, frames in items:
        idx = int(index)
        if idx < 0 or idx >= MAX_REFERENCE_VIDEOS:
            continue
        if not isinstance(frames, torch.Tensor) or frames.numel() <= 0:
            continue
        out[f"{REF_VIDEO_KEY_PREFIX}{idx}"] = frames
    return out or None
