"""Exact frame-count alignment for Director merge / cache / preview outputs."""

from __future__ import annotations

import torch


def minimax_align_frame_count(frame_count: int, continuity: bool = False) -> int:
    """Round up to the MiniMax H3 frame grid.

    Standalone segments land on 17k+5 (5, 22, 39, …). A segment that pins the
    previous tail (「引用上段」) exports on 17k instead: the head pin itself is
    17m+5 frames, so head + body still lands on the official 17k+5 grid.
    """
    n = max(5, int(frame_count))
    step = 17
    offset = 0 if continuity else 5
    if continuity:
        n = max(step, n)
    while (n - offset) % step != 0:
        n += 1
    return n


def pad_or_trim_frames(frames: torch.Tensor, target_len: int) -> torch.Tensor:
    """Trim to at most target_len frames. Does not fabricate last-frame duplicates."""
    target_len = max(0, int(target_len))
    if target_len <= 0:
        return frames[:0]
    if int(frames.shape[0]) > target_len:
        return frames[:target_len]
    return frames
