"""Exact frame-count alignment for Director merge / cache / preview outputs."""

from __future__ import annotations

import torch


def minimax_align_frame_count(frame_count: int) -> int:
    """Round up to MiniMax H3 17k+5 frame grid (5, 22, 39, …)."""
    n = max(5, int(frame_count))
    while n % 17 != 5:
        n += 1
    return n


def wan_align_frame_count(frame_count: int) -> int:
    """Legacy alias — MiniMax H3 uses 17k+5, not Wan 4n+1."""
    return minimax_align_frame_count(frame_count)


def pad_or_trim_frames(frames: torch.Tensor, target_len: int) -> torch.Tensor:
    """Trim to at most target_len frames. Does not fabricate last-frame duplicates."""
    target_len = max(0, int(target_len))
    if target_len <= 0:
        return frames[:0]
    if int(frames.shape[0]) > target_len:
        return frames[:target_len]
    return frames


def minimax_phase_aligned_export_frames(frame_count: int) -> int:
    """Snap an aligned export length down onto the 17-frame VAE cycle grid.

    Motion context pins the previous segment's tail at a latent step that is a
    multiple of 5, i.e. a pixel offset that is a multiple of 17
    (``pixel_frames_for_latent_t``: 1 + 4 + 4 + 4 + 4 = 17 per 5 steps).

    ``minimax_align_frame_count`` always yields 17k+5, so a pin window ending at
    or before the export end can only reach 17k — a fixed 5-frame remainder.
    Those frames are then either dropped from the previous export (lost
    duration) or replayed as a ~5f echo at the seam.

    Exporting 17k instead makes the pin window end exactly on the last exported
    frame, so ``gap_after_pin`` is 0 and the trim-vs-echo trade-off disappears.
    """
    n = max(1, int(frame_count))
    return max(17, n - (n % 17))
