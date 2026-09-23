"""Timeline frame maps: which source frame a logical frame reads from.

A Director timeline can reference *several* clips, reorder them, and delete ranges
from the middle. A logical frame index (what the user sees on the timeline) is
therefore not a source frame index — this layer owns that translation, plus the
readers built on it (:func:`load_timeline_segment` for one segment,
:func:`load_multi_clip_timeline` for a whole multi-clip timeline).

Depends on :mod:`av_probe` for path resolution and :mod:`video_decode` for the
actual pixel reads.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from .av_probe import resolve_video_path
from .video_decode import load_video_resampled

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.video.io.timeline")


def parse_frame_map_entry(entry: Any, default_clip: int = 0) -> tuple[int, int]:
    """Parse a frameMap entry to (clip_index, source_frame_index)."""
    if isinstance(entry, dict):
        clip = int(entry.get("clip", entry.get("videoClip", default_clip)))
        frame = int(entry.get("frame", 0))
        return clip, frame
    return default_clip, int(entry)


def video_clips_from_timeline(timeline: dict) -> list[dict]:
    """Return ordered video clip metadata; falls back to legacy single ``video`` block."""
    clips = timeline.get("videoClips") or timeline.get("video_clips")
    if clips:
        return list(clips)
    video = timeline.get("video") or {}
    if (video.get("videoFile") or video.get("fileName") or "").strip():
        return [video]
    return []


def deleted_source_ranges(timeline: dict) -> list[tuple[int, int]]:
    """Source-frame spans removed from the logical timeline (sparse single-clip edits)."""
    video = timeline.get("video") or {}
    raw = video.get("deletedSourceRanges") or video.get("deleted_source_ranges") or []
    ranges: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = int(item[0]), int(item[1])
            if end > start:
                ranges.append((start, end))
    return sorted(ranges)


def logical_frame_map(timeline: dict) -> list[Any]:
    """Explicit per-logical-frame map only; empty means sparse identity mapping."""
    video = timeline.get("video") or {}
    frame_map = video.get("frameMap")
    if frame_map:
        return list(frame_map)
    return []


def logical_frame_count(timeline: dict) -> int:
    frame_map = logical_frame_map(timeline)
    if frame_map:
        return len(frame_map)

    total = int(timeline.get("totalFrames") or 0)
    if total > 0:
        return total

    video = timeline.get("video") or {}
    source_count = int(video.get("sourceFrameCount") or 0)
    if source_count > 0:
        removed = sum(end - start for start, end in deleted_source_ranges(timeline))
        return max(0, source_count - removed)

    clips = video_clips_from_timeline(timeline)
    if len(clips) > 1:
        return sum(int(c.get("sourceFrameCount") or 0) for c in clips)

    return 0


def resolve_logical_frame_entry(timeline: dict, logical_index: int) -> tuple[int, int]:
    """Map a logical timeline index to (clip_index, source_frame_index)."""
    video = timeline.get("video") or {}
    frame_map = video.get("frameMap") or []
    if logical_index < len(frame_map):
        return parse_frame_map_entry(frame_map[logical_index])

    src = logical_index
    for start, end in deleted_source_ranges(timeline):
        if src >= start:
            src += end - start
        else:
            break

    clips = video_clips_from_timeline(timeline)
    if len(clips) <= 1:
        return 0, src

    offset = 0
    for clip_idx, clip in enumerate(clips):
        count = int(clip.get("sourceFrameCount") or 0)
        if logical_index < offset + count:
            return clip_idx, logical_index - offset
        offset += count

    last = clips[-1]
    last_count = max(1, int(last.get("sourceFrameCount") or 1))
    return len(clips) - 1, last_count - 1


def _decode_timeline_entries(
    timeline: dict,
    entries: list[tuple[int, int]],
    *,
    frame_rate: float,
    default_long_edge: int,
) -> torch.Tensor:
    clips = video_clips_from_timeline(timeline)
    if not clips:
        raise ValueError("No video clips in MiniMax H3 Director Opt timeline.")

    by_clip: dict[int, set[int]] = defaultdict(set)
    for clip_idx, frame_idx in entries:
        if clip_idx < 0 or clip_idx >= len(clips):
            clip_idx = 0
        by_clip[clip_idx].add(frame_idx)

    frame_tensors: dict[tuple[int, int], torch.Tensor] = {}
    for clip_idx, frame_set in sorted(by_clip.items()):
        clip = clips[clip_idx]
        path = resolve_video_path(clip)
        sorted_idx = sorted(frame_set)
        tensor = load_video_resampled(
            path,
            frame_rate,
            sorted_idx,
            storage_width=clip.get("storageWidth"),
            storage_height=clip.get("storageHeight"),
            long_edge=int(clip.get("longEdge") or default_long_edge),
        )
        for row, fi in enumerate(sorted_idx):
            frame_tensors[(clip_idx, fi)] = tensor[row]

    rows: list[torch.Tensor] = []
    fallback: torch.Tensor | None = None
    for clip_idx, frame_idx in entries:
        if clip_idx < 0 or clip_idx >= len(clips):
            clip_idx = 0
        key = (clip_idx, frame_idx)
        tensor = frame_tensors.get(key, fallback)
        if tensor is None:
            raise ValueError(f"Missing decoded frame for clip {clip_idx} frame {frame_idx}")
        fallback = tensor
        rows.append(tensor)

    return torch.stack(rows, dim=0)


def load_timeline_segment(timeline: dict, start: int, end: int) -> torch.Tensor:
    """Decode only logical frames in [start, end) 鈥?supports arbitrarily long timelines."""
    total = logical_frame_count(timeline)
    start = max(0, min(int(start), total))
    end = max(start, min(int(end), total))
    if start >= end:
        raise ValueError(f"No frames in timeline range [{start}, {end})")

    video = timeline.get("video") or {}
    frames_b64 = video.get("frames") or []
    if frames_b64:
        chunks: list[torch.Tensor] = []
        for frame_b64 in frames_b64[start:end]:
            chunks.append(_decode_image_b64_inline(frame_b64))
        if not chunks:
            raise ValueError("Uploaded video has no decodable frames in range.")
        return torch.cat(chunks, dim=0)

    frame_rate = float(timeline.get("frameRate") or 24)
    output_block = timeline.get("output") or {}
    default_long_edge = int(
        output_block.get("longEdge")
        or output_block.get("long_edge")
        or timeline.get("refMaxSize")
        or 848
    )

    entries = [resolve_logical_frame_entry(timeline, i) for i in range(start, end)]
    return _decode_timeline_entries(
        timeline,
        entries,
        frame_rate=frame_rate,
        default_long_edge=default_long_edge,
    )


def _decode_image_b64_inline(b64_str: str) -> torch.Tensor:
    import base64
    import io

    from PIL import Image

    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def load_multi_clip_timeline(
    timeline: dict,
    frame_map: list[Any],
    *,
    frame_rate: float,
    default_long_edge: int,
) -> torch.Tensor:
    """Decode a logical timeline that may reference multiple source videos."""
    entries = [parse_frame_map_entry(e) for e in frame_map]
    return _decode_timeline_entries(
        timeline,
        entries,
        frame_rate=frame_rate,
        default_long_edge=default_long_edge,
    )
