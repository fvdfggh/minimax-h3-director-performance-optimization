"""Timeline ranges: export and second-sample windows, counting, clip trimming.

Everything that turns the timeline JSON into *which frames of which segment* a
request covers: the explicit「分段导出」/「二次采样」requests, the run-selection
parser, and the frame-count helpers those depend on.

``prepare_segment_clip`` and ``slice_video_frames`` live here because they are the
last step of that arithmetic — trimming a source clip to the aligned segment
length. ``prepare_segment_clip`` never pads with duplicated last frames: fabricated
freeze frames make the model reproduce visible stutter.
"""

from __future__ import annotations

import copy
import json
import logging

import torch

from ..lib.task_prompts import resolve_task_key
from ..lib.video_io import logical_frame_count
from .frame_align import minimax_align_frame_count
from .gen_timeline import is_gen_timeline
from .plan_types import (
    MIN_SEGMENT_FRAMES,
    SegmentExportRequest,
    SegmentSecondSampleRequest,
    normalize_segment_export_mode,
    normalize_segment_export_source,
    parse_run_selection,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.plan.ranges")


def _segment_ranges_from_timeline(timeline: dict, total: int) -> list[tuple[int, int, dict]]:
    segments = timeline.get("segments") or []
    if segments and ("length" in segments[0] or "end" in segments[0]):
        ranges: list[tuple[int, int, dict]] = []
        for raw in sorted(segments, key=lambda s: int(s.get("start", 0))):
            start = int(raw.get("start", 0))
            if "end" in raw:
                end = int(raw["end"])
            else:
                end = start + int(raw.get("length", 0))
            start = max(0, min(start, total))
            end = max(start, min(end, total))
            if end - start >= MIN_SEGMENT_FRAMES or not ranges:
                ranges.append((start, end, raw))
        if ranges:
            return ranges

    split_points = timeline.get("splitPoints") or timeline.get("split_points") or []
    auto_count = int(timeline.get("autoSegmentCount") or timeline.get("auto_segment_count") or 0)
    if auto_count > 1:
        points = [int(round(total * i / auto_count)) for i in range(1, auto_count)]
    else:
        points = sorted({int(p) for p in split_points if 0 < int(p) < total})

    edges = [0] + points + [total]
    ranges = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        if end <= start:
            continue
        raw = segments[i] if i < len(segments) else {}
        ranges.append((start, end, raw))
    return ranges or [(0, total, {})]


def _resolve_export_total(timeline: dict, source_total: int) -> int:
    output_block = timeline.get("output") or {}
    max_export = int(output_block.get("maxExportFrames") or output_block.get("max_export_frames") or 0)
    if max_export <= 0 or source_total <= 0:
        return source_total
    return min(source_total, max_export)


def parse_segment_export(timeline: dict, segment_count: int) -> SegmentExportRequest | None:
    """Read the「分段导出」block from ``timeline.output.segmentExport``.

    Returns ``None`` when the feature is off, so every existing call site keeps
    its current behaviour. Indices are clamped to the live timeline and
    de-duplicated; an enabled request with no valid index is reported as
    disabled rather than raising, letting the node fall back to a normal run.
    """
    if not isinstance(timeline, dict) or segment_count <= 0:
        return None
    output_block = timeline.get("output") or {}
    if not isinstance(output_block, dict):
        output_block = {}
    # The frontend spreads the segment-export block at the TOP LEVEL of the
    # timeline payload (see buildTimelinePayload: `...this._segmentExportPayload()`),
    # NOT under `output`. Accept both locations so indices are never missed.
    block = (timeline.get("segmentExport") if isinstance(timeline.get("segmentExport"), dict)
             else output_block.get("segmentExport"))
    if block is None:
        block = output_block.get("segment_export")
    if not isinstance(block, dict):
        return None

    enabled = bool(block.get("enabled") or block.get("active"))
    mode = normalize_segment_export_mode(
        block.get("mode") if block.get("mode") is not None else block.get("exportMode")
    )
    source = normalize_segment_export_source(
        block.get("source") if block.get("source") is not None else block.get("cacheSource")
    )

    raw = block.get("indices")
    if raw is None:
        raw = block.get("selection")
    indices: list[int] = []
    if isinstance(raw, list):
        for item in raw:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < segment_count and idx not in indices:
                indices.append(idx)

    if not indices:
        return SegmentExportRequest(enabled=False, mode=mode, indices=(), source=source)
    # ``enabled`` is the ONE-SHOT trigger set by the「分段导出」button; ``indices``
    # is persistent on purpose (the picker reopens with the last selection).
    # Both must be live. Forcing ``enabled`` True here (as this used to do) made
    # every later 运行 an export-only pass, because the checked indices never stop
    # being checked — the export button's flag is the only thing that distinguishes
    # "export this run" from "generate this run".
    return SegmentExportRequest(
        enabled=enabled, mode=mode, indices=tuple(sorted(indices)), source=source
    )


def parse_second_sample(timeline: dict, segment_count: int) -> "SegmentSecondSampleRequest | None":
    """Read the「二次采样」block from ``timeline.output.secondSample`` (or top-level).

    Returns ``None`` when the feature is off, so every existing call site keeps its
    current behaviour. Indices are clamped to the live timeline and de-duplicated;
    an enabled request with no valid index is reported as disabled rather than
    raising, letting the node fall back to a normal run.
    """
    if not isinstance(timeline, dict) or segment_count <= 0:
        return None
    output_block = timeline.get("output") or {}
    if not isinstance(output_block, dict):
        output_block = {}
    # The frontend spreads the block at the TOP LEVEL of the timeline payload
    # (mirroring the「分段导出」picker), NOT under `output`. Accept both.
    block = (
        timeline.get("secondSample")
        if isinstance(timeline.get("secondSample"), dict)
        else output_block.get("secondSample")
    )
    if block is None:
        block = output_block.get("second_sample")
    if not isinstance(block, dict):
        return None

    enabled = bool(block.get("enabled") or block.get("active"))
    raw = block.get("indices")
    if raw is None:
        raw = block.get("selection")
    indices: list[int] = []
    if isinstance(raw, list):
        for item in raw:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < segment_count and idx not in indices:
                indices.append(idx)

    if not indices:
        return SegmentSecondSampleRequest(enabled=False, indices=())
    # ``enabled`` is the ONE-SHOT trigger set by the「二次采样」button; ``indices``
    # is persistent on purpose (the picker reopens with the last selection).
    return SegmentSecondSampleRequest(enabled=enabled, indices=tuple(sorted(indices)))


def _clip_segment_ranges(
    ranges: list[tuple[int, int, dict]], export_total: int
) -> list[tuple[int, int, dict]]:
    if export_total <= 0:
        return ranges
    clipped: list[tuple[int, int, dict]] = []
    for start, end, data in ranges:
        if start >= export_total:
            break
        end = min(end, export_total)
        if end <= start:
            continue
        if end - start < MIN_SEGMENT_FRAMES and clipped:
            ps, _, pd = clipped[-1]
            clipped[-1] = (ps, end, pd)
        else:
            clipped.append((start, end, data))
    if not clipped and export_total > 0:
        data = ranges[0][2] if ranges else {}
        clipped.append((0, export_total, data))
    return clipped


def _trim_timeline_for_export(timeline: dict, export_total: int) -> dict:
    t = copy.deepcopy(timeline)
    video = dict(t.get("video") or {})
    frames_b64 = video.get("frames") or []
    if frames_b64 and export_total < len(frames_b64):
        video["frames"] = frames_b64[:export_total]
    frame_map = video.get("frameMap") or []
    if frame_map and export_total < len(frame_map):
        video["frameMap"] = frame_map[:export_total]
    t["video"] = video
    t["totalFrames"] = export_total
    return t


def count_all_timeline_segments(timeline_data: str) -> int:
    """Total segment count on the timeline (ignores run selection)."""
    if not timeline_data or not str(timeline_data).strip():
        return 1
    try:
        timeline = json.loads(timeline_data)
    except json.JSONDecodeError:
        return 1

    segments = timeline.get("segments") or []
    global_task = (timeline.get("global") or {}).get("taskType") or ""
    task_key = resolve_task_key(global_task) if global_task else ""
    if task_key == "fl2v" or str(timeline.get("timelineMode") or "").lower() == "fl2v":
        from .fl2v_timeline import count_fl2v_runnable_shots

        return count_fl2v_runnable_shots(timeline)
    if is_gen_timeline(timeline, task_key):
        return max(1, len(segments) or 1)

    source_total = logical_frame_count(timeline) or int(timeline.get("totalFrames") or 0)
    export_total = _resolve_export_total(timeline, source_total)
    plan_total = export_total or source_total or 1
    ranges = _segment_ranges_from_timeline(timeline, source_total or plan_total)
    return max(1, len(_clip_segment_ranges(ranges, plan_total)))


def count_timeline_segments(timeline_data: str) -> int:
    """Segments that will run (respects run selection when enabled)."""
    if not timeline_data or not str(timeline_data).strip():
        return 1
    try:
        timeline = json.loads(timeline_data)
    except json.JSONDecodeError:
        return 1

    seg_count = count_all_timeline_segments(timeline_data)
    run_sel = parse_run_selection(timeline, seg_count)
    return len(run_sel) if run_sel is not None else seg_count


def slice_video_frames(source: torch.Tensor, start: int, end: int) -> torch.Tensor:
    end = min(end, source.shape[0])
    start = max(0, min(start, end))
    return source[start:end].clone()


def prepare_segment_clip(clip: torch.Tensor, target_frames: int) -> tuple[torch.Tensor, int]:
    """Trim source toward MiniMax 17k+5 length. Do **not** pad with last-frame copies.

    Fabricating freeze frames in the source makes Bernini/Wan reproduce visible
    stutter / duplicate frames. Official BerniniConditioning simply encodes
    ``source[:length]`` even when the clip is shorter than ``length``.
    """
    actual = clip.shape[0]
    if actual <= 0:
        raise ValueError("Segment has no frames.")
    num_frames = minimax_align_frame_count(max(actual, target_frames))
    if actual > num_frames:
        clip = clip[:num_frames]
    return clip, num_frames
