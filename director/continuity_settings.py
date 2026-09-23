"""Reading continuity settings from a timeline, and locating the previous segment.

The MiniMax H3 path is opt-in per timeline (「段间引导」). Everything here answers
one of three questions: is continuity on, how much overlap / redraw, and what is
the previous segment's output to carry forward.

Kept apart from :mod:`continuity_seam` (which grades pixels) and
:mod:`segment_continuity` (which concatenates), so that changing how a setting is
read can never accidentally touch the seam maths.
"""

from __future__ import annotations

import math

import torch

from .cache_readback import load_segment_cache
from .continuity_knobs import (
    CONTINUITY_REDRAW_MAX,
    CONTINUITY_REDRAW_MIN,
    DEFAULT_CONTINUITY_REDRAW,
)
from .h3_motion_context import (
    CONTINUITY_TASK_KEYS,
    DEFAULT_CONTEXT_FRAMES as DEFAULT_CONTINUITY_OVERLAP,
    snap_context_frames,
)
from .plan_types import DirectorPlan, SegmentPlan


def _truthy_continuity_flag(value) -> bool:
    """Accept bool/int/common string forms from UI or reloaded workflows."""
    if value is True or value == 1:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return False


def resolve_continuity_settings(timeline: dict, *, segment_count: int) -> tuple[bool, int]:
    """Read segment continuity flags from timeline JSON (output only; default off)."""
    if segment_count < 2:
        return False, 0
    output = timeline.get("output") or {}
    enabled = _truthy_continuity_flag(
        output.get("continuityEnabled", output.get("continuity_enabled"))
    )
    if not enabled:
        return False, 0
    raw = (
        output.get("continuityOverlapFrames")
        or output.get("continuity_overlap_frames")
        or DEFAULT_CONTINUITY_OVERLAP
    )
    return True, snap_context_frames(raw)


def resolve_continuity_redraw(timeline: dict) -> float:
    """Read「重绘幅度」from timeline JSON (output.continuityRedraw). Default 0.10."""
    output = (timeline or {}).get("output") or {}
    raw = (
        output.get("continuityRedraw")
        or output.get("continuity_redraw")
        or output.get("continueSeam")
    )
    try:
        n = float(raw)
    except (TypeError, ValueError):
        n = DEFAULT_CONTINUITY_REDRAW
    if not math.isfinite(n):
        n = DEFAULT_CONTINUITY_REDRAW
    return max(CONTINUITY_REDRAW_MIN, min(CONTINUITY_REDRAW_MAX, n))


def resolve_segment_continuity_from_prev(
    seg_data: dict | None,
    *,
    segment_index: int,
) -> bool:
    """Per-segment「引用上段」flag.

    Master「段间引导」must also be on (checked by ``is_continuity_active``).
    Missing field defaults to True so existing workflows keep pinning every segment.
    Segment index 0 never pins.
    """
    if int(segment_index) <= 0:
        return False
    if not isinstance(seg_data, dict):
        return True
    if "continuityFromPrev" in seg_data:
        raw = seg_data.get("continuityFromPrev")
    elif "continuity_from_prev" in seg_data:
        raw = seg_data.get("continuity_from_prev")
    else:
        return True
    return _truthy_continuity_flag(raw)


def resolve_segment_continuity_to_next(
    seg_data: dict | None,
    *,
    segment_index: int,
    segment_count: int = 0,
) -> bool:
    """Per-segment「对齐下段」flag (pin the next segment's opening into this tail).

    Master「段间引导」must also be on (checked by ``is_continuity_active``).
    Missing field defaults to False — unlike「引用上段」(default True), aligning
    to the next segment is strictly opt-in: it is a cache-driven, middle-out
    mode and only means something when that neighbour already holds an AV
    latent. The last segment never pins (there is no next segment).
    """
    if int(segment_index) < 0:
        return False
    if segment_count and int(segment_index) >= int(segment_count) - 1:
        return False
    if not isinstance(seg_data, dict):
        return False
    if "continuityToNext" in seg_data:
        raw = seg_data.get("continuityToNext")
    elif "continuity_to_next" in seg_data:
        raw = seg_data.get("continuity_to_next")
    else:
        return False
    return _truthy_continuity_flag(raw)


def timeline_row_for_index(timeline: dict | None, index: int) -> dict:
    """Best-effort segment/shot/group row from timeline for per-segment flags."""
    if not isinstance(timeline, dict) or int(index) < 0:
        return {}
    for key in ("segments", "shots", "r2vGroups", "fl2vGroups"):
        rows = timeline.get(key)
        if isinstance(rows, list) and int(index) < len(rows):
            row = rows[int(index)]
            if isinstance(row, dict):
                return row
    return {}


def _proxy_mad(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cheap mean-abs diff on a spatial proxy (RGB only)."""
    if a.shape[0] != b.shape[0]:
        return 1e9
    aa = a[..., :3].float()
    bb = b[..., :3].float()
    # Downsample for speed; values are 0..1 tensors from ComfyUI.
    step_h = max(1, int(aa.shape[1]) // 64)
    step_w = max(1, int(aa.shape[2]) // 64)
    aa = aa[:, ::step_h, ::step_w, :]
    bb = bb[:, ::step_h, ::step_w, :]
    # Scale to ~0..255 so thresholds match the offline video diagnostics.
    return float((aa - bb).abs().mean().item() * 255.0)


def is_continuity_active(plan: DirectorPlan, seg: SegmentPlan) -> bool:
    """True only when UI「段间引导」is ON and this segment should pin the previous.

    When False, the executor must stay on the official MiniMax H3 path
    (stock ImageToVideo / ReferenceToVideo, no motion-context pin/patch/trim).
    Supported tasks: t2v / i2v / fl2v / r2v / v2v / rv2v.
    Per-segment ``continuity_from_prev`` (default True) can opt out while master is on.
    """
    return (
        plan.continuity_enabled
        and plan.segment_count >= 2
        and seg.index > 0
        and seg.task_key in CONTINUITY_TASK_KEYS
        and bool(getattr(seg, "continuity_from_prev", True))
    )


def resolve_prev_segment_output(
    plan: DirectorPlan,
    all_segments: list[SegmentPlan],
    seg_index: int,
    completed: dict[int, torch.Tensor],
    node_id: str | None,
    workflow_name: str | None = None,
) -> torch.Tensor | None:
    prev_idx = seg_index - 1
    if prev_idx < 0:
        return None
    if prev_idx in completed:
        return completed[prev_idx]
    prev_seg = all_segments[prev_idx]
    # Pipeline-stale is ok; a different source video is not (load_segment_cache
    # refuses source-stale even with allow_stale=True).
    cached = load_segment_cache(node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name)
    if cached is not None:
        return cached
    if not plan.continuity_enabled:
        return None
    raise ValueError(
        f"段间连贯：片段 #{seg_index + 1} 需要上一段 #{prev_idx + 1} 的生成结果。"
        "换源后旧缓存已失效。请先运行上一段，或将其纳入「选择运行」；"
        "也可关闭「段间引导」后只跑本段。"
    )
