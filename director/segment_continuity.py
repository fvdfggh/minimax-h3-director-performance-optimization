"""Cross-segment continuity helpers for MiniMax H3 Director Opt.

Active path (opt-in「段间引导」): motion-context pin via
``director.h3_motion_context`` — previous segment AV tail → next segment
conditioning, then trim the pinned prefix.
Tasks: t2v / i2v / fl2v / r2v / v2v / rv2v.

The Wan/SCAIL-era surface — luma matching, lock/unlock feathering, free-latent
warm-start and the additive-luma / opening-Y-map seam blends — was never read by
the MiniMax H3 pipeline and has been removed. What remains is the concat / seam
grading this pipeline actually runs (``concat_chunks_lazy`` et al.).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch

from ..lib.image_prep import (
    cat_frames_variable_size, fit_canvas, fit_long_edge, pad_frames_to_canvas,
)
from .h3_motion_context import (
    CONTINUITY_TASK_KEYS,
    DEFAULT_CONTEXT_FRAMES as DEFAULT_CONTINUITY_OVERLAP,
    snap_context_frames,
)
from . import segment_slots
from .plan_types import (
    MAX_CONTINUITY_OVERLAP,
    MIN_CONTINUITY_OVERLAP,
    DirectorPlan,
    SegmentPlan,
)
from .segment_cache import load_segment_cache

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.continuity")

# --- Continuity knobs still read by the active MiniMax H3 path ----------------
# The upstream Wan/SCAIL-era tuning set (seam echo / join limits, lock & unlock
# feather masks, free-latent warm-start ramps, additive-luma and opening-Y-map
# blends) was never read by this pipeline and has been removed.

# No multiplicative gain / long RGB blend (画面花 / 幻影).
CONTINUITY_SEAM_SOFTEN_FRAMES = 0

CONTINUITY_TAIL_LUMA_BLEND = 0
CONTINUITY_OPENING_EXPOSURE_SMOOTH = 0
CONTINUITY_OPENING_LUMA_BLEND = 0

# Concat: additive luma ONLY — body0/hold→pop RGB caused 拖影+一顿一顿 (00035).
CONTINUITY_SEAM_ADD_LUMA_FRAMES = 12

# Concat opening grade: low-freq appearance pull replaces the additive mean-luma
# nudge in the seam pipeline (no 重影; pulls luma+chroma low-freq field instead).
CONTINUITY_EXPORT_GRADE_FRAMES = 12
CONTINUITY_EXPORT_GRADE_WEIGHT = 0.70
CONTINUITY_EXPORT_GRADE_BLUR = 64
CONTINUITY_BODY0_SEAM_WEIGHT = 0.0
CONTINUITY_BODY1_SEAM_WEIGHT = 0.0
CONTINUITY_MICRO_SEAM_MAD = 99.0
CONTINUITY_MICRO_SEAM_WEIGHT = 0.0
CONTINUITY_MICRO_SEAM_WEIGHT_MAX = 0.0
CONTINUITY_MICRO_SEAM_SECOND_WEIGHT = 0.0
CONTINUITY_MICRO_SEAM_THIRD_WEIGHT = 0.0
CONTINUITY_MICRO_SEAM_MAD_SPAN = 8.0

CONTINUITY_TAIL_SOFTEN_FRAMES = 0
CONTINUITY_HOLD_MAX_FRAMES = 0
CONTINUITY_HOLD_MAD = 4.5
CONTINUITY_HOLD_POP_JUMP_MAD = 10.0
CONTINUITY_HOLD_POP_BRIDGE_WEIGHT = 0.0
CONTINUITY_HOLD_POP_BRIDGE_WEIGHT_MAX = 0.0
CONTINUITY_HOLD_POP_LAND_WEIGHT = 0.0
CONTINUITY_HOLD_POP_LOOKAHEAD = 2
CONTINUITY_HOLD_POP_SCAN = 6
CONTINUITY_HOLD_POP_LOWFREQ = False
CONTINUITY_HOLD_POP_BLUR = 32
CONTINUITY_SPIKE_PREV_MAD = 6.5
CONTINUITY_SPIKE_JUMP_MAD = 12.0
CONTINUITY_SPIKE_WEIGHT = 0.0
CONTINUITY_SPIKE_LAND_WEIGHT = 0.0
CONTINUITY_SPIKE_SCAN = 5
CONTINUITY_HOLD_POP_ON_TAIL = False


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


# 段间「锥形重绘」幅度（=上游 seam_min_mask）。0 = 接缝硬锁（前缀几乎全重绘、仅
# 接缝保留旧尾）；0.95 = 几乎不重绘。沿用上游成熟默认 0.10。
DEFAULT_CONTINUITY_REDRAW = 0.10
CONTINUITY_REDRAW_MIN = 0.0
CONTINUITY_REDRAW_MAX = 0.95


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


















def _blur_hwc(frame: torch.Tensor, kernel: int) -> torch.Tensor:
    """Box-blur an HWC frame (reflect pad). kernel forced odd >=3."""
    k = int(kernel)
    if k < 3:
        return frame.float()
    if k % 2 == 0:
        k += 1
    x = frame.detach().float()
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    t = x.permute(2, 0, 1).unsqueeze(0)
    pad = k // 2
    t = torch.nn.functional.pad(t, (pad, pad, pad, pad), mode="reflect")
    t = torch.nn.functional.avg_pool2d(t, kernel_size=k, stride=1)
    return t.squeeze(0).permute(1, 2, 0)


def _lowfreq_appearance_pull(
    src: torch.Tensor,
    guide: torch.Tensor,
    *,
    weight: float,
    blur: int,
) -> torch.Tensor:
    """Keep src detail; absorb guide's low-frequency grade/lighting only.

    Full RGB lerp copies pose edges → 重影 + hold→pop. Low-freq residual
    transfer matches appearance without a second silhouette.
    """
    w = float(weight)
    if w <= 0:
        return src
    g = guide
    if g.dim() == 4:
        g = g[0]
    if tuple(g.shape[:2]) != tuple(src.shape[:2]):
        g = fit_canvas(g.unsqueeze(0), int(src.shape[1]), int(src.shape[0]))[0]
    b_src = _blur_hwc(src, blur)
    b_guide = _blur_hwc(g, blur)
    out = src.float() + w * (b_guide - b_src)
    return out.clamp(0.0, 1.0).to(dtype=src.dtype)


def _grade_device() -> torch.device:
    """Device for the bridge opening grade.

    The grade is a per-pixel box blur plus an elementwise lerp. Neither couples
    a pixel to any other frame or to its neighbours' ordering, so the GPU
    reproduces the CPU numbers to float32 rounding while turning a multi-frame
    pass from CPU into a single batched call. ``H3_DIRECTOR_GRADE_DEVICE=cpu``
    forces the original CPU path (useful for A/B or when VRAM is tight).
    """
    forced = os.environ.get("H3_DIRECTOR_GRADE_DEVICE", "").strip().lower()
    if forced in {"cpu", "off", "0"}:
        return torch.device("cpu")
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def _box_blur_bhwc(
    frames: torch.Tensor, kernel: int, device: torch.device
) -> torch.Tensor:
    """Batch form of :func:`_blur_hwc` — same reflect pad, same box kernel.

    ``avg_pool2d`` is independent per sample, so blurring N frames in one call
    on one device is identical to blurring them one at a time.
    """
    k = int(kernel)
    if k < 3:
        return frames.detach().to(device=device, dtype=torch.float32)
    if k % 2 == 0:
        k += 1
    x = frames.detach().to(device=device, dtype=torch.float32)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    t = x.permute(0, 3, 1, 2)
    pad = k // 2
    t = torch.nn.functional.pad(t, (pad, pad, pad, pad), mode="reflect")
    t = torch.nn.functional.avg_pool2d(t, kernel_size=k, stride=1)
    return t.permute(0, 2, 3, 1)


def _lowfreq_appearance_pull_batched(
    body: torch.Tensor,
    guide: torch.Tensor,
    weights: list,
    blur: int,
    device: torch.device,
) -> torch.Tensor:
    """Batched GPU form of the per-frame :func:`_lowfreq_appearance_pull`.

    ``guide`` is constant across frames, so its blur is hoisted out of the loop
    and the per-frame box blurs become one batched ``avg_pool2d``. Arithmetic-
    neutral vs the frame-at-a-time version: same low-frequency residual, same
    per-frame weight, same clamp.
    """
    cnt = len(weights)
    src = body[:cnt]
    g = guide
    if g.dim() == 4:
        g = g[0]
    if tuple(g.shape[:2]) != tuple(src.shape[1:3]):
        g = fit_canvas(g.unsqueeze(0), int(src.shape[2]), int(src.shape[1]))[0]
    b_guide = _box_blur_bhwc(g.unsqueeze(0), blur, device)[0]
    b_src = _box_blur_bhwc(src, blur, device)
    w = torch.tensor(weights, device=device, dtype=torch.float32).view(-1, 1, 1, 1)
    s = src.detach().to(device=device, dtype=torch.float32)
    res = (s + w * (b_guide.unsqueeze(0) - b_src)).clamp_(0.0, 1.0)
    out = body.clone()
    out[:cnt] = res.to(device=body.device, dtype=body.dtype)
    return out


def match_export_opening_grade(
    body: torch.Tensor,
    guide: torch.Tensor,
    *,
    frames: int = CONTINUITY_EXPORT_GRADE_FRAMES,
    weight0: float = CONTINUITY_EXPORT_GRADE_WEIGHT,
    blur: int = CONTINUITY_EXPORT_GRADE_BLUR,
) -> torch.Tensor:
    """Match opening lighting/grade of a concatenated clip to the previous tail.

    Low-frequency residual only (box blur) so pose edges are not copied (no 重影).
    Replaces the additive mean-luma nudge in the concat seam pipeline: pulls the
    blurred lighting+chroma field toward ``guide[-1]`` over the opening ``frames``,
    weight decaying to 0 — stronger and frequency-faithful, no ghosting.
    """
    if (
        body is None
        or guide is None
        or int(body.shape[0]) < 1
        or int(guide.shape[0]) < 1
        or int(frames) < 1
        or float(weight0) <= 0
        or int(blur) < 3
    ):
        return body
    n = min(int(frames), int(body.shape[0]))
    last = guide[-1]
    weights: list = []
    for i in range(n):
        w = float(weight0) * (1.0 - float(i) / float(n))
        if w <= 1e-4:
            break
        weights.append(w)
    if not weights:
        return body
    out = body.clone()
    try:
        out = _lowfreq_appearance_pull_batched(out, last, weights, int(blur), _grade_device())
    except Exception as exc:  # no CUDA / OOM / driver surprise
        log.warning(
            "Segment continuity: concat opening grade fell back to CPU (%s: %s)",
            type(exc).__name__,
            exc,
        )
        for i, w in enumerate(weights):
            out[i] = _lowfreq_appearance_pull(out[i], last, weight=w, blur=int(blur))
    log.info(
        "Segment continuity: concat opening grade %df weight=%.2f blur=%d",
        n,
        float(weight0),
        int(blur),
    )
    return out


def _soften_body0_toward_prev(
    body: torch.Tensor,
    guide: torch.Tensor,
    *,
    weight0: float = CONTINUITY_BODY0_SEAM_WEIGHT,
    weight1: float = CONTINUITY_BODY1_SEAM_WEIGHT,
) -> torch.Tensor:
    """Unilateral ease of body opening toward prev[-1] — no prev-tail rewrite.

    00033 顿感: hard cut / opening brake. Tiny pull on body[0]/[1] restores join
    softness without painting the next pose into the previous ending.
    Weight scales up slightly when the cut MAD is high.
    """
    if (
        body is None
        or guide is None
        or int(body.shape[0]) <= 0
        or int(guide.shape[0]) <= 0
        or (weight0 <= 0 and weight1 <= 0)
    ):
        return body
    last = guide[-1]
    if last.dim() == 4:
        last = last[0]
    if tuple(last.shape) != tuple(body[0].shape):
        last = fit_canvas(last.unsqueeze(0), int(body.shape[2]), int(body.shape[1]))[0]
    cut = _frame_mad(last, body[0])
    # Soft boost when cut is harsh; keep mild (v16 boost→ghost).
    boost = 1.0 + max(0.0, min(0.25, (cut - 10.0) / 16.0))
    w0 = min(0.10, float(weight0) * boost)
    w1 = min(0.05, float(weight1) * boost)
    out = body.clone()
    last_f = last.float()
    if w0 > 0:
        out[0] = (
            (out[0].float() * (1.0 - w0) + last_f * w0)
            .clamp(0.0, 1.0)
            .to(dtype=out.dtype)
        )
    if w1 > 0 and int(out.shape[0]) > 1:
        out[1] = (
            (out[1].float() * (1.0 - w1) + last_f * w1)
            .clamp(0.0, 1.0)
            .to(dtype=out.dtype)
        )
    log.info(
        "Segment continuity: unilateral opening seam ease "
        "w=%.2f/%.2f (cutMAD=%.1f boost=%.2f)",
        w0,
        w1,
        cut,
        boost,
    )
    return out


def _frame_mad(a: torch.Tensor, b: torch.Tensor) -> float:
    """Single-frame MAD on ~0..255 scale (matches offline diagnostics / SEAM_*)."""
    aa = a.detach().float()
    bb = b.detach().float()
    if aa.dim() == 3:
        aa = aa.unsqueeze(0)
        bb = bb.unsqueeze(0)
    return _proxy_mad(aa, bb)




def _unfreeze_held_tail(
    left: torch.Tensor,
    *,
    max_frames: int = CONTINUITY_HOLD_MAX_FRAMES,
    hold_mad: float = CONTINUITY_HOLD_MAD,
) -> torch.Tensor:
    """Spread a near-cut freeze across the tail using *only* left's own last frame.

    Morphing toward ``next[0]`` (00009) paints the next pose into the previous
    ending — visible 幻影. Intra-lerp removes the hold without cross-segment mix.
    """
    if left is None or int(left.shape[0]) < 3 or int(max_frames) <= 0:
        return left

    hold_n = 0
    limit = min(int(max_frames), int(left.shape[0]) - 1)
    for i in range(1, limit + 1):
        if _frame_mad(left[-i], left[-(i + 1)]) <= hold_mad:
            hold_n = i
            break
    if hold_n <= 0:
        return left

    out = left.clone()
    anchor = out[-(hold_n + 1)].float()
    end = out[-1].float()
    # If the freeze is the last pair, end≈anchor — nothing to spread.
    if _frame_mad(anchor, end) <= hold_mad:
        return left
    for j in range(hold_n):
        t = float(j + 1) / float(hold_n + 1)
        idx = -(hold_n - j)
        out[idx] = (anchor * (1.0 - t) + end * t).clamp(0.0, 1.0).to(dtype=out.dtype)
    log.info(
        "Segment continuity: unfreeze held tail %df (hold_mad≤%.1f, intra only)",
        hold_n,
        hold_mad,
    )
    return out


def _micro_seam_bridge(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    mad_threshold: float = CONTINUITY_MICRO_SEAM_MAD,
    weight: float = CONTINUITY_MICRO_SEAM_WEIGHT,
    second_weight: float = CONTINUITY_MICRO_SEAM_SECOND_WEIGHT,
    third_weight: float = CONTINUITY_MICRO_SEAM_THIRD_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilateral 1–2 frame soften; weight scales with seam MAD (capped)."""
    if (
        left is None
        or right is None
        or int(left.shape[0]) <= 0
        or int(right.shape[0]) <= 0
        or weight <= 0
    ):
        return left, right
    a = left[-1]
    b = right[0]
    if b.dim() == 4:
        b = b[0]
    if a.dim() == 4:
        a = a[0]
    if tuple(a.shape) != tuple(b.shape):
        b = fit_canvas(b.unsqueeze(0), int(left.shape[2]), int(left.shape[1]))[0]
    seam_mad = _frame_mad(a, b)
    if seam_mad < mad_threshold:
        return left, right

    # Harder cut → stronger bridge, but never above WEIGHT_MAX (00020 ghost zone).
    span = max(1.0, float(CONTINUITY_MICRO_SEAM_MAD_SPAN))
    t = max(0.0, min(1.0, (seam_mad - mad_threshold) / span))
    w0 = float(weight) + (float(CONTINUITY_MICRO_SEAM_WEIGHT_MAX) - float(weight)) * t
    w1 = w0 * float(second_weight) if 0 < float(second_weight) < 1.0 else float(second_weight)
    w2 = float(third_weight)
    if 0 < w2 < 1.0:
        w2 = w0 * w2

    out_l = left.clone()
    out_r = right.clone()
    a_f = a.float()
    b_f = b.float()
    weights = (w0, w1, w2)
    for k, wk in enumerate(weights):
        if wk <= 0:
            continue
        if int(out_l.shape[0]) > k:
            idx = -(k + 1)
            out_l[idx] = (
                (out_l[idx].float() * (1.0 - wk) + b_f * wk)
                .clamp(0.0, 1.0)
                .to(dtype=out_l.dtype)
            )
        if int(out_r.shape[0]) > k:
            out_r[k] = (
                (out_r[k].float() * (1.0 - wk) + a_f * wk)
                .clamp(0.0, 1.0)
                .to(dtype=out_r.dtype)
            )
    log.info(
        "Segment continuity: micro seam bridge MAD=%.1f → w=%.2f/%.2f/%.2f",
        seam_mad,
        weights[0],
        weights[1],
        weights[2],
    )
    return out_l, out_r


def _hold_pop_weight(jump: float, *, base: float, jump_mad: float) -> float:
    """Scale bridge weight with jump size; cap below ghost zone."""
    span = 10.0
    t = max(0.0, min(1.0, (float(jump) - float(jump_mad)) / span))
    w_max = float(CONTINUITY_HOLD_POP_BRIDGE_WEIGHT_MAX)
    return float(base) + (w_max - float(base)) * t


def _find_hold_pop_jump(
    frames: torch.Tensor,
    *,
    hold_start: int,
    hold_end: int,
    jump_mad: float,
    lookahead: int,
) -> tuple[int, float] | None:
    """After a hold run, find the pop within ``lookahead`` pairs (incl. delayed)."""
    n = int(frames.shape[0])
    # hold_end is first index where pair mad may exceed hold; search from there.
    last = min(hold_end + max(0, int(lookahead)), n - 2)
    for k in range(hold_end, last + 1):
        jump = _frame_mad(frames[k], frames[k + 1])
        if jump >= jump_mad:
            return k, jump
    return None


def _apply_hold_pop_rewrite(
    out: torch.Tensor,
    *,
    i: int,
    jump_at: int,
    jump: float,
    bridge_weight: float,
    jump_mad: float,
    land_weight: float,
) -> float:
    """Unstick the held frame before a pop via low-freq pull (anti-ghost)."""
    del i, land_weight
    w = _hold_pop_weight(jump, base=bridge_weight, jump_mad=jump_mad)
    target = out[jump_at + 1]
    idx = jump_at
    t = 0.70 * w
    if CONTINUITY_HOLD_POP_LOWFREQ:
        out[idx] = _lowfreq_appearance_pull(
            out[idx],
            target,
            weight=t,
            blur=int(CONTINUITY_HOLD_POP_BLUR),
        )
    else:
        out[idx] = (
            (out[idx].float() * (1.0 - t) + target.float() * t)
            .clamp(0.0, 1.0)
            .to(dtype=out.dtype)
        )
    return w


def _break_hold_pop_window(
    frames: torch.Tensor,
    *,
    scan: int = CONTINUITY_HOLD_POP_SCAN,
    hold_mad: float = CONTINUITY_HOLD_MAD,
    jump_mad: float = CONTINUITY_HOLD_POP_JUMP_MAD,
    bridge_weight: float = CONTINUITY_HOLD_POP_BRIDGE_WEIGHT,
    land_weight: float = CONTINUITY_HOLD_POP_LAND_WEIGHT,
    lookahead: int = CONTINUITY_HOLD_POP_LOOKAHEAD,
    from_end: bool = False,
) -> torch.Tensor:
    """Rewrite hold→(mid)→pop runs with adaptive intra bridge + landing ease.

    00025 often holds, drifts for 1 mid frame (MAD≈5–8), then pops — looking
    only at the pair right after hold misses those. No cross-segment mix.
    """
    if (
        frames is None
        or int(frames.shape[0]) < 4
        or int(scan) < 3
        or bridge_weight <= 0
    ):
        return frames
    n = int(frames.shape[0])
    limit = min(int(scan), n - 2)
    if limit < 1:
        return frames

    def _scan_forward(src: torch.Tensor, start: int, end: int) -> torch.Tensor:
        out = src
        touched = False
        i = start
        while i < end:
            if _frame_mad(out[i], out[i + 1]) > hold_mad:
                i += 1
                continue
            j = i
            while (
                j + 1 < n - 1
                and j - i < 3
                and _frame_mad(out[j], out[j + 1]) <= hold_mad
            ):
                j += 1
            found = _find_hold_pop_jump(
                out,
                hold_start=i,
                hold_end=j,
                jump_mad=jump_mad,
                lookahead=lookahead,
            )
            if found is None:
                i = j + 1
                continue
            jump_at, jump = found
            if not touched:
                out = src.clone()
                touched = True
            w = _apply_hold_pop_rewrite(
                out,
                i=i,
                jump_at=jump_at,
                jump=jump,
                bridge_weight=bridge_weight,
                jump_mad=jump_mad,
                land_weight=land_weight,
            )
            log.info(
                "Segment continuity: hold→pop break %s @%d→%d jump=%.1f "
                "(hold≤%.1f w=%.2f)",
                "tail" if from_end else "opening",
                i,
                jump_at,
                jump,
                hold_mad,
                w,
            )
            i = jump_at + 1
        return out

    if from_end:
        start = max(0, n - 1 - limit)
        best = None  # (i, jump_at, jump)
        i = start
        while i < n - 2:
            if _frame_mad(frames[i], frames[i + 1]) > hold_mad:
                i += 1
                continue
            j = i
            while (
                j + 1 < n - 1
                and j - i < 3
                and _frame_mad(frames[j], frames[j + 1]) <= hold_mad
            ):
                j += 1
            found = _find_hold_pop_jump(
                frames,
                hold_start=i,
                hold_end=j,
                jump_mad=jump_mad,
                lookahead=lookahead,
            )
            if found is not None:
                jump_at, jump = found
                best = (i, jump_at, jump)
            i = j + 1
        if best is None:
            return frames
        best_i, best_jump_at, best_jump = best
        out = frames.clone()
        w = _apply_hold_pop_rewrite(
            out,
            i=best_i,
            jump_at=best_jump_at,
            jump=best_jump,
            bridge_weight=bridge_weight,
            jump_mad=jump_mad,
            land_weight=land_weight,
        )
        log.info(
            "Segment continuity: hold→pop break tail @%d→%d jump=%.1f "
            "(hold≤%.1f w=%.2f)",
            best_i,
            best_jump_at,
            best_jump,
            hold_mad,
            w,
        )
        return out

    return _scan_forward(frames, 0, limit)


def _ease_opening_spikes(
    frames: torch.Tensor,
    *,
    scan: int = CONTINUITY_SPIKE_SCAN,
    prev_mad: float = CONTINUITY_SPIKE_PREV_MAD,
    jump_mad: float = CONTINUITY_SPIKE_JUMP_MAD,
    weight: float = CONTINUITY_SPIKE_WEIGHT,
    land_weight: float = CONTINUITY_SPIKE_LAND_WEIGHT,
) -> torch.Tensor:
    """Ease soft-hesitation→hard-pop in the first few body frames (00024 seam100)."""
    if (
        frames is None
        or int(frames.shape[0]) < 3
        or int(scan) < 1
        or weight <= 0
    ):
        return frames
    n = int(frames.shape[0])
    limit = min(int(scan), n - 2)
    out = frames
    touched = False
    for i in range(limit):
        m0 = _frame_mad(out[i], out[i + 1])
        m1 = _frame_mad(out[i + 1], out[i + 2])
        if m0 > prev_mad or m1 < jump_mad:
            continue
        # Skip if already a classic hold (handled by hold→pop); only soft band.
        if m0 <= float(CONTINUITY_HOLD_MAD):
            continue
        if not touched:
            out = frames.clone()
            touched = True
        t = float(weight)
        out[i + 1] = (
            (out[i + 1].float() * (1.0 - t) + out[i + 2].float() * t)
            .clamp(0.0, 1.0)
            .to(dtype=out.dtype)
        )
        lw = float(land_weight)
        if lw > 0:
            out[i + 2] = (
                (out[i + 2].float() * (1.0 - lw) + out[i + 1].float() * lw)
                .clamp(0.0, 1.0)
                .to(dtype=out.dtype)
            )
        log.info(
            "Segment continuity: opening spike ease @%d (prev=%.1f jump=%.1f w=%.2f)",
            i,
            m0,
            m1,
            t,
        )
    return out
























def apply_scail_continuity_core(
    *,
    plan: DirectorPlan,
    seg: SegmentPlan,
    prev_output: torch.Tensor | None,
    positive,
    negative,
    vae,
    width: int,
    height: int,
    ref_max_size: int = 848,
    latent: dict[str, Any] | None = None,
    source_luma_ref: torch.Tensor | None = None,
) -> tuple[Any, Any, dict[str, Any] | None, str | None]:
    """SCAIL is Wan-only; MiniMax H3 uses last-frame handoff in executor."""
    del plan, seg, prev_output, vae, width, height, ref_max_size, source_luma_ref
    return positive, negative, latent, None








def concat_continuous_chunks(
    chunks: list[torch.Tensor],
    segments: list[SegmentPlan],
    plan: DirectorPlan,
) -> torch.Tensor:
    """Concatenate with exposure-only seam fix.

    Generation settling burn-in handles flash/pulse. Concat RGB morphs are OFF
    (00035: body0/hold→pop caused 拖影 and stutter pulses).
    """
    del segments
    if not chunks:
        raise ValueError("concat_continuous_chunks: no chunks")
    if not getattr(plan, "continuity_enabled", False) or len(chunks) < 2:
        return cat_frames_variable_size(chunks)
    fixed: list[torch.Tensor] = [chunks[0]]
    for i in range(1, len(chunks)):
        left = _unfreeze_held_tail(fixed[-1])
        if CONTINUITY_HOLD_POP_ON_TAIL:
            left = _break_hold_pop_window(left, from_end=True)
        body = _break_hold_pop_window(chunks[i], from_end=False)
        if float(CONTINUITY_SPIKE_WEIGHT) > 0:
            body = _ease_opening_spikes(body)
        body = _soften_body0_toward_prev(body, left)
        body = match_export_opening_grade(body, left)
        left, body = _micro_seam_bridge(left, body)
        fixed[-1] = left
        fixed.append(body)
    return cat_frames_variable_size(fixed)


def _seam_window() -> int:
    """Frames on each side of a join that the seam pipeline can rewrite.

    Every helper below only ever touches the leading / trailing few frames, so a
    window this size is enough to reproduce the maths exactly while keeping each
    internal ``.clone()`` at ~16 frames instead of the whole merged clip.
    """
    return max(
        int(CONTINUITY_SEAM_ADD_LUMA_FRAMES),
        int(CONTINUITY_SPIKE_SCAN),
        int(CONTINUITY_HOLD_POP_SCAN),
        int(CONTINUITY_OPENING_LUMA_BLEND),
        int(CONTINUITY_OPENING_EXPOSURE_SMOOTH),
        int(CONTINUITY_SEAM_SOFTEN_FRAMES),
        int(CONTINUITY_TAIL_SOFTEN_FRAMES),
        int(CONTINUITY_TAIL_LUMA_BLEND),
        3,  # _micro_seam_bridge rewrites up to 3 frames per side
    ) + 4


def _seam_fix_windows(
    left_tail: torch.Tensor,
    body_head: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the seam pipeline on two small windows (``left_tail`` / ``body_head``).

    Same order and same helpers as ``concat_continuous_chunks``, so the result is
    bit-identical to passing the full clips — but each helper's internal
    ``.clone()`` now costs one window rather than the accumulated merge. That
    clone was the OOM: the join used to clone everything merged so far once per
    segment, i.e. O(N^2) traffic and a transient third copy of the whole video.
    """
    if left_tail is None or body_head is None:
        return left_tail, body_head
    left = _unfreeze_held_tail(left_tail)
    if CONTINUITY_HOLD_POP_ON_TAIL:
        left = _break_hold_pop_window(left, from_end=True)
    body = _break_hold_pop_window(body_head, from_end=False)
    if float(CONTINUITY_SPIKE_WEIGHT) > 0:
        body = _ease_opening_spikes(body)
    body = _soften_body0_toward_prev(body, left)
    body = match_export_opening_grade(body, left)
    return _micro_seam_bridge(left, body)


def concat_chunks_lazy(
    node_id: int,
    plan: DirectorPlan,
    export_segments: list,
    overrides: dict[int, torch.Tensor] | None = None,
    *,
    fill: float = 0.5,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> torch.Tensor:
    """Streaming merge: allocate the result once, then copy each segment in.

    Two costs the previous incremental ``torch.cat`` version paid on every join:

    * **O(N^2) copying** — each ``cat`` reallocated and re-copied everything
      merged so far, so a 10-segment merge moved ~5x the final video.
    * **O(N) transient clones** — the continuity seam pipeline cloned the whole
      accumulated result (``_unfreeze_held_tail`` / ``_micro_seam_bridge``),
      putting 2-3 full copies of the merge in RAM at once. That is the
      ``not enough memory`` the batch merge hit.

    Now: one allocation for the result, one copy-in per segment, and the seam
    fix runs on two ``_seam_window()``-sized windows (see ``_seam_fix_windows``).
    Peak memory is the result plus one segment.

    ``overrides`` carries in-memory chunks keyed by timeline index — both the
    source-passthrough fills (which have no disk cache) and, in batch mode, the
    segments this run just decoded, so they are reused instead of being
    re-read from disk. Everything else is read from disk, trying an exact
    fingerprint match first and falling back to a stale render — the same pick
    order the caller used to build ``export_segments``. Without that fallback a
    「选择运行」+「全部导出」merge crashes: the caller selects an unselected
    segment via ``allow_stale=True`` (or passthrough), but a strict-only read
    here returns None for the very segment that was just accepted.
    """
    from .segment_cache import (
        load_segment_cache as _load_seg,
        probe_segment_cache_shape as _probe_seg,
    )

    def _load_seg_fp(node_id, seg, plan, *, allow_stale=False, workflow_name=None):
        return _load_seg(
            node_id, seg, plan,
            allow_stale=allow_stale, return_fp=True,
            workflow_name=workflow_name, variant=variant,
        )

    if not export_segments:
        raise ValueError("concat_chunks_lazy: no export_segments")
    overrides = dict(overrides or {})
    continuity = getattr(plan, "continuity_enabled", False)
    if workflow_name is None:
        workflow_name = getattr(plan, "workflow_name", None)
    window = _seam_window() if continuity else 0

    def _miss(seg, label: int) -> RuntimeError:
        return RuntimeError(
            f"concat_chunks_lazy: segment {label} cache miss "
            f"(node {node_id}; timeline #{int(seg.index) + 1})"
        )

    def _read(seg, label: int) -> torch.Tensor:
        # pop so the override reference is released once merged.
        chunk = overrides.pop(int(seg.index), None)
        fp = None
        if chunk is None:
            chunk, fp = _load_seg_fp(node_id, seg, plan, workflow_name=workflow_name)
        if chunk is None:
            chunk, fp = _load_seg_fp(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name)
        if chunk is None:
            raise _miss(seg, label)
        chunk = chunk.float()
        # Segment clip caches persist the *full* VAE decode (replayed head
        # included), so a standalone download is the exact requested length. The
        # merge must reproduce the trimmed clip: drop the replayed head again,
        # turning the full clip back into the replay-free segment the seam
        # pipeline expects. Without this every non-first segment would carry its
        # replay into the join and the merged video would grow / stutter at each
        # seam, exactly the regression we move to the segment download instead.
        if fp is not None:
            trim = int(fp.get("trim_frames") or 0)
            exp = int(fp.get("export_frames") or 0)
            n = int(chunk.shape[0])
            # The persisted clip is already the trimmed body (continuity prefix
            # dropped at decode time). Only trim when the clip on disk still
            # carries the prefix — i.e. it is longer than ``export_frames`` (older
            # caches, or a raw sample). Re-trimming an already-trimmed body would
            # drop another 22 frames and shorten the merged video.
            if trim > 0 and exp > 0 and n > exp:
                chunk = chunk[trim:].contiguous()
        return chunk

    # ---- Pass 1: geometry only (no pixels) ---------------------------------
    # Probing reads the file header rather than the tensor, so measuring the
    # merge costs almost nothing and every clip is loaded exactly once below.
    shapes: list[tuple[int, int, int, int]] = []
    for seg in export_segments:
        in_mem = overrides.get(int(seg.index))
        if in_mem is not None:
            # The in-memory chunk is the *trimmed* export clip (the generator
            # already dropped the continuity prefix), matching what the merge wants.
            shapes.append(tuple(int(d) for d in in_mem.shape))
            continue
        shape = _probe_seg(node_id, seg, plan, workflow_name=workflow_name, variant=variant)
        if shape is None:
            shape = _probe_seg(
                node_id, seg, plan, allow_stale=True, workflow_name=workflow_name, variant=variant
            )
        if shape is None:
            # Legacy / unreadable header: fall back to a full read, and keep the
            # pixels in ``overrides`` so pass 2 does not read them twice. Apply the
            # same continuity prefix trim pass 2 would, so the measured length is
            # the merged length (not the full clip length).
            chunk, fp = _load_seg_fp(node_id, seg, plan, workflow_name=workflow_name)
            if chunk is None:
                chunk, fp = _load_seg_fp(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name)
            if chunk is None:
                raise _miss(seg, len(shapes))
            if fp is not None:
                trim = int(fp.get("trim_frames") or 0)
                exp = int(fp.get("export_frames") or 0)
                n = int(chunk.shape[0])
                # Same rule as _read: the persisted clip is already trimmed, so only
                # drop the prefix when it is still present (n > export_frames).
                if trim > 0 and exp > 0 and n > exp:
                    chunk = chunk[trim:]
            overrides[int(seg.index)] = chunk
            shapes.append(tuple(int(d) for d in chunk.shape))
            del chunk
        else:
            shapes.append(shape)

    total = sum(int(s[0]) for s in shapes)
    if total <= 0:
        raise ValueError("concat_chunks_lazy: export segments contain no frames")
    max_h = max(int(s[1]) for s in shapes)
    max_w = max(int(s[2]) for s in shapes)
    max_c = max(int(s[3]) for s in shapes)

    out = torch.empty((total, max_h, max_w, max_c), dtype=torch.float32)
    if any(int(s[3]) != max_c for s in shapes):
        # Mixed channel counts would leave uninitialised lanes.
        out[..., max(1, min(int(s[3]) for s in shapes)) :] = fill

    # ---- Pass 2: stream each segment into the result -----------------------
    pos = 0
    # Tail of the previous clip at its *native* resolution. Kept so the seam
    # maths sees real pixels: padding to the shared canvas first would fold the
    # letterbox bars into the exposure statistics and skew the correction. It is
    # only ``_seam_window()`` frames, so holding it costs a few MB.
    prev_tail = None
    prev_hw = None

    for i, seg in enumerate(export_segments):
        n = int(shapes[i][0])
        if n <= 0:
            continue
        chunk = _read(seg, i)
        h, w, c = int(chunk.shape[1]), int(chunk.shape[2]), int(chunk.shape[3])

        if window > 0 and prev_tail is not None:
            # ``k_body`` must not be clamped by ``pos``: a short segment still
            # needs its real length here, or helpers that bail out below 3
            # frames would silently skip the correction.
            k_body = min(window, n)
            k_left = int(prev_tail.shape[0])
            # Two small clones instead of cloning the accumulated merge — this
            # join used to cost a full copy of everything merged so far.
            left_tail = prev_tail.clone()
            body_head = chunk[:k_body].clone()
            left_tail, body_head = _seam_fix_windows(left_tail, body_head)
            if prev_hw != (max_h, max_w):
                left_tail = pad_frames_to_canvas(left_tail, max_w, max_h, fill=fill)
            out[pos - k_left : pos] = left_tail
            chunk[:k_body] = body_head
            del left_tail, body_head

        if window > 0:
            k_keep = min(window, n)
            prev_tail = chunk[n - k_keep : n].clone()
            prev_hw = (h, w)

        if (h, w) != (max_h, max_w):
            chunk = pad_frames_to_canvas(chunk, max_w, max_h, fill=fill)
        out[pos : pos + n, :, :, :c] = chunk
        pos += n
        del chunk

    return out


