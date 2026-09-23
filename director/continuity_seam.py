"""Seam grading and hold/pop repair for a concatenated timeline.

Extracted from :mod:`segment_continuity`. Two families live here:

* **low-frequency grade** — ``match_export_opening_grade`` plus the
  ``_lowfreq_appearance_pull*`` / ``_box_blur_bhwc`` / ``_blur_hwc`` helpers pull
  the opening frames' low-frequency luma+chroma towards the previous segment's
  tail. This replaced the additive mean-luma nudge, which produced 重影;
* **hold / pop / spike repair** — ``_unfreeze_held_tail``, ``_micro_seam_bridge``,
  ``_break_hold_pop_window``, ``_ease_opening_spikes`` and their helpers detect a
  frozen tail or a jump at the seam and rewrite a bounded window around it.

Every threshold is a knob in :mod:`continuity_knobs`, and most are pinned to 0, so
the active path does little beyond the low-frequency pull. The callers are the
concat functions in :mod:`segment_continuity`.
"""

from __future__ import annotations

import logging
import os

import torch

from ..lib.image_prep import fit_canvas
from .continuity_knobs import (
    CONTINUITY_BODY0_SEAM_WEIGHT,
    CONTINUITY_BODY1_SEAM_WEIGHT,
    CONTINUITY_EXPORT_GRADE_BLUR,
    CONTINUITY_EXPORT_GRADE_FRAMES,
    CONTINUITY_EXPORT_GRADE_WEIGHT,
    CONTINUITY_HOLD_MAD,
    CONTINUITY_HOLD_MAX_FRAMES,
    CONTINUITY_HOLD_POP_BLUR,
    CONTINUITY_HOLD_POP_BRIDGE_WEIGHT,
    CONTINUITY_HOLD_POP_BRIDGE_WEIGHT_MAX,
    CONTINUITY_HOLD_POP_JUMP_MAD,
    CONTINUITY_HOLD_POP_LAND_WEIGHT,
    CONTINUITY_HOLD_POP_LOOKAHEAD,
    CONTINUITY_HOLD_POP_LOWFREQ,
    CONTINUITY_HOLD_POP_SCAN,
    CONTINUITY_MICRO_SEAM_MAD,
    CONTINUITY_MICRO_SEAM_MAD_SPAN,
    CONTINUITY_MICRO_SEAM_SECOND_WEIGHT,
    CONTINUITY_MICRO_SEAM_THIRD_WEIGHT,
    CONTINUITY_MICRO_SEAM_WEIGHT,
    CONTINUITY_MICRO_SEAM_WEIGHT_MAX,
    CONTINUITY_SPIKE_JUMP_MAD,
    CONTINUITY_SPIKE_LAND_WEIGHT,
    CONTINUITY_SPIKE_PREV_MAD,
    CONTINUITY_SPIKE_SCAN,
    CONTINUITY_SPIKE_WEIGHT,
)
from .continuity_settings import _proxy_mad

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.continuity.seam")


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
