"""Concatenating per-segment chunks into the timeline (continuity facade).

Active path (opt-in「段间引导」): motion-context pin via
``director.h3_motion_context`` — previous segment AV tail → next segment
conditioning, then trim the pinned prefix. Tasks: t2v / i2v / fl2v / r2v / v2v /
rv2v.

What lives here now is the concat step itself: joining cached chunk tensors into
the video the node returns, plus the seam windows that decide where frames may be
rewritten. The rest was split by concern:

* :mod:`continuity_settings` — read the continuity flags / overlap / redraw from
  the timeline and locate the previous segment;
* :mod:`continuity_seam`     — grade the seam, repair hold / pop / spike;
* :mod:`continuity_knobs`    — every ``CONTINUITY_*`` tuning value.

The Wan/SCAIL-era surface — luma matching, lock/unlock feathering, free-latent
warm-start, the additive-luma / opening-Y-map blends — was never read by the
MiniMax H3 pipeline and has been removed, together with the
``apply_scail_continuity_core`` no-op that outlived it.

The settings helpers are re-exported (see ``__all__``) because the node, the HTTP
routes and the batch executor still import them from this module; new code should
import them from :mod:`continuity_settings`.
"""

from __future__ import annotations

import logging

import torch

from ..lib.image_prep import cat_frames_variable_size, pad_frames_to_canvas
from . import segment_slots
from .continuity_knobs import (
    CONTINUITY_HOLD_POP_ON_TAIL,
    CONTINUITY_HOLD_POP_SCAN,
    CONTINUITY_OPENING_EXPOSURE_SMOOTH,
    CONTINUITY_OPENING_LUMA_BLEND,
    CONTINUITY_SEAM_ADD_LUMA_FRAMES,
    CONTINUITY_SEAM_SOFTEN_FRAMES,
    CONTINUITY_SPIKE_SCAN,
    CONTINUITY_SPIKE_WEIGHT,
    CONTINUITY_TAIL_LUMA_BLEND,
    CONTINUITY_TAIL_SOFTEN_FRAMES,
)
from .continuity_seam import (
    match_export_opening_grade,
    _break_hold_pop_window,
    _ease_opening_spikes,
    _micro_seam_bridge,
    _soften_body0_toward_prev,
    _unfreeze_held_tail,
)
from .continuity_settings import (
    is_continuity_active,
    resolve_continuity_redraw,
    resolve_continuity_settings,
    resolve_prev_segment_output,
    resolve_segment_continuity_from_prev,
    resolve_segment_continuity_to_next,
    timeline_row_for_index,
)
from .plan_types import DirectorPlan, SegmentPlan

__all__ = [
    # Defined here: the concat step.
    "concat_chunks_lazy",
    "concat_continuous_chunks",
    # Re-exported settings helpers, still imported from this module by callers.
    "is_continuity_active",
    "resolve_continuity_redraw",
    "resolve_continuity_settings",
    "resolve_prev_segment_output",
    "resolve_segment_continuity_from_prev",
    "resolve_segment_continuity_to_next",
    "timeline_row_for_index",
]

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.continuity")

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


