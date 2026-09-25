"""Reading a segment's cached decode back: frames, tail and audio.

Extracted from :mod:`segment_cache`. These readers are what the next segment, the
continuity pipeline and the export path call when they need an earlier segment's
pixels or audio. They run the same fingerprint check as the writers, so a stale
artefact is reported as "no cache" instead of being silently spliced in.

Layering: depends on :mod:`cache_paths` (layout + fingerprint),
:mod:`cache_codecs` (frame conversion, head/tail window) and :mod:`cache_store`
(handoff meta). It imports nothing from the slot-sync or export machinery.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from . import segment_slots
from .cache_codecs import (
    HEADTAIL_N,
    _frames_as,
    _frames_from_disk,
    _frames_to_float,
    _has_headtail,
    _load_full_segment_via_clip,
    _load_headtail_paths,
    _splice_headtail,
)
from .cache_paths import _cache_root, _fingerprint_matches, _slot_paths
from .cache_store import load_segment_handoff_meta
from .plan_types import DirectorPlan, SegmentPlan

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache.readback")


def load_segment_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
    return_fp: bool = False,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | tuple[torch.Tensor | None, dict[str, Any] | None]:
    """Load cached segment frames (the FULL segment tensor).

    ``dtype`` selects the pixel domain of the returned tensor.
    ``torch.float32`` (default) is the ComfyUI IMAGE convention and what every
    existing caller gets. ``torch.uint8`` returns pixel-exact [0,255] at a
    quarter of the memory — the right choice for pure transport such as「分段导出」,
    whose frames are copied straight to a node output or re-encoded. Anything
    that blends pixels (the continuity seam pipeline) must stay float32.

    ``allow_stale=True``: used for「选择运行」+「全部导出」fill of unselected
    segments. Prefer the last render on disk over blank/gray source placeholders
    when the fingerprint drifted (pipeline bump, minor plan churn). A different
    source video is never treated as usable stale — callers then passthrough
    the current clip (v2v/rv2v) or skip (gen timelines).

    Space optimisation: the full ``frames.pt`` is no longer persisted. We rebuild
    the whole segment from ``clip.mp4`` (which is always written) and, when a
    ``frames_ht`` window exists, overlay the head/tail lanes with the cached
    pixels so the seam keeps the frames it was built from. Both files are
    written by one :func:`save_segment_cache` call; when their frame counts
    disagree they belong to different renders and the clip is returned untouched
    instead of being stitched into a hybrid. Legacy ``frames.pt`` and
    ``frames_ht.pt`` caches are still honoured for backward compatibility.

    ``return_fp=True``: returns ``(tensor, handoff)`` instead of just the tensor.
    ``handoff`` carries ``trim_frames`` — the replayed head the generator
    dropped. Merge callers (``concat_chunks_lazy``) re-apply it so the on-disk
    full clip is turned back into the replay-free segment the seam pipeline
    expects.
    """
    root = _cache_root(node_id, workflow_name)
    if root is None:
        if return_fp:
            return None, None
        return None
    idx = int(seg.index)
    paths = _slot_paths(node_id, workflow_name, idx, variant=variant)
    if paths is None:
        if return_fp:
            return None, None
        return None

    def _load_from(group: dict[str, Path]) -> torch.Tensor | None:
        # 1) Legacy full tensor (older runs) — still valid, used as-is.
        if group["frames"].is_file():
            try:
                return _frames_as(
                    _frames_from_disk(
                        torch.load(group["frames"], map_location="cpu", weights_only=True)
                    ),
                    dtype,
                )
            except Exception as exc:
                log.warning("Failed to load legacy segment %d frames: %s", idx + 1, exc)
        # 2) Rebuild the segment from the rendered clip, restoring the exact
        #    head/tail. The two files are written together by
        #    ``save_segment_cache``; ``_splice_headtail`` refuses to stitch them
        #    when their frame counts disagree, because that means they belong to
        #    different renders.
        clip_path = group["clip"]
        full = (
            _load_full_segment_via_clip(clip_path, dtype=dtype)
            if clip_path.is_file()
            else None
        )
        if full is not None:
            head, tail, total = _load_headtail_paths(group)
            if head is not None:
                full = _splice_headtail(full, head, tail, total)
            return full
        return None

    found = _load_from(paths)
    if found is not None:
        handoff = (
            load_segment_handoff_meta(
                node_id, seg, plan, workflow_name=workflow_name, variant=variant
            )
            if return_fp
            else None
        )
        if return_fp:
            return found, handoff
        return found
    if allow_stale:
        # The current file group is empty but the superseded one (kept for one
        # generation) still holds this segment's last render — that is exactly
        # the「选择运行」fill case after a fingerprint churn.
        previous = _slot_paths(node_id, workflow_name, idx, stale=True, variant=variant)
        if previous is not None:
            found = _load_from(previous)
            if found is not None:
                log.warning(
                    "Segment %d: no current cache; filled from its previous render.",
                    idx + 1,
                )
                handoff = (
                    load_segment_handoff_meta(
                        node_id, seg, plan, allow_stale=True,
                        workflow_name=workflow_name, variant=variant,
                    )
                    if return_fp
                    else None
                )
                if return_fp:
                    return found, handoff
                return found
    # 3) Nothing usable (clip missing + no legacy frames).
    log.debug("Segment %d cache miss: no clip.mp4 and no legacy frames.pt", idx + 1)
    if return_fp:
        return None, None
    return None


def load_segment_tail(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    n: int,
    *,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> torch.Tensor | None:
    """Load only the last ``n`` frames of a cached segment (for seam prev_tail).

    Reads directly from the seam window (``frames_ht``, head/tail only) without
    decoding the whole clip, so the seam pipeline holds only ``_seam_window()``
    frames in memory instead of an entire segment tensor.
    """
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    idx = int(seg.index)
    paths = _slot_paths(node_id, workflow_name, idx, variant=variant)
    if paths is None:
        return None
    if not _has_headtail(paths):
        # Fall back to the full segment's tail if only legacy frames exist.
        full = load_segment_cache(
            node_id, seg, plan, workflow_name=workflow_name, variant=variant
        )
        if full is None:
            return None
        k = min(int(n), int(full.shape[0]))
        return full[int(full.shape[0]) - k:].clone()
    head, tail, total = _load_headtail_paths(paths)
    if head is None:
        return None
    if tail is not None and int(tail.shape[0]) > 0:
        window = tail
    elif total > 0:
        # Short segment: the payload holds every frame of the render.
        window = head[max(0, int(total) - HEADTAIL_N):]
    else:
        # Legacy payload: ``[head_N, tail_N, ...]``, so split it by shape.
        n_head = int(head.shape[0])
        window = head[max(0, n_head - min(HEADTAIL_N, max(1, n_head // 2))):]
    k = min(int(n), int(window.shape[0]))
    if k <= 0:
        return None
    return _frames_to_float(window[int(window.shape[0]) - k:]).clone()


def load_segment_audio(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> dict[str, Any] | None:
    """Load cached export audio for a segment (same fingerprint policy as video)."""
    if not node_id or not _fingerprint_matches(
        node_id, seg, plan, allow_stale=allow_stale,
        workflow_name=workflow_name, variant=variant,
    ):
        return None
    idx = int(seg.index)
    paths = _slot_paths(node_id, workflow_name, idx, variant=variant)
    audio_path = paths["audio"] if paths else None
    if allow_stale and (audio_path is None or not audio_path.is_file()):
        # The fingerprint may have matched the superseded group, whose audio is
        # the one that belongs to that render.
        previous = _slot_paths(node_id, workflow_name, idx, stale=True, variant=variant)
        if previous is not None and previous["audio"].is_file():
            audio_path = previous["audio"]
    if audio_path is None or not audio_path.is_file():
        return None
    return _read_audio_file(audio_path, f"segment {idx + 1}")


def _read_audio_file(audio_path, label: str) -> dict[str, Any] | None:
    """Read one cached audio payload; ``None`` when it is unusable."""
    try:
        payload = torch.load(audio_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return None
        wave = payload.get("waveform")
        if not isinstance(wave, torch.Tensor) or wave.numel() <= 0:
            return None
        sr = int(payload.get("sample_rate") or 0) or 32000
        return {"waveform": wave.contiguous(), "sample_rate": sr}
    except Exception as exc:
        log.warning("Failed to load %s audio cache: %s", label, exc)
        return None


def load_segment_audio_by_position(
    node_id: str | None,
    position: int,
    *,
    allow_stale: bool = False,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> dict[str, Any] | None:
    """Cached audio of a timeline *position*, ignoring the content fingerprint.

    The「音频有效性校验」button grades the audio that already exists against the
    prompt as it is **now**. Editing that prompt churns the fingerprint — it
    carries ``seg.prompt`` — so going through :func:`load_segment_audio` would
    report a cache miss exactly when the check is most worth running. Resolve by
    position instead.
    """
    if not node_id:
        return None
    for stale in ((False, True) if allow_stale else (False,)):
        paths = _slot_paths(
            node_id, workflow_name, int(position), stale=stale, variant=variant
        )
        if paths is None:
            continue
        audio_path = paths["audio"]
        if audio_path.is_file():
            found = _read_audio_file(audio_path, f"segment {int(position) + 1}")
            if found is not None:
                return found
    return None
