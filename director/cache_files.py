"""Per-segment cache file access: clip artefacts, availability probes, shapes.

Extracted from :mod:`segment_cache`. This layer answers the questions the UI and
the export pickers ask *before* a run — "does this position hold a rendered clip,
a second-pass render, a text encoding" — and writes the one artefact the export
path owns (``clip.mp4``).

It sits on top of :mod:`cache_paths` (which decides where a segment's files are
and whether they are still current) and :mod:`cache_codecs` (which converts
frames), and it only imports from those — never from the slot-map or export
machinery that the rest of :mod:`segment_cache` holds.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import torch

from ..lib.fs import (
    atomic_publish as _atomic_publish,
    safe_unlink as _safe_unlink,
    write_via_temp as _write_via_temp,
)
from . import cache_layout
from . import segment_slots
from .cache_codecs import _frames_to_disk
from .cache_paths import (
    _cache_root,
    _fingerprint_compatible,
    _fingerprint_diff_keys,
    _fingerprint_matches,
    _reject_source_stale,
    _slot_paths,
    segment_cache_fingerprint,
    slot_content_hash,
)
from .plan_types import DirectorPlan, SegmentPlan

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache.files")


def clip_cache_path(
    node_id: str | None,
    seg_index: int,
    workflow_name: str | None = None,
    *,
    allow_prev: bool = False,
    variant: str = segment_slots.VARIANT_FIRST,
) -> Path | None:
    """``.../minimax_director_opt_cache/<slug>/node_<id>/seg_<hash>_clip.mp4``.

    Resolved through the slot map, so it follows the segment living at
    ``seg_index`` rather than the index itself. Does not create dirs — callers
    only stat/unlink this.

    ``allow_prev=True``: when the current file group holds no clip, fall back to
    the superseded group (one generation back). A plan edit re-hashes a position
    and hands it a fresh, still-empty stem, so the newest clip on disk is then
    the one under ``prev``. Read-only probes (export status) use this; writers
    must not, or they would overwrite the previous render.

    ``variant``: which pass's clip is addressed (``"2nd"`` → ``seg2_*_clip.mp4``).
    """
    if not node_id:
        return None
    try:
        root = cache_layout.node_cache_dir(str(node_id), workflow_name, create=False)
    except Exception:
        return None
    paths = segment_slots.slot_paths(root, int(seg_index), variant=variant)
    if paths is None:
        return None
    clip = paths["clip"]
    if allow_prev and not clip.is_file():
        previous = segment_slots.slot_paths(root, int(seg_index), stale=True, variant=variant)
        if previous is not None:
            clip = previous["clip"]
    return clip


def has_segment_clip(
    node_id: str | None,
    seg_index: int,
    workflow_name: str | None = None,
    *,
    allow_prev: bool = False,
    variant: str = segment_slots.VARIANT_FIRST,
) -> bool:
    path = clip_cache_path(
        node_id, seg_index, workflow_name=workflow_name,
        allow_prev=allow_prev, variant=variant,
    )
    if path is None:
        return False
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def save_segment_clip(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    frames: torch.Tensor,
    audio: dict[str, Any] | None = None,
    *,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> str | None:
    """Encode ``frames`` into the clip cache. Never raises.

    ``clip.mp4`` *is* the segment's video: everything that serves a segment on its
    own (「分段导出」, the node IMAGE output) reads it, and :func:`load_segment_cache`
    only overlays the seam's head/tail window on top of it. That window is stored
    by :func:`save_segment_cache`, which calls this function from the same tensor —
    the two files must never describe different renders.

    Best-effort like the rest of the cache: a missing ffmpeg or a read-only mount
    must not break generation.
    """
    if not node_id:
        return None
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or int(frames.shape[0]) <= 0:
        return None
    dest = clip_cache_path(node_id, int(seg.index), workflow_name=workflow_name, variant=variant)
    if dest is None:
        return None
    try:
        from ..lib.video_export import write_frames_to_mp4
        from .audio_export import prepare_segment_audio_for_file_export

        wave = prepare_segment_audio_for_file_export(
            plan, seg, audio_dict=audio, frame_count=int(frames.shape[0]),
        )
        tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp.mp4")
        try:
            write_frames_to_mp4(
                tmp,
                # The encoder accepts both domains; a bare ``.float()`` would
                # silently reinterpret uint8 [0,255] as float [0,255].
                frames.detach().cpu(),
                fps=float(getattr(plan, "frame_rate", 24) or 24),
                audio=wave,
            )
            _atomic_publish(tmp, dest)
        finally:
            _safe_unlink(tmp)
        log.debug("Cached segment %d clip for node %s", int(seg.index) + 1, node_id)
        return str(dest)
    except Exception as exc:
        log.warning(
            "Segment %d clip cache write skipped (%s). 分段导出 will re-encode on demand.",
            int(seg.index) + 1,
            exc,
        )
        _safe_unlink(dest)
        # Fallback: the head/tail-only cache cannot rebuild a full segment on its
        # own, so if the mp4 is missing there would be NO pixel cache left and
        # every later export would re-run the VAE. Keep the full tensor instead —
        # rare (ffmpeg missing / encode error) and worth the disk.
        try:
            fallback_paths = _slot_paths(node_id, workflow_name, int(seg.index), variant=variant)
            if fallback_paths is not None:
                payload = _frames_to_disk(frames)
                _write_via_temp(
                    fallback_paths["frames"],
                    lambda p: torch.save(payload, p),
                )
                log.warning(
                    "Segment %d: clip encode failed; kept full frames.pt as fallback cache.",
                    int(seg.index) + 1,
                )
        except Exception as fallback_exc:  # pragma: no cover - defensive
            log.debug("Segment %d frames.pt fallback skipped: %s", int(seg.index) + 1, fallback_exc)
        return None


def segment_export_availability(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> dict[str, Any]:
    """What「分段导出」can use for one segment, without loading pixel data.

    ``variant`` selects the pass whose cache is probed: the picker's「缓存来源」
    toggle drives it, so choosing「二采」reports *only* the ``seg2_*`` state and
    never silently falls back to the first-pass render.
    """
    idx = int(seg.index)
    # ``allow_prev=True`` / ``allow_stale=True``: right after a plan edit the
    # position has been handed a fresh empty stem, so everything exportable is
    # under the superseded group. Falling back keeps the last render available
    # until a new run produces a replacement (same policy as the merge fill).
    has_clip = has_segment_clip(
        node_id, idx, workflow_name=workflow_name, allow_prev=True, variant=variant
    )
    fp_ok = _fingerprint_matches(
        node_id, seg, plan, allow_stale=True, workflow_name=workflow_name, variant=variant
    )
    tensor_path = resolve_segment_cache_path(
        node_id, seg, plan, allow_stale=True, workflow_name=workflow_name, variant=variant
    )
    latent_path = None
    paths = _slot_paths(node_id, workflow_name, idx, variant=variant)
    if paths is not None:
        candidate = paths["latent"]
        if candidate.is_file():
            latent_path = candidate
        else:
            # The latent may live in the superseded file group (one generation
            # back) after a fingerprint churn —「仅有 latent」exports still work.
            # No ``paths["meta"]`` guard here: a fresh stem has no meta *because*
            # nothing has been rendered into it yet, which is exactly when the
            # previous group's latent is the newest one on disk.
            prev = _slot_paths(node_id, workflow_name, idx, stale=True, variant=variant)
            if prev is not None and prev["latent"].is_file():
                latent_path = prev["latent"]
    frames = fp_ok and tensor_path is not None
    latent = latent_path is not None
    return {
        "index": idx,
        "hasClip": has_clip,
        "hasFrames": bool(frames),
        "hasLatent": bool(latent),
        # Exportable when anything on disk can produce frames. A clip alone
        # counts even if the fingerprint drifted: it is still this segment's
        # last render, which is exactly what the user asked to export.
        "exportable": bool(has_clip or frames or latent),
        # ``stale`` means "what you would export is the *previous* render": the
        # current group is empty (a plan edit moved this position to a fresh
        # stem) and every source we found came from the superseded group. The
        # clip counts too — a「分段导出」copying a verbatim clip from ``prev`` is
        # serving old frames just as much as a latent decode would.
        "stale": bool(
            (has_clip or frames or latent)
            and not _fingerprint_matches(
                node_id, seg, plan, workflow_name=workflow_name, variant=variant
            )
        ),
    }


def inspect_segment_export_status(
    node_id: str | None,
    plan: DirectorPlan,
    *,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> dict[str, Any]:
    """Per-segment export availability for the「分段导出」picker."""
    segments = list(getattr(plan, "segments", None) or [])
    rows = [
        segment_export_availability(
            node_id, seg, plan, workflow_name=workflow_name, variant=variant
        )
        for seg in segments
    ]
    return {
        "node_id": str(node_id or ""),
        "segments": rows,
        "exportable_count": sum(1 for row in rows if row["exportable"]),
        "source": str(variant),
    }


def _second_slot_paths(
    node_id: str | None,
    workflow_name: str | None,
    position: int,
    *,
    stale: bool = False,
) -> dict[str, Path] | None:
    """Second-pass artefact paths for timeline ``position`` (``seg2_*``)."""
    return _slot_paths(
        node_id,
        workflow_name,
        position,
        stale=stale,
        variant=segment_slots.VARIANT_SECOND,
    )


def _has_segment_text_conditioning(
    node_id: str | None,
    workflow_name: str | None,
    seg_index: int,
    *,
    slot_key: str | None = None,
) -> bool:
    """Whether *this* segment's own text encoding is on disk.

    The node-level probe above only proves "some segment has an encoding", but
    the runtime loads by the per-segment ``text_key`` the first pass recorded.
    Probing that exact file is what keeps the picker from marking a segment
    second-sampleable that the run would then silently skip (a segment sampled
    before this map existed, or after the text cache was cleared).

    ``slot_key`` is the segment's content hash — the probe has to look the entry
    up the same way the run does, or a reordered timeline makes it report a
    neighbour's encoding as this segment's.
    """
    if not node_id:
        return False
    try:
        from .conditioning_cache import load_segment_second_params

        params = (
            load_segment_second_params(
                node_id, workflow_name, int(seg_index), slot_key=slot_key
            )
            or {}
        )
    except Exception:
        return False
    text_key = str(params.get("text_key") or "").strip()
    if not text_key:
        return False
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return False
    try:
        return (root / f"{cache_layout.TEXT_PREFIX}_{text_key}.pt").is_file()
    except OSError:
        return False


def second_sample_availability(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    workflow_name: str | None = None,
) -> dict[str, Any]:
    """What「二次采样」can use for one segment (mirrors segment_export_availability).

    A segment is second-sampleable when its **first-pass latent** is on disk
    (the upscale + re-sample source) AND **its own** text encoding is cached,
    keyed by the ``text_key`` the first pass recorded for it.
    """
    idx = int(seg.index)
    # First-pass latent — the input of the second sample (with the usual
    # superseded-group fallback after a plan edit).
    latent_path = None
    paths = _slot_paths(node_id, workflow_name, idx)
    if paths is not None:
        candidate = paths["latent"]
        if candidate.is_file():
            latent_path = candidate
        else:
            prev = _slot_paths(node_id, workflow_name, idx, stale=True)
            if prev is not None and prev["latent"].is_file():
                latent_path = prev["latent"]
    has_latent = latent_path is not None

    # Whether a second-pass render already exists for this position.
    has_second = False
    second = _second_slot_paths(node_id, workflow_name, idx)
    if second is not None and second["latent"].is_file():
        has_second = True
    else:
        prev2 = _second_slot_paths(node_id, workflow_name, idx, stale=True)
        if prev2 is not None and prev2["latent"].is_file():
            has_second = True

    # Per-segment, not node-level: the run loads this segment's own ``text_key``.
    has_text = _has_segment_text_conditioning(
        node_id, workflow_name, idx, slot_key=slot_content_hash(seg, plan)
    )
    return {
        "index": idx,
        "hasLatent": bool(has_latent),
        "hasTextCond": bool(has_text),
        "hasSecondLatent": bool(has_second),
        "canSecondSample": bool(has_latent and has_text),
        # 「引用上段」connection state — the front end groups segments by it.
        "continuityFromPrev": bool(getattr(seg, "continuity_from_prev", True)),
    }


def inspect_second_sample_status(
    node_id: str | None,
    plan: DirectorPlan,
    *,
    workflow_name: str | None = None,
) -> dict[str, Any]:
    """Per-segment second-sample availability for the「二次采样」picker."""
    segments = list(getattr(plan, "segments", None) or [])
    rows = [
        second_sample_availability(node_id, seg, plan, workflow_name=workflow_name)
        for seg in segments
    ]
    return {
        "node_id": str(node_id or ""),
        "segments": rows,
        "sampleable_count": sum(1 for row in rows if row["canSecondSample"]),
    }


def resolve_segment_cache_path(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> Path | None:
    """Validate a segment's disk cache and return its tensor path (no pixels read).

    Split out of :func:`load_segment_cache` so callers that only need the frame
    count / canvas can probe the cache without materialising the whole clip —
    the merge pass used to load every segment twice (once to measure it, once to
    concatenate it), which doubled both merge I/O and peak RAM.
    """
    if not node_id:
        return None
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    idx = int(seg.index)
    paths = _slot_paths(node_id, workflow_name, idx, variant=variant)
    if paths is None:
        return None
    meta_path = paths["meta"]
    tensor_path = paths["frames"]
    if not tensor_path.is_file():
        # Mirror :func:`load_segment_cache`: the current group can be empty
        # because a plan edit just handed this position a fresh stem, in which
        # case the newest render lives in the superseded group. Probing has to
        # see it — otherwise「分段导出」reports nothing exportable right after
        # the user re-words a prompt.
        if not allow_stale:
            return None
        previous = _slot_paths(node_id, workflow_name, idx, stale=True, variant=variant)
        if previous is None:
            return None
        meta_path = previous["meta"]
        tensor_path = previous["frames"]
        if not tensor_path.is_file():
            return None
    try:
        expected = segment_cache_fingerprint(seg, plan)
        if meta_path.is_file():
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            if not _fingerprint_compatible(stored, expected):
                if _reject_source_stale(stored, expected, seg_index=idx):
                    return None
                diff = _fingerprint_diff_keys(stored, expected)
                if not allow_stale:
                    log.info(
                        "Segment %d cache stale (diff=%s); re-run this segment to refresh.",
                        idx + 1,
                        diff[:8],
                    )
                    return None
                log.warning(
                    "Segment %d: using stale cache for export fill (diff=%s).",
                    idx + 1,
                    diff[:8],
                )
        elif not allow_stale:
            log.info(
                "Segment %d cache missing meta; re-run this segment to refresh.",
                idx + 1,
            )
            return None
        else:
            log.warning(
                "Segment %d: using cache without meta for export fill.",
                idx + 1,
            )
        return tensor_path
    except Exception as exc:
        log.warning("Failed to load segment %d cache: %s", idx + 1, exc)
        return None


def _probe_clip_shape(
    node_id: str | None,
    seg: SegmentPlan,
    workflow_name: str | None = None,
    *,
    allow_prev: bool = False,
    variant: str = segment_slots.VARIANT_FIRST,
) -> tuple[int, int, int, int] | None:
    """``(F, H, W, C)`` of ``seg_XXXX_clip.mp4`` from container metadata alone.

    Mirrors what :func:`load_segment_cache` would rebuild, so callers measuring
    a merge get the same numbers without decoding a single pixel.

    ``allow_prev=True`` also probes the superseded file group, mirroring
    :func:`clip_cache_path`. Without it a plan edit is indistinguishable from an
    absent render: the edit re-hashes the position onto a fresh, still-empty
    stem, so the owning group holds no clip while every rendered frame sits one
    generation back — and every caller concluded "nothing cached".
    """
    idx = int(seg.index)
    groups = [_slot_paths(node_id, workflow_name, idx, variant=variant)]
    if allow_prev:
        groups.append(_slot_paths(node_id, workflow_name, idx, stale=True, variant=variant))
    # PyAV only — the portable build runs ``python -s``, which hides the
    # user-level site-packages where cv2 would live.
    for paths in groups:
        if paths is None:
            continue
        clip_path = paths["clip"]
        if not clip_path.is_file():
            continue
        try:
            from ..lib.video_io import av_video_meta

            meta = av_video_meta(str(clip_path))
            w = int(meta.get("width") or 0)
            h = int(meta.get("height") or 0)
            f = int(meta.get("frame_count") or 0)
            if f > 0 and w > 0 and h > 0:
                return (f, h, w, 3)
        except Exception as exc:
            log.debug("Segment %d clip shape probe failed: %s", idx + 1, exc)
    return None


def probe_segment_cache_shape(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> tuple[int, int, int, int] | None:
    """``(F, H, W, C)`` of the cached clip, reading only the file header.

    Uses ``mmap`` so the pixels stay on disk until the caller actually copies
    them. Returns ``None`` when the cache is unusable (same policy as
    :func:`load_segment_cache`) or the shape cannot be read cheaply.

    Without a legacy ``frames.pt`` there is no tensor header to read, so the
    shape is probed from ``clip.mp4`` instead — still no pixel decode, which is
    what keeps ``concat_chunks_lazy`` Pass 1 cheap.
    """
    path = resolve_segment_cache_path(
        node_id, seg, plan, allow_stale=allow_stale,
        workflow_name=workflow_name, variant=variant,
    )
    if path is None:
        # Only reached when no usable frames.pt resolved; the durable cache is a
        # clip, and it too may live one generation back.
        return _probe_clip_shape(
            node_id, seg, workflow_name=workflow_name,
            allow_prev=allow_stale, variant=variant,
        )
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not torch.is_tensor(loaded) or loaded.ndim != 4:
            return None
        shape = tuple(int(d) for d in loaded.shape)
        del loaded
        return shape  # type: ignore[return-value]
    except TypeError:
        # torch < 2.1 has no mmap kwarg — fall back below.
        pass
    except Exception as exc:
        log.debug("Segment %d shape probe failed (mmap): %s", seg.index + 1, exc)
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(loaded) or loaded.ndim != 4:
            return None
        shape = tuple(int(d) for d in loaded.shape)
        del loaded
        return shape  # type: ignore[return-value]
    except Exception as exc:
        log.debug("Segment %d shape probe failed: %s", seg.index + 1, exc)
        return None
