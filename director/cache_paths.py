"""Cache path layout and segment fingerprinting — the bottom layer of the cache.

Extracted from :mod:`segment_cache`, which had grown past 3000 lines. Everything
here answers two questions and nothing else:

* **where** does a segment's cache live — position → file group, via the slot map
  (:mod:`segment_slots`) and ``cache_layout``'s single root;
* **is it still ours** — the content/sampling fingerprint that decides whether an
  artefact on disk still describes the current segment, or was left behind by an
  edit, an fps change or a pipeline bump.

The slot-map, codec, availability and export layers import from here; nothing
here imports them, and nothing here writes cache files.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from . import cache_layout
from . import segment_slots
from .h3_motion_context import CONTINUITY_PIPELINE_ID
from .plan_types import DirectorPlan, SegmentPlan, resolve_ref_image_size

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache.paths")


SOURCE_VIDEO_FP_KEY = segment_slots.SOURCE_VIDEO_FP_KEY


def source_video_identity(plan: DirectorPlan) -> list[str]:
    """Stable source-clip identity: relative path + size + mtime (overwrite-safe)."""
    from ..lib.video_io import resolve_video_path, video_clips_from_timeline

    clips = video_clips_from_timeline((plan.raw or {}) if plan is not None else {})
    tokens: list[str] = []
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        rel = str(clip.get("videoFile") or clip.get("fileName") or "").strip().replace("\\", "/")
        if not rel:
            continue
        try:
            path = resolve_video_path(clip)
            st = os.stat(path)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
            tokens.append(f"{rel}:{st.st_size}:{mtime_ns}")
        except Exception:
            tokens.append(f"{rel}:missing")
    return tokens


def source_identity_changed(stored: Any, expected: dict[str, Any]) -> bool:
    """True when the current plan has a source video that does not match cache meta.

    Gen timelines (no source clips) never count as a source change, so stale
    fill/continuity still work after pipeline-only fingerprint churn.
    """
    exp = expected.get(SOURCE_VIDEO_FP_KEY) or []
    if not exp:
        return False
    if not isinstance(stored, dict) or SOURCE_VIDEO_FP_KEY not in stored:
        return True
    return stored.get(SOURCE_VIDEO_FP_KEY) != exp


def _reject_source_stale(
    stored: Any,
    expected: dict[str, Any],
    *,
    seg_index: int,
    quiet: bool = False,
) -> bool:
    if not source_identity_changed(stored, expected):
        return False
    if not quiet:
        log.info(
            "Segment %d cache is from a different source video; ignoring stale render.",
            seg_index + 1,
        )
    return True


def _cache_root(node_id: str, workflow_name: str | None = None) -> Path | None:
    """Per-node directory inside the unified Director cache root.

    Segments sit alongside the encoding caches and batch scratch files, all told
    apart by file-name prefix (see :mod:`cache_layout`).
    """
    try:
        return cache_layout.node_cache_dir(str(node_id), workflow_name)
    except OSError as exc:
        log.warning("Segment cache dir unavailable (%s); cache disabled for this run.", exc)
        return None


def _segment_identity_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Identity that affects sampling output (cache invalidation key)."""
    ref_files = sorted(
        f"img{ref.index}:{(getattr(ref, 'image_file', '') or '')}"
        for ref in seg.refs
    )
    ref_audio_files = sorted(
        f"aud{getattr(a, 'index', i)}:{(getattr(a, 'audio_file', '') or '')}"
        for i, a in enumerate(getattr(seg, "ref_audios", None) or [])
    )
    ref_video_files = sorted(
        f"vid{getattr(v, 'index', i)}:{(getattr(v, 'video_file', '') or '')}"
        for i, v in enumerate(getattr(seg, "ref_videos", None) or [])
    )
    ref_video_file = (
        seg.reference_video_meta.get("videoFile")
        or seg.reference_video_meta.get("fileName")
        or ""
    ).strip()
    source_identity = source_video_identity(plan)
    identity: dict[str, Any] = {
        "prompt": seg.prompt,
        "negative": seg.negative_prompt,
        "task_key": seg.task_key,
        "width": plan.width,
        "height": plan.height,
        "frame_rate": float(getattr(plan, "frame_rate", 24) or 24),
        "output_mode": plan.output_mode,
        "ref_max": plan.ref_max_size,
        "ref_image_size": resolve_ref_image_size(seg, plan),
        "refs": ref_files,
        "ref_audios": ref_audio_files,
        "ref_videos": ref_video_files,
        "ref_video": ref_video_file,
        "ref_video_start": seg.reference_video_start_frame,
        SOURCE_VIDEO_FP_KEY: source_identity,
        "continuity": plan.continuity_enabled,
        "continuity_overlap": plan.continuity_overlap_frames if plan.continuity_enabled else 0,
        "continuity_from_prev": bool(getattr(seg, "continuity_from_prev", True)),
        # An align-to-next render pins an extra latent into the tail, so its
        # latent is not interchangeable with a plain continuity render. Without
        # this the two would share a key and reuse each other's cache.
        "continuity_to_next": bool(getattr(seg, "continuity_to_next", False)),
        "continuity_pipeline": CONTINUITY_PIPELINE_ID,
    }
    # Position-free on purpose. ``index``, and (without a source video) the
    # absolute ``start``/``end``, describe *where a segment sits* rather than
    # *what it renders*. Keeping them here made every cache belong to a slot, so
    # deleting a group in the middle silently re-pointed every later render at
    # the deleted group's files. The slot map now owns position → files; this
    # fingerprint only describes content.
    if source_identity:
        # A source timeline: the absolute range selects real source frames.
        identity["start"] = seg.start_frame
        identity["end"] = seg.end_frame
    else:
        # Gen timelines: only the duration changes the render.
        identity["length"] = max(0, int(seg.end_frame) - int(seg.start_frame))
    return identity


def segment_cache_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Stable identity for a segment — cache invalidates when edit params change."""
    return _segment_identity_fingerprint(seg, plan)


def slot_content_hash(seg: SegmentPlan, plan: DirectorPlan) -> str:
    """Content hash behind a segment's cache file names.

    Built from the identity fingerprint only, so it is independent of the
    segment's position (that is the slot map's job).
    """
    return segment_slots.content_hash_of_fingerprint(
        _segment_identity_fingerprint(seg, plan), defaults=_fingerprint_defaults()
    )


def _slot_paths(
    node_id: str | None,
    workflow_name: str | None,
    position: int,
    *,
    stale: bool = False,
    variant: str = segment_slots.VARIANT_FIRST,
) -> dict[str, Path] | None:
    """Durable artefact paths for timeline ``position``.

    ``None`` means this position owns no cache (timeline shrank / never ran) —
    deliberately distinct from "paths that do not exist yet".

    ``variant`` selects which pass's file group is addressed
    (``"1st"`` → ``seg_*`` / ``"2nd"`` → ``seg2_*``). It defaults to the first
    pass so every existing caller keeps reading exactly what it read before;
    only the second-pass writer and the「分段导出」source selector pass ``"2nd"``.
    """
    if not node_id:
        return None
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    return segment_slots.slot_paths(root, position, stale=stale, variant=variant)


def _fingerprint_defaults() -> dict[str, Any]:
    """Defaults for fingerprint keys added after older caches were written.

    Adding a key to the fingerprint makes every pre-existing cache look stale
    (``stored != expected``), forcing a full re-render for no user-visible
    reason. Keys listed here are filled in from their default when *absent* from
    a stored fingerprint, so old caches stay valid as long as the feature they
    describe was off — which is exactly what "absent" meant at the time.
    """
    return {"continuity_to_next": False}


#: Keys that used to be part of the fingerprint but belong to a removed feature.
#: Dropping them from a stored fingerprint keeps those caches valid instead of
#: forcing a full re-render purely because the feature disappeared.
_RETIRED_FINGERPRINT_KEYS = ("refine",)

#: Keys the *writer* stamps into ``meta.json`` for the reader's benefit. They
#: describe how the artefacts were stored, not what was rendered, so they are
#: not part of the content identity and must not take part in the comparison.
#:
#: ``save_segment_cache`` appends ``ht_n`` (the ``frames_ht`` window length) to
#: every meta it writes. ``_segment_identity_fingerprint`` never produces that
#: key, so keeping it in ``stored`` made ``_normalize_stored_fingerprint(stored)
#: == expected`` false for **every** cache on disk — every strict read
#: (``allow_stale=False``) degraded to a miss and「分段导出」flagged each
#: segment「stale」even immediately after a render. ``_fingerprint_diff_keys``
#: reported it as the lone diff (``diff=['ht_n']``).
#:
#: Unlike :data:`_fingerprint_defaults` these keys cannot be fixed by filling in
#: a default: the extra key is *present* in ``stored``, so it has to be dropped.
_STORAGE_ONLY_FINGERPRINT_KEYS = ("ht_n",)


def _normalize_stored_fingerprint(stored: dict) -> dict:
    patched = dict(stored)
    for key, default in _fingerprint_defaults().items():
        patched.setdefault(key, default)
    for key in _RETIRED_FINGERPRINT_KEYS:
        patched.pop(key, None)
    for key in _STORAGE_ONLY_FINGERPRINT_KEYS:
        patched.pop(key, None)
    return patched


def _fingerprint_compatible(stored: Any, expected: dict[str, Any]) -> bool:
    """Compare fingerprints, treating keys added later as their default.

    Storage-only keys (:data:`_STORAGE_ONLY_FINGERPRINT_KEYS`) are dropped from
    ``stored`` first — both sides then describe content identity only, which is
    what the fingerprint is for.
    """
    if not isinstance(stored, dict):
        return stored == expected
    return _normalize_stored_fingerprint(stored) == expected


def _fingerprint_diff_keys(stored: Any, expected: dict[str, Any]) -> list[str]:
    if not isinstance(stored, dict):
        return ["<invalid-meta>"]
    patched = _normalize_stored_fingerprint(stored)
    keys = sorted(set(patched) | set(expected))
    return [k for k in keys if patched.get(k) != expected.get(k)]


def _render_present(paths: dict[str, Path]) -> bool:
    """Whether a usable rendered segment exists in this file group.

    The full ``frames.pt`` is no longer written (space), so presence is
    satisfied by ANY source that can reproduce frames: the legacy full tensor,
    or the encoded ``clip.mp4`` that :func:`load_segment_cache` rebuilds from.
    """
    if paths["frames"].is_file():
        return True
    return paths["clip"].is_file()


def _fingerprint_matches(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> bool:
    if not node_id:
        return False
    idx = int(seg.index)
    # Current generation first; with ``allow_stale`` the superseded file group
    # (kept for exactly one generation) is accepted too — that is what keeps a
    # fingerprint churn from blanking an unselected slot on「全部导出」.
    for stale in ((False, True) if allow_stale else (False,)):
        paths = _slot_paths(node_id, workflow_name, idx, stale=stale, variant=variant)
        if paths is None:
            continue
        meta_path = paths["meta"]
        if not meta_path.is_file():
            continue
        try:
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = segment_cache_fingerprint(seg, plan)
            if _fingerprint_compatible(stored, expected):
                return True
            if _reject_source_stale(stored, expected, seg_index=idx, quiet=True):
                continue
            if allow_stale and _render_present(paths):
                return True
        except Exception:
            continue
    return False
