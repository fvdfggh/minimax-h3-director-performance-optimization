"""Disk cache for MiniMax H3 Director segment decode outputs (partial re-run + merge).

Cache is best-effort: write failures (cloud RO mounts, same-name overwrite
blocks, full disks) must never abort the main generation run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

import torch

import folder_paths

from . import cache_layout
from . import segment_slots
from .h3_motion_context import CONTINUITY_PIPELINE_ID
from .plan import DirectorPlan, SegmentPlan, resolve_ref_image_size

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache")

#: Re-exported so the rest of the package keeps importing it from here.
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
    """Identity that affects first-pass sampling (no Refine settings)."""
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


def first_pass_cache_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Exact-match key for first-pass AV latent. Refine knobs are excluded."""
    fp = _segment_identity_fingerprint(seg, plan)
    fp.update({
        "kind": "first_pass",
        "seed": int(getattr(plan, "sample_seed", 0) or 0),
        "cfg": round(float(getattr(plan, "sample_cfg", 1.0) or 1.0), 6),
        "steps": int(getattr(plan, "sample_steps", 25) or 25),
        "sampler": str(getattr(plan, "sample_sampler", "") or ""),
        "scheduler": str(getattr(plan, "sample_scheduler", "") or ""),
        "shift_video": round(float(getattr(plan, "sample_shift_video", 12.0) or 12.0), 6),
        "shift_audio": round(float(getattr(plan, "sample_shift_audio", 3.0) or 3.0), 6),
    })
    return fp


def segment_cache_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Stable identity for a segment — cache invalidates when edit params change."""
    fp = _segment_identity_fingerprint(seg, plan)
    from .refine_pack import refine_fingerprint

    fp.update(refine_fingerprint(plan))
    return fp


def slot_content_hash(seg: SegmentPlan, plan: DirectorPlan) -> str:
    """Content hash behind a segment's cache file names.

    Built from the identity fingerprint only, so it is independent of the
    segment's position (that is the slot map's job) and of the Refine knobs
    (so confirm-first-pass ``.pre`` files survive a Refine change).
    """
    return segment_slots.content_hash_of_fingerprint(
        _segment_identity_fingerprint(seg, plan), defaults=_fingerprint_defaults()
    )


def sync_segment_slots(
    node_id: str | None,
    plan: DirectorPlan,
    workflow_name: str | None = None,
) -> None:
    """Reconcile this node's cache files with the current timeline.

    The slot-aware replacement for the old index-based ``prune_segment_cache``:
    a group deleted in the middle of the timeline takes its own files with it,
    while every other group keeps — or re-adopts — the render matching its
    content. Never raises; a failed sync only costs cache reuse.
    """
    if not node_id:
        return
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return
    try:
        segments = list(getattr(plan, "segments", None) or [])
        hashes = [slot_content_hash(seg, plan) for seg in segments]
        adoptable = None
        if not segment_slots.has_manifest(root):
            # First run after upgrading: adopt the positional caches by content
            # so nothing is re-rendered just because the file names changed.
            adoptable = _adoptable_stems(root)
        segment_slots.sync_slots(root, hashes, adoptable=adoptable)
    except Exception as exc:
        log.warning("Segment cache slot sync skipped (%s).", exc)


def remove_segment_slot(
    node_id: str | None,
    index: int,
    workflow_name: str | None = None,
) -> bool:
    """Drop the cache of the group living at timeline ``index``.

    Called when the UI deletes a group: only that group's file group is
    unlinked, every other group keeps the files it already owns, and the slot
    list closes the gap so the following groups keep their own position → files
    mapping (the next :func:`sync_segment_slots` re-adopts them by content).
    Never raises — a failed drop only leaves files behind.
    """
    if not node_id:
        return False
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return False
    try:
        return segment_slots.remove_slot(root, int(index))
    except Exception as exc:
        log.warning("Segment cache drop at position %s skipped (%s).", index, exc)
        return False


def _adoptable_stems(root: Path) -> dict[str, list[str]]:
    """``content hash -> file stems`` for caches written before the slot map.

    Those files are named after a *position*, so the only way to know which
    render one holds is to read the fingerprint it stored.
    """
    out: dict[str, list[str]] = {}
    try:
        for meta_path in sorted(root.glob("*_meta.json")):
            if meta_path.name.endswith("_pre_meta.json"):
                continue
            stem = cache_layout.stem_of_filename(meta_path.name)
            if not stem or segment_slots.stem_content_hash(stem):
                continue
            try:
                stored = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(stored, dict):
                continue
            content_hash = segment_slots.content_hash_of_fingerprint(
                stored, defaults=_fingerprint_defaults()
            )
            out.setdefault(content_hash, []).append(stem)
    except OSError:
        return out
    return out


def _slot_paths(
    node_id: str | None,
    workflow_name: str | None,
    position: int,
    *,
    stale: bool = False,
) -> dict[str, Path] | None:
    """Durable artefact paths for timeline ``position``.

    ``None`` means this position owns no cache (timeline shrank / never ran) —
    deliberately distinct from "paths that do not exist yet".
    """
    if not node_id:
        return None
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    return segment_slots.slot_paths(root, position, stale=stale)


def _first_pass_paths(
    node_id: str | None,
    workflow_name: str | None,
    position: int,
) -> dict[str, Path] | None:
    """confirm-first-pass artefact paths for timeline ``position``."""
    if not node_id:
        return None
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    stem = segment_slots.resolve_stem(root, position)
    if not stem:
        return None
    return cache_layout.first_pass_paths(root, stem)


def _safe_unlink(path: Path) -> bool:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
        return True
    except OSError:
        return False


def _atomic_publish(tmp: Path, dest: Path) -> None:
    """Move ``tmp`` 鈫?``dest``, tolerating clouds that block same-name overwrite."""
    try:
        os.replace(tmp, dest)
        return
    except OSError:
        pass
    # Some cloud mounts reject overwrite of an existing name 鈥?remove then rename.
    _safe_unlink(dest)
    try:
        os.replace(tmp, dest)
        return
    except OSError:
        pass
    try:
        tmp.rename(dest)
        return
    except OSError:
        # Last resort: keep the unique temp as the published file name is blocked.
        # Caller may still fail if even create-new is denied.
        raise


def _write_via_temp(dest: Path, write_fn: Callable[[Path], None]) -> None:
    """Write to a unique temp name in the same folder, then publish to ``dest``."""
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_fn(tmp)
        _atomic_publish(tmp, dest)
    finally:
        _safe_unlink(tmp)


def _audio_payload_to_cpu(audio: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize export AUDIO dict for disk cache (waveform on CPU)."""
    if not isinstance(audio, dict):
        return None
    wave = audio.get("waveform")
    if not isinstance(wave, torch.Tensor) or wave.numel() <= 0:
        return None
    sr = int(audio.get("sample_rate") or 0) or 32000
    return {
        "waveform": wave.detach().cpu().contiguous(),
        "sample_rate": sr,
    }


def _frames_to_disk(tensor: torch.Tensor) -> torch.Tensor:
    """Store pixel frames as uint8 [0,255]. Export is 8-bit anyway; float32 is 4× larger."""
    x = tensor.detach().cpu()
    if x.dtype == torch.uint8:
        return x.contiguous()
    return x.float().clamp(0, 1).mul(255).round().clamp(0, 255).to(torch.uint8).contiguous()


def _frames_from_disk(loaded: Any) -> torch.Tensor | None:
    """Restore uint8 cache to float32 [0,1]; pass through legacy float caches."""
    if not isinstance(loaded, torch.Tensor):
        return None
    if loaded.dtype == torch.uint8:
        return loaded.float().div(255.0)
    return loaded.float()


# Head/tail retained per segment. Mirrors segment_continuity._seam_window() so the
# seam pipeline has real pixels on each side of a join without persisting the
# *whole* segment tensor (which is what made the cache balloon to GBs).
HEADTAIL_N = cache_layout.FRAMES_HT_N


def _frames_to_headtail(tensor: torch.Tensor) -> torch.Tensor:
    """Keep only the first/last ``HEADTAIL_N`` frames as uint8 [0,255].

    Layout: ``[head_N, ..., tail_N, H, W, C]``. When a segment is shorter than
    ``2 * HEADTAIL_N`` we keep every frame (deduped) rather than erroring.
    """
    x = tensor.detach().cpu()
    if x.dtype != torch.uint8:
        x = x.float().clamp(0, 1).mul(255).round().clamp(0, 255).to(torch.uint8)
    x = x.contiguous()
    n = int(x.shape[0])
    if n <= 2 * HEADTAIL_N:
        return x
    head = x[:HEADTAIL_N]
    tail = x[n - HEADTAIL_N:]
    return torch.cat([head, tail], dim=0).contiguous()


def _headtail_to_disk(tensor: torch.Tensor) -> torch.Tensor:
    return _frames_to_headtail(tensor)


def _decode_clip_frames(clip_path: Path) -> torch.Tensor | None:
    """Decode a clip to float32 [0,1] RGB frames with PyAV (no OpenCV).

    The portable build launches ComfyUI with ``python -s``, which hides the
    user-level site-packages where ``cv2`` would live, so all video I/O goes
    through the bundled PyAV. ``None`` on any failure.
    """
    from ..lib.video_io import decode_video_frames

    return decode_video_frames(clip_path)


def _load_full_segment_via_clip(clip_path: Path) -> torch.Tensor | None:
    """Decode a rendered ``clip.mp4`` back into a float32 [0,1] frame tensor.

    Used when the full ``frames.pt`` was dropped to save space; the seam
    pipeline only ever needs the head/tail (see :func:`load_segment_tail`), but
    export/merge still needs the whole segment and ``clip.mp4`` is already on
    disk. Falls back to ``None`` on any decode failure (caller keeps the run
    alive) — including an unreadable container or a missing decoder backend.
    """
    if not clip_path or not Path(clip_path).is_file():
        return None
    try:
        return _decode_clip_frames(Path(clip_path))
    except Exception as exc:
        log.warning("Segment clip decode failed for %s: %s", clip_path.name, exc)
        return None


def save_segment_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    tensor: torch.Tensor,
    *,
    av_latent: dict | None = None,
    handoff: dict[str, Any] | None = None,
    audio: dict[str, Any] | None = None,
    replace_audio: bool = True,
    workflow_name: str | None = None,
) -> None:
    """Persist a segment tensor (+ optional AV latent / export audio). Never raises.

    ``replace_audio``:
      - True (default): write ``audio`` when present, otherwise delete stale audio.pt
        (fresh sample with mute/empty decode).
      - False: write ``audio`` when present, otherwise **keep** existing audio.pt
        (phase-align trim re-save must not wipe a prior audio cache).
    """
    if not node_id:
        return
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return
    fp = segment_cache_fingerprint(seg, plan)
    idx = seg.index
    # Resolve through the slot map: the files follow this segment's content,
    # not its position, so a group removed elsewhere cannot touch them.
    stem = segment_slots.resolve_stem(root, idx)
    if not stem:
        return
    paths = cache_layout.segment_paths(root, stem)
    pt_path = paths["frames"]
    ht_path = paths["frames_ht"]
    meta_path = paths["meta"]
    latent_path = paths["latent"]
    handoff_path = paths["handoff"]
    audio_path = paths["audio"]
    try:
        # Keep only head/tail frames on disk — the full segment tensor is the
        # dominant space cost (hundreds of MB/segment). Merge/export reconstruct
        # the whole segment from ``clip.mp4`` (already written), and the seam
        # pipeline reads at most HEADTAIL_N frames via load_segment_tail().
        ht_payload = _frames_to_headtail(tensor)
        _write_via_temp(ht_path, lambda p: torch.save(ht_payload, p))
        fp = dict(fp, ht_n=HEADTAIL_N)
        # Legacy full ``frames.pt`` is intentionally NOT written anymore. Stale
        # copies from earlier runs are ignored by load_segment_cache below.
        _safe_unlink(pt_path)
        text = json.dumps(fp, ensure_ascii=False, sort_keys=True)
        _write_via_temp(
            meta_path,
            lambda p: p.write_text(text, encoding="utf-8"),
        )
        if av_latent is not None and isinstance(av_latent, dict) and "samples" in av_latent:
            cpu_latent = _av_latent_to_cpu(av_latent)
            _write_via_temp(latent_path, lambda p: torch.save(cpu_latent, p))
        if handoff:
            _write_via_temp(
                handoff_path,
                lambda p: p.write_text(
                    json.dumps(handoff, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                ),
            )
        audio_cpu = _audio_payload_to_cpu(audio)
        if audio_cpu is not None:
            _write_via_temp(audio_path, lambda p: torch.save(audio_cpu, p))
        elif replace_audio:
            # Fresh sample with no waveform — drop stale audio from an older run.
            _safe_unlink(audio_path)
        log.debug(
            "Cached segment %d for node %s (%d frames%s%s)",
            idx + 1,
            node_id,
            int(tensor.shape[0]),
            ", +av_latent" if av_latent is not None else "",
            ", +audio" if audio_cpu is not None else (
                ", keep-audio" if not replace_audio else ""
            ),
        )
    except Exception as exc:
        # Xiangong / similar: RO mount or same-name write → skip cache, keep run alive.
        log.warning(
            "Segment %d cache write skipped (%s). Generation continues without disk cache.",
            idx + 1,
            exc,
        )
        for stray in root.glob(f".{stem}.*"):
            _safe_unlink(stray)


def _fingerprint_defaults() -> dict[str, Any]:
    """Defaults for fingerprint keys added after older caches were written.

    Adding a key to the fingerprint makes every pre-existing cache look stale
    (``stored != expected``), forcing a full re-render for no user-visible
    reason. Keys listed here are filled in from their default when *absent* from
    a stored fingerprint, so old caches stay valid as long as the feature they
    describe was off — which is exactly what "absent" meant at the time.
    """
    return {"continuity_to_next": False}


def _fingerprint_compatible(stored: Any, expected: dict[str, Any]) -> bool:
    """Compare fingerprints, treating keys added later as their default."""
    if not isinstance(stored, dict):
        return stored == expected
    patched = dict(stored)
    for key, default in _fingerprint_defaults().items():
        patched.setdefault(key, default)
    return patched == expected


def _fingerprint_diff_keys(stored: Any, expected: dict[str, Any]) -> list[str]:
    if not isinstance(stored, dict):
        return ["<invalid-meta>"]
    patched = dict(stored)
    for key, default in _fingerprint_defaults().items():
        patched.setdefault(key, default)
    keys = sorted(set(patched) | set(expected))
    return [k for k in keys if patched.get(k) != expected.get(k)]


def load_segment_handoff_meta(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
    workflow_name: str | None = None,
) -> dict[str, Any] | None:
    """Load trim/export handoff metadata (fingerprint must match unless ``allow_stale``)."""
    if not node_id:
        return None
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    idx = seg.index
    paths = _slot_paths(node_id, workflow_name, idx)
    if paths is None:
        return None
    meta_path = paths["meta"]
    handoff_path = paths["handoff"]
    if not handoff_path.is_file():
        return None
    # A handoff can exist without its ``.meta.json`` (latent-only leftovers). With
    # ``allow_stale=True`` we still serve it — there is no fingerprint to compare.
    if meta_path.is_file():
        try:
            expected = segment_cache_fingerprint(seg, plan)
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            if not _fingerprint_compatible(stored, expected):
                if _reject_source_stale(stored, expected, seg_index=idx, quiet=True) or not allow_stale:
                    return None
        except Exception:
            if not allow_stale:
                return None
    elif not allow_stale:
        return None
    try:
        data = json.loads(handoff_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _av_latent_to_cpu(av_latent: dict) -> dict:
    samples = av_latent["samples"]
    if hasattr(samples, "unbind"):
        parts = [p.detach().cpu().contiguous() for p in samples.unbind()]
        try:
            import comfy.nested_tensor

            samples_cpu = comfy.nested_tensor.NestedTensor(tuple(parts))
        except Exception:
            samples_cpu = tuple(parts)
    elif isinstance(samples, (tuple, list)):
        samples_cpu = tuple(p.detach().cpu().contiguous() for p in samples)
    elif torch.is_tensor(samples):
        samples_cpu = samples.detach().cpu().contiguous()
    else:
        samples_cpu = samples
    out = {"samples": samples_cpu}
    for key, value in av_latent.items():
        if key == "samples":
            continue
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().contiguous()
        else:
            out[key] = value
    return out


def load_segment_av_latent(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
    workflow_name: str | None = None,
) -> dict | None:
    """Load cached AV latent for continuity handoff (fingerprint must match unless stale-ok)."""
    if not node_id:
        return None
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    idx = seg.index
    paths = _slot_paths(node_id, workflow_name, idx)
    if paths is None:
        return None
    meta_path = paths["meta"]
    latent_path = paths["latent"]
    if not latent_path.is_file():
        return None
    # A latent can exist without its ``.meta.json`` (e.g. a latent written by an
    # older/batch-only path, or the meta write failed). With ``allow_stale=True``
    # we still serve it — there is no fingerprint to compare against, and the
    #「仅有 latent」export path only needs to decode it. Strict callers
    # (``allow_stale=False``) still require the meta.
    if meta_path.is_file():
        try:
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = segment_cache_fingerprint(seg, plan)
            if not _fingerprint_compatible(stored, expected):
                if _reject_source_stale(stored, expected, seg_index=idx, quiet=True) or not allow_stale:
                    return None
        except Exception as exc:
            if not allow_stale:
                log.warning("Failed to read segment %d AV latent meta: %s", idx + 1, exc)
                return None
    elif not allow_stale:
        return None
    try:
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "samples" not in payload:
            return None
        return payload
    except Exception as exc:
        log.warning("Failed to load segment %d AV latent cache: %s", idx + 1, exc)
        return None


def next_segment_av_latent_path(node_id: str | None, seg_index: int, workflow_name: str | None = None) -> Path | None:
    """Cache path of the AV latent one segment after ``seg_index``."""
    if not node_id:
        return None
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    paths = segment_slots.slot_paths(root, int(seg_index) + 1)
    return paths["latent"] if paths else None


def has_next_segment_av_latent(node_id: str | None, seg_index: int, workflow_name: str | None = None) -> bool:
    """Whether the next segment holds a cached AV latent for「对齐下段」.

    Presence of the file is the whole contract: align-to-next is a cache-driven
    mode, so the UI disables the checkbox exactly when this returns False.
    """
    path = next_segment_av_latent_path(node_id, seg_index, workflow_name)
    return path is not None and path.is_file()


def load_next_segment_av_latent(node_id: str | None, seg_index: int, workflow_name: str | None = None) -> dict | None:
    """Load the next segment's cached AV latent to pin into this segment's tail.

    Unlike :func:`load_segment_av_latent` this reads a *neighbour's* cache and
    deliberately skips fingerprint validation: the next segment's fingerprint
    describes its own render, not ours, so validating it would disable
    align-to-next whenever that neighbour happened to be rendered under any
    other settings. Being a reference for someone else's handoff does not
    require that neighbour to be reproducible by us.
    """
    path = next_segment_av_latent_path(node_id, seg_index, workflow_name)
    if path is None or not path.is_file():
        return None
    idx = int(seg_index) + 1
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        log.warning(
            "Failed to load next segment %d AV latent for align-to-next: %s",
            idx + 1,
            exc,
        )
        return None
    if not isinstance(payload, dict) or "samples" not in payload:
        return None
    return payload


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
) -> bool:
    if not node_id:
        return False
    idx = int(seg.index)
    # Current generation first; with ``allow_stale`` the superseded file group
    # (kept for exactly one generation) is accepted too — that is what keeps a
    # fingerprint churn from blanking an unselected slot on「全部导出」.
    for stale in ((False, True) if allow_stale else (False,)):
        paths = _slot_paths(node_id, workflow_name, idx, stale=stale)
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


# --------------------------------------------------------------------------
# Segment video-clip cache (seg_XXXX.clip.mp4)
#
# An encoded clip kept beside the latent/frame cache so「分段导出」can serve a
# segment straight from disk. Same lifetime as the other ``seg_<hash>.*`` files:
# the slot map's garbage collection drops a removed group's clip, and a fresh
# decode overwrites it in place.
# --------------------------------------------------------------------------

CLIP_CACHE_SUFFIX = "clip.mp4"


def clip_cache_path(
    node_id: str | None, seg_index: int, workflow_name: str | None = None
) -> Path | None:
    """``.../minimax_director_cache/<slug>/node_<id>/seg_<hash>_clip.mp4``.

    Resolved through the slot map, so it follows the segment living at
    ``seg_index`` rather than the index itself. Does not create dirs — callers
    only stat/unlink this.
    """
    if not node_id:
        return None
    try:
        root = cache_layout.node_cache_dir(str(node_id), workflow_name, create=False)
    except Exception:
        return None
    paths = segment_slots.slot_paths(root, int(seg_index))
    return paths["clip"] if paths else None


def has_segment_clip(node_id: str | None, seg_index: int, workflow_name: str | None = None) -> bool:
    path = clip_cache_path(node_id, seg_index, workflow_name=workflow_name)
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
) -> str | None:
    """Encode ``frames`` into the clip cache. Never raises.

    Called right after a successful decode so every rendered segment is
    exportable without re-decoding. Best-effort like the rest of the cache: a
    missing ffmpeg or a read-only mount must not break generation.
    """
    if not node_id:
        return None
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or int(frames.shape[0]) <= 0:
        return None
    dest = clip_cache_path(node_id, int(seg.index), workflow_name=workflow_name)
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
                frames.detach().cpu().float(),
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
            fallback_paths = _slot_paths(node_id, workflow_name, int(seg.index))
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


def clear_segment_clip(node_id: str | None, seg_index: int, workflow_name: str | None = None) -> bool:
    """Drop one segment's clip so a stale file can never be exported."""
    path = clip_cache_path(node_id, seg_index, workflow_name=workflow_name)
    if path is None:
        return False
    return _safe_unlink(path)


def segment_export_availability(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    workflow_name: str | None = None,
) -> dict[str, Any]:
    """What「分段导出」can use for one segment, without loading pixel data."""
    idx = int(seg.index)
    has_clip = has_segment_clip(node_id, idx, workflow_name=workflow_name)
    # ``allow_stale=True``: an export can still serve the last render after
    # harmless fingerprint churn (same policy as the merge fill).
    fp_ok = _fingerprint_matches(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name)
    tensor_path = resolve_segment_cache_path(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name)
    latent_path = None
    paths = _slot_paths(node_id, workflow_name, idx)
    if paths is not None:
        candidate = paths["latent"]
        if candidate.is_file():
            latent_path = candidate
        elif paths["meta"].is_file():
            # The latent may live in the superseded file group (one generation
            # back) after a fingerprint churn —「仅有 latent」exports still work.
            prev = _slot_paths(node_id, workflow_name, idx, stale=True)
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
        "stale": bool((frames or latent) and not _fingerprint_matches(node_id, seg, plan, workflow_name=workflow_name)),
    }


def inspect_segment_export_status(
    node_id: str | None,
    plan: DirectorPlan,
    *,
    workflow_name: str | None = None,
) -> dict[str, Any]:
    """Per-segment export availability for the「分段导出」picker."""
    segments = list(getattr(plan, "segments", None) or [])
    rows = [segment_export_availability(node_id, seg, plan, workflow_name=workflow_name) for seg in segments]
    return {
        "node_id": str(node_id or ""),
        "segments": rows,
        "exportable_count": sum(1 for row in rows if row["exportable"]),
    }


# --------------------------------------------------------------------------
# Segment export (piecewise / continuous)
# --------------------------------------------------------------------------

def _load_segment_export_source(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    vae: Any = None,
    workflow_name: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any] | None] | None:
    """Frames + audio for one segment's export, or None if no frame source exists.

    Sources, in order:
      1. raw frame cache (``seg_XXXX.pt``) — lossless, no extra dependency.
      2. latent cache (``seg_XXXX.av.pt``) decoded with the loaded VAE — the
        「仅有 latent 缓存」case; only works when ``vae`` is supplied.

    The encoded clip cache (``seg_XXXX.clip.mp4``) is deliberately **not** read
    back here — that would need a video decoder. It is only used by the piecewise
    exporter as a copy-on-disk source (see ``_export_piecewise``), so a segment
    with only a clip is still exportable without any ffmpeg/cv2 read.
    """
    # 1) raw frame cache (fastest, lossless)
    frames = load_segment_cache(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name)
    if frames is not None:
        audio = load_segment_audio(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name)
        return frames.float(), (audio if isinstance(audio, dict) else None)

    # 2) latent → decode (requires a VAE; supplied by the caller)
    if vae is not None:
        decoded = _decode_latent_to_frames(node_id, seg, plan, vae, workflow_name=workflow_name)
        if decoded is not None:
            return decoded

    return None


def _decode_latent_to_frames(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    vae: Any,
    *,
    workflow_name: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any] | None] | None:
    """Decode the cached AV latent (the「仅有 latent 缓存」export path).

    ``vae`` is a tuple ``(video_vae, audio_vae)`` as passed from the HTTP layer.
    Requires ComfyUI's decode nodes; returns None when the latent is missing or
    decoding fails so the caller falls through to other sources.
    """
    if not isinstance(vae, (tuple, list)) or len(vae) < 1 or vae[0] is None:
        log.warning(
            "Segment %d latent decode skipped: no video VAE provided.",
            int(seg.index) + 1,
        )
        return None
    latent = load_segment_av_latent(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name)
    if not isinstance(latent, dict) or "samples" not in latent:
        log.warning(
            "Segment %d latent decode skipped: no AV latent cache (.av.pt) present.",
            int(seg.index) + 1,
        )
        return None
    try:
        from comfy_extras.nodes_lt import LTXVSeparateAVLatent
        from nodes import VAEDecode

        # The cached latent was written to CPU (``_av_latent_to_cpu``); the decode
        # nodes expect it on the VAE's device (usually CUDA). Move it there before
        # separating / decoding — a CPU latent makes LTXV separate or VAEDecode
        # fail with a device mismatch, which is why「仅有 latent」exports silently
        # skipped before.
        #
        # ``LTXVSeparateAVLatent.execute`` takes the whole AV-latent dict (it does
        # ``av_latent["samples"]``), so move ``latent["samples"]`` in place and pass
        # the dict — NOT ``latent["samples"]`` directly, which would index into a
        # 5-D NestedTensor and raise "too many indices for tensor of dimension 5".
        latent = _move_latent_to_device(latent, vae[0])
        sep = LTXVSeparateAVLatent.execute(latent)
        if hasattr(sep, "args"):
            sep = sep.args
        video_latent = sep[0]
        images, = VAEDecode().decode(vae[0], video_latent)
        images = images.cpu().float()
        audio = None
        audio_vae = vae[1] if len(vae) > 1 else None
        if audio_vae is not None and len(sep) > 1:
            try:
                from comfy_extras.nodes_audio import VAEDecodeAudio as _AD
            except ImportError:
                from comfy_extras.nodes_lt import VAEDecodeAudio as _AD
            audio_out = _AD.execute(audio_vae, sep[1])
            if hasattr(audio_out, "args"):
                audio_out = audio_out.args
            audio = audio_out[0] if len(audio_out) > 0 else None
        # Trim the decoded frames to this segment's *export* length, exactly like
        # Phase 3 does (``_trim_decoded_to_export``). The latent holds the full
        # motion-context-extended frames (e.g. 158 for a 120-frame/5s segment);
        # without this trim a「连续导出」merge of two segments would grow to
        # 158+158 frames instead of 120+120 — the "two 5s became 13s" symptom.
        images, audio = _trim_decoded_for_export(
            node_id, seg, plan, images, audio, workflow_name=workflow_name,
        )
        # Segmented exports decode many segments in sequence. Drop the GPU decode
        # intermediates (separated video/audio latent, the GPU-moved latent dict)
        # and release cached VRAM so memory does not accumulate across segments.
        try:
            del sep, video_latent, latent
        except Exception:
            pass
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        return images, audio
    except Exception as exc:
        log.warning("Segment %d latent decode for export failed: %s", int(seg.index) + 1, exc)
        return None


def _expected_export_frames(plan, seg, fallback_n=0):
    """Compute the segment's expected ``(trim_frames, export_len)`` exactly like
    Phase 2's ``generation_frame_budget``, straight from the node's parameters
    (``continuity_overlap_frames`` / ``frame_count`` / ``seg.index``).

    Used both to trim decoded latent frames and to validate an existing frame
    cache's length (a leftover untrimmed frame cache has the wrong count and must
    be re-decoded).
    """
    from .frame_align import minimax_align_frame_count, minimax_phase_aligned_export_frames
    from .h3_motion_context import snap_context_frames

    frame_count = int(getattr(seg, "frame_count", 0) or 0)
    visible = minimax_align_frame_count(max(5, frame_count)) if frame_count > 0 else int(fallback_n)
    use_mc = int(getattr(plan, "continuity_overlap_frames", 0) or 0) > 0
    has_prev = int(getattr(seg, "index", 0) or 0) > 0
    ctx = (
        snap_context_frames(int(getattr(plan, "continuity_overlap_frames", 0) or 0))
        if (use_mc and has_prev) else 0
    )
    export_len = minimax_phase_aligned_export_frames(visible) if ctx > 0 else visible
    if export_len <= 0:
        export_len = int(fallback_n)
    return int(ctx), int(export_len)


def _trim_decoded_for_export(node_id, seg, plan, images, audio, workflow_name=None):
    """Trim decoded (latent) frames + audio to the segment's export length.

    Prefers the persisted handoff (``trim_frames``/``export_frames``) which is
    exactly what Phase 3 wrote for this segment; falls back to the node-parameter
    derived boundary (``_expected_export_frames``) when no handoff exists yet.
    """
    from .h3_motion_context import trim_context_prefix

    fps = float(getattr(plan, "frame_rate", 0) or 24)
    handoff = load_segment_handoff_meta(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name) or {}
    try:
        trim_frames = int(handoff.get("trim_frames") or 0)
    except Exception:
        trim_frames = 0
    try:
        export_len = int(handoff.get("export_frames") or 0)
    except Exception:
        export_len = 0
    if trim_frames <= 0 or export_len <= 0:
        # Rebuild the boundary exactly like Phase 2's ``generation_frame_budget``
        # — from the node's own parameters, not by approximating off the decoded
        # length. ``continuity_overlap_frames`` (the motion-context widget) drives
        # the trim; only segments that have a previous segment actually pin.
        _trim_frames, export_len = _expected_export_frames(
            plan, seg, fallback_n=int(images.shape[0]),
        )
        if trim_frames <= 0:
            trim_frames = _trim_frames
    if export_len <= 0:
        export_len = int(images.shape[0])
    if trim_frames > 0:
        images, audio = trim_context_prefix(
            images, audio, trim_frames,
            fps=fps, match_tail=True,
        )
    if int(images.shape[0]) > export_len:
        images = images[:export_len]
        if isinstance(audio, dict) and audio.get("waveform") is not None:
            sr = int(audio.get("sample_rate") or 32000)
            want = int(round((export_len / fps) * sr))
            wf = audio["waveform"]
            if int(wf.shape[-1]) > want:
                audio = {"waveform": wf[..., :want], "sample_rate": sr}
    return images, audio


def _move_latent_to_device(latent: dict, vae) -> dict:
    """Move a cached (CPU) AV-latent dict to the VAE's device.

    ``latent`` is the full ``{"samples": ..., ...}`` dict; only the ``samples``
    tensor (a plain tensor, tuple/list, or ComfyUI NestedTensor) is moved, the
    other keys (noise_seed etc.) are kept. Returns the dict on the VAE's device
    so ``LTXVSeparateAVLatent`` / ``VAEDecode`` can run. Never raises — on
    failure the original dict is returned and the decode surfaces a clearer error.
    """
    samples = latent.get("samples")
    if samples is None:
        return latent
    device = getattr(vae, "device", None)
    if device is None:
        try:
            device = next(vae.parameters()).device
        except Exception:
            device = None
    if device is None:
        try:
            import torch as _t
            device = _t.device("cuda") if _t.cuda.is_available() else _t.device("cpu")
        except Exception:
            device = None
    if device is None:
        return latent
    try:
        if hasattr(samples, "unbind"):
            parts = [p.to(device) for p in samples.unbind()]
            try:
                import comfy.nested_tensor
                moved = comfy.nested_tensor.NestedTensor(tuple(parts))
            except Exception:
                moved = tuple(parts)
        elif isinstance(samples, (tuple, list)):
            moved = tuple(p.to(device) if hasattr(p, "to") else p for p in samples)
        elif hasattr(samples, "to"):
            moved = samples.to(device)
        else:
            moved = samples
        latent = dict(latent)
        latent["samples"] = moved
    except Exception as exc:
        log.warning("Latent device move skipped (%s); decoding may fail.", exc)
    return latent


def _segments_by_index(
    plan: DirectorPlan,
) -> dict[int, SegmentPlan]:
    return {int(s.index): s for s in (getattr(plan, "segments", None) or [])}


def predecode_latent_segments(
    node_id: str | None,
    plan: DirectorPlan,
    indices: list[int],
    vae: Any = None,
    workflow_name: str | None = None,
) -> dict[str, Any]:
    """Unified decode for「分段导出」: decode every checked segment that has only
    a latent (no ``seg_XXXX.pt`` frame cache) **once**, then write the decoded
    frames back to the segment cache.

    This runs *before* merge / node-output / mp4 export so every later consumer
    reads frames from disk instead of re-decoding the latent per consumer. In
    particular the batch MERGE path used to decode the same latent twice (once to
    fill the node IMAGE output, once to write the mp4) — this removes that.

    ``vae`` is the ``(video_vae, audio_vae)`` tuple (see ``_decode_latent_to_frames``).
    Returns ``{"decoded": [...], "failed": [(index, reason), ...]}``. Never raises
    for a per-segment failure.
    """
    decoded: list[int] = []
    failed: list[tuple[int, str]] = []
    if not node_id or not vae:
        return {"decoded": decoded, "failed": failed}
    segments = _segments_by_index(plan)
    for idx in sorted({int(i) for i in indices}):
        seg = segments.get(idx)
        if seg is None:
            continue
        # A correctly-trimmed render is authoritative — reuse it, never re-decode.
        # Only latent-only leftovers (no clip / no legacy frames) need the VAE; a
        # leftover *untrimmed* render (wrong frame count) is re-decoded.
        #
        # The check probes cheap sources only (legacy .pt header, or the mp4
        # container) instead of load_segment_cache: that would decode the whole
        # mp4 just to count its frames, which is the very cost this predecode
        # exists to avoid.
        expected_trim, expected_export = _expected_export_frames(plan, seg)
        paths = _slot_paths(node_id, workflow_name, int(seg.index))
        if paths is not None:
            cached_n = 0
            if paths["frames"].is_file():
                shape = probe_segment_cache_shape(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name)
                cached_n = int(shape[0]) if shape else 0
            elif paths["clip"].is_file():
                shape = _probe_clip_shape(node_id, seg, workflow_name=workflow_name)
                cached_n = int(shape[0]) if shape else 0
            if cached_n > 0:
                if cached_n == expected_export:
                    continue  # authoritative, correctly trimmed
                log.warning(
                    "分段导出 predecode: seg #%d render has %d frames (expected %d) — "
                    "leftover untrimmed frames, re-decoding from latent.",
                    idx + 1, cached_n, expected_export,
                )
        # Only decode when a latent actually exists.
        latent = load_segment_av_latent(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name)
        if latent is None:
            continue
        result = _decode_latent_to_frames(node_id, seg, plan, vae, workflow_name=workflow_name)
        if result is None:
            failed.append((idx, "latent decode failed"))
            continue
        frames, audio = result
        # We got here because the persisted render was *not* the correctly trimmed
        # export (frame count mismatch / no cache). Drop the stale full tensor now:
        # ``load_segment_cache`` prefers ``frames.pt`` over ``clip.mp4``, so leaving
        # it behind makes every later consumer (node output, merge, mp4) read the
        # old untrimmed frames instead of the frames just decoded.
        try:
            if paths is not None and paths["frames"].is_file():
                _safe_unlink(paths["frames"])
                log.info(
                    "分段导出 predecode: seg #%d dropped stale frames.pt "
                    "(%d frames) in favour of the fresh %d-frame decode.",
                    idx + 1, cached_n, int(frames.shape[0]),
                )
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Segment %d stale frames.pt cleanup skipped: %s", idx + 1, exc)
        handoff = load_segment_handoff_meta(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name) or {}
        if not handoff:
            # Rebuild the boundary from the node's own parameters (matches Phase 2)
            # so it persists exactly.
            trim_frames, export_len = _expected_export_frames(
                plan, seg, fallback_n=int(frames.shape[0]),
            )
            handoff = {
                "trim_frames": int(trim_frames),
                "export_frames": int(export_len),
                "sample_frames": int(frames.shape[0]),
                "official_mc_length": False,
            }
        save_segment_cache(
            node_id,
            seg,
            plan,
            frames,
            av_latent=latent,
            handoff=handoff,
            audio=audio if isinstance(audio, dict) else None,
            workflow_name=workflow_name,
        )
        # Also encode the clip cache. save_segment_cache only persists the
        # head/tail window (space), so without this the next export would find no
        # clip.mp4 and pay the full VAE decode again — exactly the recompute this
        # predecode exists to avoid. Best-effort like the rest of the cache.
        try:
            save_segment_clip(
                node_id, seg, plan, frames,
                audio=audio if isinstance(audio, dict) else None,
                workflow_name=workflow_name,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Segment %d predecode clip cache skipped: %s", idx + 1, exc)
        decoded.append(idx)
        log.info(
            "分段导出 predecode: seg #%d latent → head/tail + clip cache (%d frames)",
            idx + 1,
            int(frames.shape[0]),
        )
        # Free per-segment tensors promptly so a long piecewise export does not
        # accumulate CPU/VRAM across many segments.
        try:
            del frames, audio, latent, handoff
        except Exception:
            pass
        try:
            import gc
            gc.collect()
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
    return {"decoded": decoded, "failed": failed}


def continuous_export_runs(indices) -> list[list[int]]:
    """Group sorted segment indices into runs of timeline-adjacent segments.

    ``[1, 2, 3, 7, 8]`` → ``[[1, 2, 3], [7, 8]]``. Used by「连续导出」: a run of
    two or more is stitched into one clip, a run of one is exported standalone.
    """
    runs: list[list[int]] = []
    for idx in sorted({int(i) for i in indices}):
        if runs and idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs


def _audio_has_samples(audio: Any) -> bool:
    """True when an audio dict carries actual samples (mirrors audio_export)."""
    if not isinstance(audio, dict):
        return False
    wave = audio.get("waveform")
    return isinstance(wave, torch.Tensor) and int(wave.numel()) > 0


def merge_run_audio(
    plan: DirectorPlan,
    audios: list,
    frame_counts: list[int],
) -> dict[str, Any] | None:
    """Concatenate the per-segment audio of one「连续导出」run.

    Mirrors the「全部导出」merge (``_merge_generated_segment_audios``) so a stitched
    clip keeps A/V sync: each part is padded/trimmed to its own frame count, then
    concatenated. Returns ``None`` when there is nothing to merge so the caller
    writes a silent clip instead of failing.
    """
    counts = [max(0, int(c)) for c in (frame_counts or [])]
    total = sum(counts)
    if total <= 0 or not audios:
        return None
    if not any(_audio_has_samples(a) for a in audios):
        # Nothing recorded for this run — keep the clip silent (no audio track)
        # instead of encoding a silent AAC stream.
        return None
    try:
        from .audio_export import _merge_generated_segment_audios

        fps = float(getattr(plan, "frame_rate", 24) or 24)
        merged = _merge_generated_segment_audios(
            plan,
            list(audios),
            total_frames=total,
            fps=fps,
            frame_counts=counts,
        )
        return merged if isinstance(merged, dict) else None
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("分段导出 连续导出: audio merge failed (%s)", exc)
        return None


def _run_frame_counts(
    node_id,
    plan: DirectorPlan,
    run_segments: list,
    total_frames: int,
    workflow_name: str | None = None,
) -> list[int]:
    """Per-segment frame counts of one「连续导出」run, read from the cache header.

    The stitched clip length is authoritative (the merge is what actually lands on
    disk), so probed counts are rescaled when they disagree with it — a wrong
    split would desync the merged audio.
    """
    counts: list[int] = []
    for seg in run_segments:
        shape = probe_segment_cache_shape(node_id, seg, plan, workflow_name=workflow_name)
        if shape is None:
            shape = probe_segment_cache_shape(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name)
        counts.append(max(0, int(shape[0])) if shape else 0)
    probed = sum(counts)
    if probed == total_frames:
        return counts
    if probed <= 0:
        n = max(1, len(counts))
        base, rem = divmod(max(0, int(total_frames)), n)
        return [base + (1 if i < rem else 0) for i in range(n)]
    # Proportional rescale onto the real merged length.
    scaled: list[int] = []
    used = 0
    for i, c in enumerate(counts):
        want = int(round(total_frames * (c / probed)))
        if i == len(counts) - 1:
            want = max(0, int(total_frames) - used)
        scaled.append(max(0, want))
        used += scaled[-1]
    return scaled


def run_segment_export(
    node_id: str | None,
    plan: DirectorPlan,
    indices: list[int],
    *,
    mode: str = "piecewise",
    vae: Any = None,
    out_dir: str | None = None,
    workflow_name: str | None = None,
) -> dict[str, Any]:
    """Execute a「分段导出」request.

    ``mode="piecewise"`` (default) writes one mp4 per checked segment.
    ``mode="continuous"`` stitches checked segments that sit next to each other on
    the timeline into a single mp4 (``seg_AA-BB``); a checked segment that has no
    checked neighbour is exported on its own, so a sparse selection produces a mix
    of stitched runs and standalone clips.

    Latent-only segments are predecoded once up front (same as piecewise) and read
    back from the frame cache, so both modes share one decode path.

    Returns a summary dict: ``files``, ``skipped``, ``export_dir``, ``mode``.
    Never raises for a per-segment failure — it is reported in ``skipped`` and the
    rest still exports. When ``out_dir`` is None the files land in
    ``<output>/minimax_segment_export/<node_id>``.
    """
    from .plan import normalize_segment_export_mode

    normalized = normalize_segment_export_mode(mode)
    segments = _segments_by_index(plan)
    valid = sorted({int(i) for i in indices if int(i) in segments})
    if not valid:
        return {"files": [], "skipped": [], "export_dir": "", "mode": normalized}

    # Unified decode first: any checked segment that has only a latent (no frames)
    # is decoded once and written back to the frame cache, so the exports below
    # read frames directly instead of re-decoding per consumer.
    pre = predecode_latent_segments(node_id, plan, valid, vae=vae, workflow_name=workflow_name)
    if pre["decoded"]:
        log.info(
            "分段导出 predecode: unified-decoded %d latent-only segment(s) before export.",
            len(pre["decoded"]),
        )
    for idx, reason in pre["failed"]:
        log.warning("分段导出 predecode seg #%d failed: %s", idx + 1, reason)

    if out_dir:
        export_dir = str(out_dir)
    else:
        base = Path(folder_paths.get_output_directory()) / "minimax_segment_export"
        export_dir = str(base / str(node_id or "node"))
    os.makedirs(export_dir, exist_ok=True)

    files: list[str] = []
    skipped: list[dict[str, Any]] = []

    def _export_one(seg, frames, audio, tag: str) -> None:
        try:
            path = _write_export_mp4(export_dir, seg, frames, audio, plan, tag=tag)
            if path:
                files.append(path)
        except Exception as exc:  # pragma: no cover - defensive
            skipped.append({"index": int(seg.index), "reason": str(exc)})

    def _export_clip_copy(idx: int) -> None:
        """Copy the encoded clip cache verbatim — no re-decode, no re-encode."""
        clip = clip_cache_path(node_id, idx, workflow_name=workflow_name)
        if clip is not None and clip.is_file() and clip.stat().st_size > 0:
            try:
                stamp = uuid.uuid4().hex[:6]
                dest = os.path.join(
                    export_dir, f"segment_export_seg_{idx + 1:02d}_{stamp}.mp4"
                )
                shutil.copy2(str(clip), dest)
                files.append(dest)
            except Exception as exc:  # pragma: no cover - defensive
                skipped.append({"index": idx, "reason": str(exc)})
        else:
            skipped.append({"index": idx, "reason": "no exportable cache"})

    def _export_standalone(idx: int) -> None:
        """Export one segment on its own: frames → latent decode → clip copy."""
        source = _load_segment_export_source(node_id, segments[idx], plan, vae=vae, workflow_name=workflow_name)
        if source is None:
            # No frame source (no .pt, no latent). Fall back to copying the
            # encoded clip cache (seg_XXXX.clip.mp4) verbatim — no re-decode,
            # no re-encode, just a file copy.
            _export_clip_copy(idx)
            return
        _export_one(segments[idx], source[0], source[1], tag=f"seg_{idx + 1:02d}")

    def _segment_audio(idx: int) -> dict[str, Any] | None:
        audio = load_segment_audio(node_id, segments[idx], plan, allow_stale=True, workflow_name=workflow_name)
        return audio if isinstance(audio, dict) else None

    if normalized != "continuous":
        for idx in valid:
            _export_standalone(idx)
        return {
            "files": files,
            "skipped": skipped,
            "export_dir": export_dir,
            "mode": normalized,
        }

    # --- 连续导出: stitch runs of adjacent checked segments ------------------
    # Only a segment with a frame cache can be stitched; the merge streams the
    # clips in from disk (peak ≈ result + one segment, same as「全部导出」) instead
    # of holding a whole run in RAM. Latent-only segments were predecoded above,
    # so they are on disk by now and take part like any other cached segment.
    from .segment_continuity import concat_chunks_lazy

    stitchable = [
        idx
        for idx in valid
        if resolve_segment_cache_path(node_id, segments[idx], plan, allow_stale=True, workflow_name=workflow_name)
        is not None
    ]
    log.info("[DEBUG-EXPORT] valid=%s stitchable=%s runs=%s", valid, stitchable, continuous_export_runs(stitchable))
    for run in continuous_export_runs(stitchable):
        first, last = run[0], run[-1]
        if len(run) == 1:
            # No checked neighbour — export it on its own.
            _export_standalone(first)
            continue
        run_segs = [segments[i] for i in run]
        try:
            merged = concat_chunks_lazy(node_id, plan, run_segs, workflow_name=workflow_name)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "分段导出 连续导出: stitching #%d–#%d failed (%s); "
                "falling back to standalone clips.",
                first + 1, last + 1, exc,
            )
            for i in run:
                _export_standalone(i)
            continue
        counts = _run_frame_counts(node_id, plan, run_segs, int(merged.shape[0]), workflow_name=workflow_name)
        audio = merge_run_audio(plan, [_segment_audio(i) for i in run], counts)
        _export_one(
            segments[first],
            merged,
            audio,
            tag=f"seg_{first + 1:02d}-{last + 1:02d}",
        )
        del merged

    # Checked segments without a frame cache (clip-only, or a latent that could
    # not be decoded here) are exported verbatim / skipped.
    for idx in valid:
        if idx not in stitchable:
            _export_standalone(idx)

    return {
        "files": files,
        "skipped": skipped,
        "export_dir": export_dir,
        "mode": normalized,
    }


def _write_export_mp4(
    export_dir: str,
    seg: SegmentPlan,
    frames: torch.Tensor,
    audio: dict[str, Any] | None,
    plan: DirectorPlan,
    *,
    tag: str,
) -> str | None:
    """Write one export mp4 into ``export_dir`` with a safe, unique name."""
    from ..lib.video_export import write_frames_to_mp4

    fps = float(getattr(plan, "frame_rate", 24) or 24)
    stamp = uuid.uuid4().hex[:6]
    base = f"segment_export_{tag}_{stamp}.mp4"
    path = os.path.join(export_dir, base)
    tmp = path + ".tmp"
    try:
        write_frames_to_mp4(
            tmp,
            frames.detach().cpu().float(),
            fps=fps,
            audio=audio,
        )
        os.replace(tmp, path)
    finally:
        _safe_unlink(Path(tmp))
    return path


def resolve_segment_cache_path(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
    workflow_name: str | None = None,
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
    paths = _slot_paths(node_id, workflow_name, idx)
    if paths is None:
        return None
    meta_path = paths["meta"]
    tensor_path = paths["frames"]
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
) -> tuple[int, int, int, int] | None:
    """``(F, H, W, C)`` of ``seg_XXXX_clip.mp4`` from container metadata alone.

    Mirrors what :func:`load_segment_cache` would rebuild, so callers measuring
    a merge get the same numbers without decoding a single pixel.
    """
    paths = _slot_paths(node_id, workflow_name, int(seg.index))
    if paths is None:
        return None
    clip_path = paths["clip"]
    if not clip_path.is_file():
        return None
    # PyAV only — the portable build runs ``python -s``, which hides the
    # user-level site-packages where cv2 would live.
    try:
        from ..lib.video_io import av_video_meta

        meta = av_video_meta(str(clip_path))
        w = int(meta.get("width") or 0)
        h = int(meta.get("height") or 0)
        f = int(meta.get("frame_count") or 0)
        if f > 0 and w > 0 and h > 0:
            return (f, h, w, 3)
    except Exception as exc:
        log.debug("Segment %d clip shape probe failed: %s", seg.index + 1, exc)
    return None


def probe_segment_cache_shape(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
    workflow_name: str | None = None,
) -> tuple[int, int, int, int] | None:
    """``(F, H, W, C)`` of the cached clip, reading only the file header.

    Uses ``mmap`` so the pixels stay on disk until the caller actually copies
    them. Returns ``None`` when the cache is unusable (same policy as
    :func:`load_segment_cache`) or the shape cannot be read cheaply.

    Without a legacy ``frames.pt`` there is no tensor header to read, so the
    shape is probed from ``clip.mp4`` instead — still no pixel decode, which is
    what keeps ``concat_chunks_lazy`` Pass 1 cheap.
    """
    path = resolve_segment_cache_path(node_id, seg, plan, allow_stale=allow_stale, workflow_name=workflow_name)
    if path is None:
        return _probe_clip_shape(node_id, seg, workflow_name=workflow_name)
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


def load_segment_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
    workflow_name: str | None = None,
) -> torch.Tensor | None:
    """Load cached segment frames (the FULL segment tensor).

    ``allow_stale=True``: used for「选择运行」+「全部导出」fill of unselected
    segments. Prefer the last render on disk over blank/gray source placeholders
    when the fingerprint drifted (pipeline bump, minor plan churn). A different
    source video is never treated as usable stale — callers then passthrough
    the current clip (v2v/rv2v) or skip (gen timelines).

    Space optimisation: the full ``frames.pt`` is no longer persisted. We rebuild
    the whole segment from ``clip.mp4`` (which is always written) and, when a
    ``frames_ht.pt`` exists, overwrite the head/tail lanes with the exact cached
    pixels so seam continuity stays bit-exact. Legacy ``frames.pt`` caches are
    still honoured for backward compatibility.
    """
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    idx = int(seg.index)
    paths = _slot_paths(node_id, workflow_name, idx)
    if paths is None:
        return None

    def _load_from(group: dict[str, Path]) -> torch.Tensor | None:
        # 1) Legacy full tensor (older runs) — still valid, used as-is.
        if group["frames"].is_file():
            try:
                return _frames_from_disk(
                    torch.load(group["frames"], map_location="cpu", weights_only=True)
                )
            except Exception as exc:
                log.warning("Failed to load legacy segment %d frames: %s", idx + 1, exc)
        # 2) Rebuild full segment from the rendered clip; restore exact head/tail.
        clip_path = group["clip"]
        full = _load_full_segment_via_clip(clip_path) if clip_path.is_file() else None
        if full is not None:
            ht_path = group["frames_ht"]
            if ht_path.is_file():
                try:
                    ht = torch.load(ht_path, map_location="cpu", weights_only=True)
                    if torch.is_tensor(ht) and ht.ndim == 4:
                        n = int(ht.shape[0])
                        hn = min(HEADTAIL_N, n // 2, int(full.shape[0]))
                        head = ht[:hn].float().div(255.0) if ht.dtype == torch.uint8 else ht[:hn].float()
                        tail = ht[n - hn:].float().div(255.0) if ht.dtype == torch.uint8 else ht[n - hn:].float()
                        if int(head.shape[0]) <= int(full.shape[0]):
                            full[:hn] = head.to(full.dtype)
                        if int(tail.shape[0]) <= int(full.shape[0]):
                            full[int(full.shape[0]) - hn:] = tail.to(full.dtype)
                except Exception as exc:
                    log.debug("Segment %d head/tail restore skipped: %s", idx + 1, exc)
            return full
        return None

    found = _load_from(paths)
    if found is not None:
        return found
    if allow_stale:
        # The current file group is empty but the superseded one (kept for one
        # generation) still holds this segment's last render — that is exactly
        # the「选择运行」fill case after a fingerprint churn.
        previous = _slot_paths(node_id, workflow_name, idx, stale=True)
        if previous is not None:
            found = _load_from(previous)
            if found is not None:
                log.warning(
                    "Segment %d: no current cache; filled from its previous render.",
                    idx + 1,
                )
                return found
    # 3) Nothing usable (clip missing + no legacy frames).
    log.debug("Segment %d cache miss: no clip.mp4 and no legacy frames.pt", idx + 1)
    return None


def load_segment_tail(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    n: int,
    *,
    workflow_name: str | None = None,
) -> torch.Tensor | None:
    """Load only the last ``n`` frames of a cached segment (for seam prev_tail).

    Reads directly from ``frames_ht.pt`` (head/tail only) without decoding the
    whole clip, so the seam pipeline holds only ``_seam_window()`` frames in
    memory instead of an entire segment tensor.
    """
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    idx = int(seg.index)
    paths = _slot_paths(node_id, workflow_name, idx)
    if paths is None:
        return None
    ht_path = paths["frames_ht"]
    if not ht_path.is_file():
        # Fall back to the full segment's tail if only legacy frames exist.
        full = load_segment_cache(node_id, seg, plan, workflow_name=workflow_name)
        if full is None:
            return None
        k = min(int(n), int(full.shape[0]))
        return full[int(full.shape[0]) - k:].clone()
    try:
        ht = torch.load(ht_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        log.debug("Segment %d head/tail load failed: %s", idx + 1, exc)
        return None
    if not torch.is_tensor(ht) or ht.ndim != 4:
        return None
    total = int(ht.shape[0])
    hn = min(HEADTAIL_N, total // 2)
    tail = ht[total - hn:] if ht.dtype == torch.uint8 else ht[total - hn:]
    tail = tail.float().div(255.0) if ht.dtype == torch.uint8 else tail.float()
    k = min(int(n), int(tail.shape[0]))
    return tail[int(tail.shape[0]) - k:].clone()


def load_segment_audio(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
    workflow_name: str | None = None,
) -> dict[str, Any] | None:
    """Load cached export audio for a segment (same fingerprint policy as video)."""
    if not node_id or not _fingerprint_matches(
        node_id, seg, plan, allow_stale=allow_stale, workflow_name=workflow_name
    ):
        return None
    idx = int(seg.index)
    paths = _slot_paths(node_id, workflow_name, idx)
    audio_path = paths["audio"] if paths else None
    if allow_stale and (audio_path is None or not audio_path.is_file()):
        # The fingerprint may have matched the superseded group, whose audio is
        # the one that belongs to that render.
        previous = _slot_paths(node_id, workflow_name, idx, stale=True)
        if previous is not None and previous["audio"].is_file():
            audio_path = previous["audio"]
    if audio_path is None or not audio_path.is_file():
        return None
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
        log.warning("Failed to load segment %d audio cache: %s", seg.index + 1, exc)
        return None


def save_first_pass_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    av_latent: dict | None = None,
    frames: torch.Tensor | None = None,
    handoff: dict[str, Any] | None = None,
    workflow_name: str | None = None,
) -> None:
    """Persist first-pass AV latent for confirm-then-refine. Never raises."""
    if not node_id:
        return
    if av_latent is None or not isinstance(av_latent, dict) or "samples" not in av_latent:
        return
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return
    fp = first_pass_cache_fingerprint(seg, plan)
    idx = seg.index
    stem = segment_slots.resolve_stem(root, idx)
    if not stem:
        return
    paths = cache_layout.first_pass_paths(root, stem)
    meta_path = paths["meta"]
    latent_path = paths["latent"]
    frames_path = paths["frames"]
    handoff_path = paths["handoff"]
    try:
        cpu_latent = _av_latent_to_cpu(av_latent)
        _write_via_temp(latent_path, lambda p: torch.save(cpu_latent, p))
        text = json.dumps(fp, ensure_ascii=False, sort_keys=True)
        _write_via_temp(meta_path, lambda p: p.write_text(text, encoding="utf-8"))
        if handoff:
            _write_via_temp(
                handoff_path,
                lambda p: p.write_text(
                    json.dumps(handoff, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                ),
            )
        if isinstance(frames, torch.Tensor) and frames.numel() > 0:
            payload = _frames_to_disk(frames)
            _write_via_temp(frames_path, lambda p: torch.save(payload, p))
        log.debug(
            "Cached first-pass segment %d for node %s (seed=%s)",
            idx + 1,
            node_id,
            fp.get("seed"),
        )
    except Exception as exc:
        log.warning(
            "Segment %d first-pass cache write skipped (%s).",
            idx + 1,
            exc,
        )
        for stray in root.glob(f".{stem}.*"):
            _safe_unlink(stray)


def load_first_pass_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    workflow_name: str | None = None,
) -> dict[str, Any] | None:
    """Load first-pass cache only on exact fingerprint match. Never stale."""
    if not node_id:
        return None
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    idx = seg.index
    paths = _first_pass_paths(node_id, workflow_name, idx)
    if paths is None:
        return None
    meta_path = paths["meta"]
    latent_path = paths["latent"]
    frames_path = paths["frames"]
    handoff_path = paths["handoff"]
    if not meta_path.is_file() or not latent_path.is_file():
        return None
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = first_pass_cache_fingerprint(seg, plan)
        if not _fingerprint_compatible(stored, expected):
            if isinstance(stored, dict) and _reject_source_stale(
                stored, expected, seg_index=idx, quiet=True,
            ):
                return None
            diff = _fingerprint_diff_keys(stored, expected) if isinstance(stored, dict) else ["<invalid-meta>"]
            log.info(
                "Segment %d first-pass cache miss (diff=%s); will sample first pass.",
                idx + 1,
                diff[:8],
            )
            return None
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "samples" not in payload:
            return None
        frames = None
        if frames_path.is_file():
            try:
                loaded = torch.load(frames_path, map_location="cpu", weights_only=True)
                if isinstance(loaded, torch.Tensor) and loaded.numel() > 0:
                    frames = _frames_from_disk(loaded)
            except Exception as exc:
                log.debug("Segment %d first-pass frames skipped: %s", idx + 1, exc)
        handoff: dict[str, Any] = {}
        if handoff_path.is_file():
            try:
                data = json.loads(handoff_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    handoff = data
            except Exception:
                handoff = {}
        return {"av_latent": payload, "frames": frames, "handoff": handoff}
    except Exception as exc:
        log.warning("Failed to load segment %d first-pass cache: %s", idx + 1, exc)
        return None


def prune_orphan_segment_files(node_id: str | None, workflow_name: str | None = None) -> int:
    """Delete per-segment files that no slot in the map claims.

    The slot-aware successor to the old index-based ``prune_segment_cache``:
    orphan detection is driven by the slot map instead of by which numeric
    indices still exist, so a file displaced by a mid-timeline deletion is
    removed rather than silently inherited by the segment that slid into its
    place. Never raises.
    """
    if not node_id:
        return 0
    try:
        root = cache_layout.node_cache_dir(str(node_id), workflow_name, create=False)
        if not root.is_dir():
            return 0
        keep: set[str] = set()
        for slot in segment_slots.read_slots(root):
            keep.add(str(slot.get("stem") or ""))
            if slot.get("prev"):
                keep.add(str(slot["prev"]))
        return segment_slots.gc_orphan_files(root, keep)
    except Exception as exc:
        log.debug("Segment cache prune skipped (%s).", exc)
        return 0


def first_pass_cache_disk_signature(
    node_id: str | None, workflow_name: str | None = None
) -> str:
    """Fingerprint confirm-first-pass ``*.pre.*`` files without creating the cache dir.

    Director ``IS_CHANGED`` cannot see the linked Refine pack (ComfyUI only
    forwards widgets). These ``.pre`` files are written only by the confirmation
    hold, so a second Queue observes a new signature and continues into refine.
    """
    if not node_id:
        return ""
    root = cache_layout.node_cache_dir(str(node_id), workflow_name, create=False)
    if not root.is_dir():
        return ""
    parts: list[str] = []
    try:
        for path in sorted(root.glob("seg_*_pre_*")):
            try:
                st = path.stat()
            except OSError:
                continue
            parts.append(f"{path.name}:{int(st.st_mtime_ns)}:{int(st.st_size)}")
    except OSError:
        return ""
    return "|".join(parts)


def inspect_first_pass_cache(
    node_id: str | None,
    plan: DirectorPlan,
    workflow_name: str | None = None,
) -> dict[str, Any]:
    """Inspect first-pass cache files without loading their tensor payloads."""
    current_seed = int(getattr(plan, "sample_seed", 0) or 0)
    result: dict[str, Any] = {
        "exists": False,
        "matches": False,
        "current_seed": current_seed,
        "cached_seeds": [],
        "segment_total": 0,
        "cached_count": 0,
        "matched_count": 0,
        "diff_keys": [],
        "segments": [],
    }
    if not node_id:
        return result

    root = cache_layout.node_cache_dir(str(node_id), workflow_name, create=False)
    all_segments = list(getattr(plan, "segments", None) or [])
    run_indices = getattr(plan, "run_indices", None)
    if run_indices is None:
        selected = all_segments
    else:
        selected = [
            all_segments[i]
            for i in sorted(run_indices)
            if 0 <= i < len(all_segments)
        ]
    result["segment_total"] = len(selected)

    cached_seeds: set[int] = set()
    all_diffs: set[str] = set()
    rows: list[dict[str, Any]] = []
    for seg in selected:
        idx = int(seg.index)
        # ``_pre_*`` names, resolved through the slot map like every other
        # per-segment artefact (these two used an older ``.pre.`` spelling that
        # the writer no longer produced, so the probe never found anything).
        pre_paths = _first_pass_paths(node_id, workflow_name, idx)
        meta_path = pre_paths["meta"] if pre_paths else None
        latent_path = pre_paths["latent"] if pre_paths else None
        meta_exists = bool(meta_path and meta_path.is_file())
        latent_exists = bool(latent_path and latent_path.is_file())
        cache_exists = meta_exists and latent_exists
        stored: Any = None
        read_error = ""
        if meta_exists:
            try:
                stored = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                read_error = str(exc)

        expected = first_pass_cache_fingerprint(seg, plan)
        matches = bool(cache_exists and _fingerprint_compatible(stored, expected))
        diff = (
            _fingerprint_diff_keys(stored, expected)
            if isinstance(stored, dict)
            else (["<invalid-meta>"] if meta_exists else ["<missing-cache>"])
        )
        cached_seed = stored.get("seed") if isinstance(stored, dict) else None
        try:
            if cached_seed is not None:
                cached_seed = int(cached_seed)
                cached_seeds.add(cached_seed)
        except (TypeError, ValueError):
            cached_seed = None
        all_diffs.update(diff)
        rows.append(
            {
                "segment": idx + 1,
                "exists": cache_exists,
                "matches": matches,
                "cached_seed": cached_seed,
                "diff_keys": diff,
                "error": read_error,
            }
        )

    cached_count = sum(1 for row in rows if row["exists"])
    matched_count = sum(1 for row in rows if row["matches"])
    total = len(rows)
    result.update(
        {
            "exists": cached_count > 0,
            "matches": total > 0 and matched_count == total,
            "cached_seeds": sorted(cached_seeds),
            "cached_count": cached_count,
            "matched_count": matched_count,
            "diff_keys": sorted(all_diffs),
            "segments": rows,
        }
    )
    return result
