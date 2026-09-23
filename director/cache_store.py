"""Writing a segment's cache: decode artefacts, handoff meta, AV latents.

Extracted from :mod:`segment_cache`. This is the write side of the per-segment
cache:

* ``save_segment_cache`` — latent / frames / head-tail / audio / clip / meta /
  handoff for one segment;
* ``save_second_pass_cache`` — the「二采」variant, written to its own ``seg2_*``
  file group so a second sample never overwrites the first-pass render;
* the small readers a run needs *while it is still in flight* — the handoff meta
  and the previous segment's AV latent that the next segment conditions on.

Layering: depends on :mod:`cache_paths` (where files live, whether they are still
current), :mod:`cache_codecs` (frame conversion) and :mod:`cache_files` (clip
artefacts). It does not import the export or slot-sync machinery.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from ..lib.fs import safe_unlink as _safe_unlink, write_via_temp as _write_via_temp
from . import cache_layout
from . import segment_slots
from .cache_codecs import (
    HEADTAIL_N,
    _audio_payload_to_cpu,
    _frames_to_headtail,
    _save_headtail,
)
from .cache_files import save_segment_clip
from .cache_paths import (
    _cache_root,
    _fingerprint_compatible,
    _reject_source_stale,
    _slot_paths,
    segment_cache_fingerprint,
)
from .plan_types import DirectorPlan, SegmentPlan

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache.store")


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
    variant: str = segment_slots.VARIANT_FIRST,
) -> None:
    """Persist a segment tensor (+ optional AV latent / export audio). Never raises.

    A segment's video lives in ``clip.mp4``; this call writes it **together with**
    the head/tail window (``frames_ht``), because :func:`load_segment_cache`
    stitches the two back into one clip. Updating only one of them is what mixed
    two different renders into a single cached segment.

    ``variant`` redirects the whole write into the other pass's file group
    (``"2nd"`` → ``seg2_*``). The second pass reuses this function *verbatim*
    rather than keeping its own partial writer: that is what guarantees a
    ``seg2_`` group holds exactly the artefacts the「分段导出」merge expects
    (clip + head/tail seam window + latent + audio + handoff + meta) and that
    the two passes can never drift apart in layout.

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
    stem = segment_slots.resolve_stem(root, idx, variant=variant)
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
        # dominant space cost (hundreds of MB/segment). Merge/export rebuild the
        # whole segment from ``clip.mp4`` (written below, in this same call) and
        # restore the seam frames from this window.
        if not _save_headtail(
            paths,
            tensor,
            fps=float(getattr(plan, "frame_rate", 24) or 24),
        ):
            # No encoder available / unsupported pix_fmt: fall back to the
            # lossless tensor rather than leaving the segment with no seam
            # window at all. Rare, and worth the disk in that case.
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
        # The segment's video, written from the *same* tensor as the head/tail
        # window above. load_segment_cache splices the two, so they must always
        # be refreshed together — a partial update (phase-align re-save, a failed
        # encode) used to leave a clip body from one render stitched to the
        # head/tail of another. Runs last because a failed encode falls back to
        # persisting the full frames.pt, which must not be unlinked afterwards.
        # Best-effort: never raises.
        save_segment_clip(
            node_id, seg, plan, tensor, audio=audio,
            workflow_name=workflow_name, variant=variant,
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


def load_segment_handoff_meta(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> dict[str, Any] | None:
    """Load trim/export handoff metadata (fingerprint must match unless ``allow_stale``)."""
    if not node_id:
        return None
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    idx = seg.index
    paths = _slot_paths(node_id, workflow_name, idx, variant=variant)
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
    variant: str = segment_slots.VARIANT_FIRST,
) -> dict | None:
    """Load cached AV latent for continuity handoff (fingerprint must match unless stale-ok).

    ``variant="2nd"`` reads the second-pass latent (``seg2_*_latent.pt``).
    """
    if not node_id:
        return None
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    idx = seg.index
    paths = _slot_paths(node_id, workflow_name, idx, variant=variant)
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


def resolve_second_stem(
    node_id: str | None,
    seg_index: int,
    workflow_name: str | None = None,
) -> str | None:
    """``seg2_*`` file stem owning timeline ``seg_index`` (second-pass map).

    ``None`` when the second-pass map has not been synced for this position yet
    (call :func:`sync_second_segment_slots` first, exactly as first-pass reads
    rely on the slot map being current).
    """
    if not node_id:
        return None
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return None
    return segment_slots.resolve_stem(
        root, int(seg_index), variant=segment_slots.VARIANT_SECOND
    )


def save_second_pass_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    tensor: torch.Tensor | None,
    *,
    av_latent: dict | None = None,
    audio: dict[str, Any] | None = None,
    handoff: dict[str, Any] | None = None,
    meta_extra: dict[str, Any] | None = None,
    workflow_name: str | None = None,
) -> bool:
    """Write a second-pass artefact set under the slot's ``seg2_*`` group.

    Deliberately a thin wrapper over :func:`save_segment_cache`: the second pass
    must land *exactly* the artefacts the「分段导出」merge reads for the first
    pass — ``clip.mp4`` (the segment's video), ``frames_ht`` (the head/tail
    seam window), ``latent.pt``, ``audio.pt`` and a fingerprint-compatible
    ``meta.json`` — or a later re-export of a second-pass result would silently
    degrade to「仅有 latent」or be refused as stale. Writing them by hand twice
    is how the two passes drifted apart before (no clip, no handoff).

    ``handoff`` carries the trim the second pass applied (``trim_frames`` /
    ``export_frames`` / ``sample_frames``) so a later merge reproduces the same
    boundary instead of re-deriving it. ``meta_extra`` is second-pass-only
    provenance stored in ``handoff.json`` — it is NOT part of ``meta.json``,
    because that file is compared byte-for-byte against the segment fingerprint
    and any extra key there would mark every second-pass cache stale.

    Returns ``False`` (with a warning) when the second-pass slot map has no
    stem for this position — i.e. :func:`sync_second_segment_slots` did not run
    or failed. Callers must surface that, not swallow it.
    """
    if not node_id:
        return False
    if tensor is None or not isinstance(tensor, torch.Tensor) or int(tensor.shape[0]) <= 0:
        log.warning("二采: 段 #%d 无可用帧，跳过缓存写入。", int(seg.index) + 1)
        return False
    stem = resolve_second_stem(node_id, int(seg.index), workflow_name=workflow_name)
    if not stem:
        log.warning(
            "二采: 段 #%d 的 seg2 槽位缺失（二采 slot map 未同步），缓存写入失败。",
            int(seg.index) + 1,
        )
        return False
    payload: dict[str, Any] = {"pass": "2nd"}
    if handoff:
        payload.update(handoff)
    if meta_extra:
        payload["second_pass"] = dict(meta_extra)
    try:
        save_segment_cache(
            node_id,
            seg,
            plan,
            tensor,
            av_latent=av_latent,
            handoff=payload,
            audio=audio,
            workflow_name=workflow_name,
            variant=segment_slots.VARIANT_SECOND,
        )
    except Exception as exc:  # pragma: no cover - save_segment_cache never raises
        log.warning("Second-pass cache write #%d skipped (%s).", int(seg.index) + 1, exc)
        return False
    return True
