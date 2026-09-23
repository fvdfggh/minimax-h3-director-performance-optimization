"""Segment export: decode from cache, stitch runs, write the mp4s.

Extracted from :mod:`segment_cache` — the largest group it held. This is the
「分段导出」/「全部导出」pipeline:

* decide whether a cached render is authoritative for a segment
  (:func:`_render_export_authoritative` / :func:`_expected_export_frames`), so an
  untrimmed or foreign render is not exported as if it were current;
* decode latent-only segments **once** and write the frames back to the cache
  (:func:`predecode_latent_segments`) — every consumer below then reads frames;
* group the checked indices into continuous runs
  (:func:`continuous_export_runs`) and build one clip per run, merging each
  segment's audio (:func:`build_run_selection_clips` / :func:`merge_run_audio`);
* drive the whole thing and write the mp4s (:func:`run_segment_export` /
  :func:`_write_export_mp4`).

Layering: sits on top of :mod:`cache_paths`, :mod:`cache_codecs`,
:mod:`cache_files`, :mod:`cache_store` and :mod:`cache_readback`; it imports
nothing from the slot-sync machinery that the rest of :mod:`segment_cache` holds.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch

import folder_paths

from ..lib.fs import safe_unlink as _safe_unlink, write_via_temp as _write_via_temp
from . import segment_slots
from .cache_files import (
    clip_cache_path,
    has_segment_clip,
    probe_segment_cache_shape,
    resolve_segment_cache_path,
    _probe_clip_shape,
)
from .cache_paths import _slot_paths
from .cache_readback import load_segment_audio, load_segment_cache
from .cache_store import (
    load_segment_av_latent,
    load_segment_handoff_meta,
    save_segment_cache,
)
from .plan_types import DirectorPlan, SegmentPlan

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache.export")


def _load_segment_export_source(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    vae: Any = None,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, Any] | None] | None:
    """Frames + audio for one segment's export, or None if no frame source exists.

    Sources, in order:
      1. the rendered video cache (``clip.mp4``, with the legacy ``frames.pt``
         taking precedence when it exists) — no VAE needed.
      2. latent cache (``seg_XXXX.av.pt``) decoded with the loaded VAE — the
         「仅有 latent 缓存」case; only works when ``vae`` is supplied.

    Note ``clip.mp4`` *is* read back here. The old note claiming otherwise was
    stale: :func:`load_segment_cache` rebuilds the segment from the clip through
    PyAV, which is exactly how every other consumer gets frames.

    ``dtype`` selects the pixel domain. Requests uint8 [0,255] from the cache so
    nothing is expanded before its fate is decided; pass float32 (the default)
    when the frames go anywhere that blends them. See :func:`load_segment_cache`.

    Two destinations impose different domains, so the caller must choose:
    re-encoding straight to disk can stay uint8 end-to-end, while a ComfyUI
    IMAGE output is float32 [0,1] by contract and would render as near-white
    noise if uint8 were put on it.
    """
    # 1) rendered frames already on disk (clip.mp4 / legacy frames.pt)
    frames = load_segment_cache(
        node_id, seg, plan, allow_stale=True,
        workflow_name=workflow_name, variant=variant,
        dtype=dtype,
    )
    if frames is not None:
        audio = load_segment_audio(
            node_id, seg, plan, allow_stale=True,
            workflow_name=workflow_name, variant=variant,
        )
        return frames, (audio if isinstance(audio, dict) else None)

    # 2) latent → decode (requires a VAE; supplied by the caller)
    if vae is not None:
        decoded = _decode_latent_to_frames(
            node_id, seg, plan, vae, workflow_name=workflow_name, variant=variant
        )
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
    variant: str = segment_slots.VARIANT_FIRST,
) -> tuple[torch.Tensor, dict[str, Any] | None] | None:
    """Decode the cached AV latent (the「仅有 latent 缓存」export path).

    ``vae`` is a tuple ``(video_vae, audio_vae)`` as passed from the HTTP layer.
    Requires ComfyUI's decode nodes; returns None when the latent is missing or
    decoding fails so the caller falls through to other sources.

    ``variant="2nd"`` decodes the second-pass latent instead.
    """
    if not isinstance(vae, (tuple, list)) or len(vae) < 1 or vae[0] is None:
        log.warning(
            "Segment %d latent decode skipped: no video VAE provided.",
            int(seg.index) + 1,
        )
        return None
    latent = load_segment_av_latent(
        node_id, seg, plan, allow_stale=True,
        workflow_name=workflow_name, variant=variant,
    )
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
        # Trim to the export body: drop the continuity prefix (the replayed tail of
        # the previous segment) and crop to the export length. A standalone segment
        # must start at its own first frame, not with ~1s of the previous clip.
        images, audio = _trim_decoded_for_export(
            node_id, seg, plan, images, audio,
            workflow_name=workflow_name, variant=variant,
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
    Phase 2's ``generation_frame_budget``, straight from the node's parameters.

    A continuity pin replays the previous segment's tail at the head of this
    one, so ``trim_frames`` is the pin length and ``export_len`` is the
    segment's own aligned length — *not* snapped down onto the 17-frame cycle
    grid, which used to cost the last 5 frames of every segment that had a
    predecessor.

    Used both to trim decoded latent frames and to validate an existing frame
    cache's length (a leftover untrimmed frame cache has the wrong count and must
    be re-decoded).
    """
    from .frame_align import minimax_align_frame_count
    from .h3_motion_context import snap_context_frames

    frame_count = int(getattr(seg, "frame_count", 0) or 0)
    visible = minimax_align_frame_count(max(5, frame_count)) if frame_count > 0 else int(fallback_n)
    use_mc = int(getattr(plan, "continuity_overlap_frames", 0) or 0) > 0
    has_prev = int(getattr(seg, "index", 0) or 0) > 0
    next_pin = bool(getattr(seg, "continuity_to_next", False))
    role = (
        "both" if (use_mc and has_prev and next_pin) else
        "prev" if (use_mc and has_prev) else
        "next" if next_pin else
        "none"
    )
    overlap = int(getattr(plan, "continuity_overlap_frames", 0) or 0)
    if visible <= 0:
        visible = int(fallback_n)
    if role == "prev":
        # Referenced segments carry the +5 VAE phase via the head reference, so
        # the export is the clean 17k, not 17k+5.
        return int(snap_context_frames(overlap)), int(visible - 5)
    if role in ("next", "both"):
        # Referenced segments carry the +5 VAE phase via the frozen references, so
        # the export is the clean 17k, not 17k+5. ctx follows head pin availability.
        ctx = snap_context_frames(overlap) if use_mc else 0
        return int(ctx), int(visible - 5)
    return 0, int(visible)


#: How far a container-derived frame count may sit from the expected export
#: length and still count as a match. ``av_video_meta`` falls back to
#: ``round(duration * fps)`` when the stream lacks a frame count, which lands one
#: frame off at non-integer rates; a render that is off by *more* than this is
#: genuinely untrimmed rather than mis-measured.
CLIP_FRAME_COUNT_TOLERANCE = 2


def _render_export_authoritative(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> tuple[bool, int, int, str]:
    """Whether a rendered export can be reused instead of paying for a VAE decode.

    Returns ``(ok, cached_n, expected_export, reason)``.

    ``expected_export`` prefers the ``handoff.json`` written by the run that
    produced the render — that is the same number the seam pin was placed
    against, so it is ground truth. :func:`_expected_export_frames` instead
    re-derives a continuity *role* from the widgets, and the two disagree
    whenever ``context_n`` differed at sample time: an unran predecessor leaves
    it 0, so the real run was ``role="none"`` exporting ``17k+5`` while the
    re-derivation says ``"prev"`` and expects ``17k``. That permanent off-by-5
    made every later export look untrimmed and re-decoded the latent even though
    a perfectly good ``clip.mp4`` was sitting on disk.

    A clip is authoritative on its own: it *is* the last render of this segment,
    so a mismatch only means "repositioned measurement", not "wrong pixels".
    """
    _nominal_trim, expected_export = _expected_export_frames(plan, seg)
    handoff = load_segment_handoff_meta(
        node_id, seg, plan, allow_stale=True,
        workflow_name=workflow_name, variant=variant,
    ) or {}
    hf_export = int(handoff.get("export_frames") or 0)
    if hf_export > 0:
        expected_export = hf_export

    paths = _slot_paths(node_id, workflow_name, int(seg.index), variant=variant)
    if paths is None:
        return False, 0, expected_export, "no cache slot"

    cached_n = 0
    if paths["frames"].is_file():
        shape = probe_segment_cache_shape(
            node_id, seg, plan, allow_stale=True,
            workflow_name=workflow_name, variant=variant,
        )
        cached_n = int(shape[0]) if shape else 0
    else:
        # Clip first in the owning group, then the superseded one: a plan edit
        # hands the position a fresh, still-empty stem, so the only render left
        # is one generation back.
        shape = _probe_clip_shape(
            node_id, seg, workflow_name=workflow_name,
            allow_prev=True, variant=variant,
        )
        cached_n = int(shape[0]) if shape else 0

    if cached_n <= 0:
        return False, 0, expected_export, "no render on disk"
    if expected_export <= 0:
        # Nothing to compare against — trust whatever was rendered.
        return True, cached_n, expected_export, "no expected length"
    if cached_n == expected_export:
        return True, cached_n, expected_export, "exact"
    if abs(cached_n - expected_export) <= CLIP_FRAME_COUNT_TOLERANCE:
        return True, cached_n, expected_export, "within tolerance"
    return False, cached_n, expected_export, "frame count mismatch"


def _trim_decoded_for_export(
    node_id, seg, plan, images, audio, workflow_name=None,
    variant: str = segment_slots.VARIANT_FIRST,
):
    """Trim decoded (latent) frames + audio to the segment's export length.

    Prefers the persisted handoff (``trim_frames``/``export_frames``) which is
    exactly what Phase 3 wrote for this segment; falls back to the node-parameter
    derived boundary (``_expected_export_frames``) when no handoff exists yet.

    The replayed head is always dropped: a pin makes the model reproduce the
    previous segment's tail — picture and sound — so keeping it would play that
    tail a second time before this segment starts. The merged result is
    unaffected — the merge stitches these same trimmed bodies.
    """
    from .h3_motion_context import trim_context_prefix

    fps = float(getattr(plan, "frame_rate", 0) or 24)
    handoff = (
        load_segment_handoff_meta(
            node_id, seg, plan, allow_stale=True,
            workflow_name=workflow_name, variant=variant,
        )
        or {}
    )
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
    variant: str = segment_slots.VARIANT_FIRST,
) -> dict[str, Any]:
    """Unified decode for「分段导出」: decode every checked segment that has only
    a latent (no ``seg_XXXX.pt`` frame cache) **once**, then write the decoded
    frames back to the segment cache.

    ``variant="2nd"`` operates on the second-pass group (``seg2_*``), which is
    what lets a「缓存来源 = 二采」export decode a second-pass latent-only cache
    into the same clip + head/tail pair the first pass would have written.

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
        # A rendered clip is authoritative — reuse it, never re-decode. Only a
        # latent-only leftover (no clip in either file group) reaches the VAE.
        #
        # The check probes cheap sources only (legacy .pt header, or the mp4
        # container) instead of load_segment_cache: that would decode the whole
        # mp4 just to count its frames, which is the very cost this predecode
        # exists to avoid.
        ok, cached_n, expected_export, reason = _render_export_authoritative(
            node_id, seg, plan,
            workflow_name=workflow_name, variant=variant,
        )
        if ok:
            log.info(
                "分段导出 predecode: seg #%d reuse render (%df, %s) — no VAE decode.",
                idx + 1, cached_n, reason,
            )
            continue
        if cached_n > 0:
            log.warning(
                "分段导出 predecode: seg #%d render has %d frames (expected %d) — "
                "leftover untrimmed frames, re-decoding from latent.",
                idx + 1, cached_n, expected_export,
            )
        else:
            log.info(
                "分段导出 predecode: seg #%d has no render (%s); decoding from latent.",
                idx + 1, reason,
            )
        # Only decode when a latent actually exists.
        latent = load_segment_av_latent(
            node_id, seg, plan, allow_stale=True,
            workflow_name=workflow_name, variant=variant,
        )
        if latent is None:
            continue
        result = _decode_latent_to_frames(
            node_id, seg, plan, vae, workflow_name=workflow_name, variant=variant
        )
        if result is None:
            failed.append((idx, "latent decode failed"))
            continue
        frames, audio = result
        paths = _slot_paths(node_id, workflow_name, idx, variant=variant)
        # We got here because the persisted render was *not* a usable export
        # (frame count mismatch / nothing rendered). Drop the stale full tensor:
        # ``load_segment_cache`` prefers ``frames.pt`` over ``clip.mp4``, so leaving
        # it behind makes every later consumer (node output, merge, mp4) read the
        # old frames instead of the ones just decoded.
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
        handoff = (
            load_segment_handoff_meta(
                node_id, seg, plan, allow_stale=True,
                workflow_name=workflow_name, variant=variant,
            )
            or {}
        )
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
            variant=variant,
        )
        # The clip cache is written by save_segment_cache itself (head/tail window
        # and mp4 must always be refreshed together), so the next export finds a
        # clip.mp4 instead of paying the full VAE decode again.
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
    variant: str = segment_slots.VARIANT_FIRST,
) -> list[int]:
    """Per-segment frame counts of one「连续导出」run, read from the cache header.

    The stitched clip length is authoritative (the merge is what actually lands on
    disk), so probed counts are rescaled when they disagree with it — a wrong
    split would desync the merged audio.
    """
    counts: list[int] = []
    for seg in run_segments:
        shape = probe_segment_cache_shape(
            node_id, seg, plan, workflow_name=workflow_name, variant=variant
        )
        if shape is None:
            shape = probe_segment_cache_shape(
                node_id, seg, plan, allow_stale=True,
                workflow_name=workflow_name, variant=variant,
            )
        counts.append(max(0, int(shape[0])) if shape else 0)
    return _rescale_counts(counts, total_frames)


def _rescale_counts(counts: list[int], total_frames: int) -> list[int]:
    """Fit per-segment frame counts onto ``total_frames``.

    The merged clip length is authoritative — it is what actually landed on disk
    — so a probe that disagrees with it is rescaled. A wrong split would desync
    the merged audio.
    """
    counts = [max(0, int(c)) for c in (counts or [])]
    probed = sum(counts)
    total_frames = max(0, int(total_frames))
    if probed == total_frames:
        return counts
    if probed <= 0:
        n = max(1, len(counts))
        base, rem = divmod(total_frames, n)
        return [base + (1 if i < rem else 0) for i in range(n)]
    # Proportional rescale onto the real merged length.
    scaled: list[int] = []
    used = 0
    for i, c in enumerate(counts):
        want = int(round(total_frames * (c / probed)))
        if i == len(counts) - 1:
            want = max(0, total_frames - used)
        scaled.append(max(0, want))
        used += scaled[-1]
    return scaled


def build_run_selection_clips(
    node_id,
    plan: DirectorPlan,
    run_indices,
    chunks: list,
    audios: list | None = None,
    *,
    all_segments: list | None = None,
    mp4_run_dir=None,
    workflow_name: str | None = None,
) -> tuple[list, list, list[int], list[str]]:
    """Collapse a「选择运行」into one clip per contiguous run — the「连续导出」layout.

    Returns ``(clips, audios, frame_counts, mp4_paths)``, all aligned 1:1 and
    ordered by timeline position.

    A partial run used to be merged back onto the full timeline: unselected
    slots were re-read from the segment cache (or filled from the source video)
    so the node could still emit a single「全部导出」clip. That defeated the point
    of the selection — the output was dominated by footage the user did not ask
    to regenerate, and the freshly decoded clips were never visible on their own.

    Now each run of timeline-adjacent selected indices is stitched with
    :func:`concat_chunks_lazy` (same streaming merge, same seam handling as
    「连续导出」) and handed back as its own clip. Runs are never joined across a
    gap, so a selection of ``#3,#4,#5,#9`` yields two clips.

    ``chunks``/``audios`` are the in-memory per-segment results, ordered like
    ``sorted(run_indices)``; they are passed to the merge as overrides so nothing
    is re-read from disk. Missing entries are skipped rather than raising — a
    segment that failed to decode must not lose the whole run.
    """
    from .segment_continuity import concat_chunks_lazy
    from .segment_mp4_export import export_run_mp4

    run_list = sorted({int(i) for i in (run_indices or [])})
    chunk_by_index = {
        idx: chunks[pos] for pos, idx in enumerate(run_list) if pos < len(chunks or [])
    }
    audio_by_index: dict[int, Any] = {}
    if audios:
        audio_by_index = {
            idx: audios[pos] for pos, idx in enumerate(run_list) if pos < len(audios)
        }
    seg_by_index = {
        int(getattr(s, "index", -1)): s for s in (all_segments or [])
    }

    clips: list = []
    run_audios: list = []
    counts: list[int] = []
    mp4_paths: list[str] = []

    for run in continuous_export_runs(run_list):
        present = [i for i in run if i in chunk_by_index]
        if not present:
            continue
        first, last = run[0], run[-1]
        overrides = {i: chunk_by_index[i] for i in present}
        if len(present) == 1:
            # Lone segment — no neighbour to stitch to, emit as-is (complete 5s).
            clip = overrides[present[0]]
            audio = audio_by_index.get(present[0]) or {}
        else:
            # Multi-segment run: chunks are already the trimmed bodies (continuity
            # prefix dropped at decode time), so they stitch as-is. Concat only
            # trims a disk-read clip that still carries the prefix.
            run_segs = [seg_by_index[i] for i in present if i in seg_by_index]
            if len(run_segs) == len(present):
                try:
                    clip = concat_chunks_lazy(
                        node_id, plan, run_segs, overrides=overrides,
                        workflow_name=workflow_name,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning(
                        "选择运行: stitching #%d–#%d failed (%s); "
                        "falling back to a plain concatenation.",
                        first + 1, last + 1, exc,
                    )
                    clip = torch.cat([overrides[i] for i in present], dim=0)
            else:
                clip = torch.cat([overrides[i] for i in present], dim=0)
            part_counts = [int(overrides[i].shape[0]) for i in present]
            audio = merge_run_audio(
                plan,
                [audio_by_index.get(i) or {} for i in present],
                _rescale_counts(part_counts, int(clip.shape[0])),
            )
        clips.append(clip)
        run_audios.append(audio if isinstance(audio, dict) else {})
        counts.append(int(clip.shape[0]))
        if mp4_run_dir is not None:
            path = export_run_mp4(
                mp4_run_dir, plan,
                seg_by_index.get(first) or seg_by_index.get(present[0]),
                seg_by_index.get(last) or seg_by_index.get(present[-1]),
                clip,
                run_audios[-1] or None,
            )
            if path:
                mp4_paths.append(path)

    return clips, run_audios, counts, mp4_paths


def _segment_can_stitch(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> bool:
    """Whether「连续导出」can merge this segment, i.e. whether a source exists.

    ``concat_chunks_lazy`` reads every segment through :func:`load_segment_cache`,
    which rebuilds the whole clip from ``clip.mp4`` and restores the seam frames
    from ``frames_ht`` — the legacy full ``frames.pt`` is deliberately no
    longer written, since only the head/tail frames have to survive on disk.

    The gate therefore has to accept a clip, not just a legacy tensor: testing
    only the tensor path excluded every modern render and silently degraded a
    continuous export into one standalone clip per segment, never reaching the
    clip+seam merge.

    Both lookups fall back to the superseded file group, matching the
    availability probe: after a plan edit the newest render sits one generation
    back. A clip that exists but fails to decode is accepted here — the merge
    then raises and the caller falls back to a standalone export.
    """
    if (
        resolve_segment_cache_path(
            node_id, seg, plan, allow_stale=True,
            workflow_name=workflow_name, variant=variant,
        )
        is not None
    ):
        return True
    return has_segment_clip(
        node_id, int(seg.index), workflow_name=workflow_name,
        allow_prev=True, variant=variant,
    )


def run_segment_export(
    node_id: str | None,
    plan: DirectorPlan,
    indices: list[int],
    *,
    mode: str = "piecewise",
    vae: Any = None,
    out_dir: str | None = None,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> dict[str, Any]:
    """Execute a「分段导出」request.

    ``variant`` is driven by the picker's「缓存来源」toggle. It selects which
    pass's file group every read below targets, so a「二采」export reads only
    ``seg2_*`` and can never silently mix in a first-pass render.

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
    from .plan_types import normalize_segment_export_mode

    normalized = normalize_segment_export_mode(mode)
    segments = _segments_by_index(plan)
    valid = sorted({int(i) for i in indices if int(i) in segments})
    if not valid:
        return {"files": [], "skipped": [], "export_dir": "", "mode": normalized, "source": variant}

    # Unified decode first: any checked segment that has only a latent (no frames)
    # is decoded once and written back to the frame cache, so the exports below
    # read frames directly instead of re-decoding per consumer.
    pre = predecode_latent_segments(
        node_id, plan, valid, vae=vae, workflow_name=workflow_name, variant=variant
    )
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
        if variant == segment_slots.VARIANT_SECOND:
            base = base / "second_pass"
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
        # ``allow_prev``: right after a plan edit the newest clip lives under the
        # superseded group, and that is exactly what the availability probe just
        # reported as exportable. Without the fallback the export would skip a
        # segment the UI marked as ready.
        clip = clip_cache_path(
            node_id, idx, workflow_name=workflow_name, allow_prev=True, variant=variant
        )
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
        """Export one segment on its own: verbatim clip copy → latent decode.

        A piecewise export is one standalone video per segment, so the finished
        render in the clip cache IS the output: copy it byte-for-byte instead of
        re-decoding the latent or re-stitching anything (no seam pass, no re-trim,
        no chance of the export differing from what the run produced).
        Only when no clip exists yet do we fall back to decoding the latent.
        """
        clip = clip_cache_path(
            node_id, idx, workflow_name=workflow_name, allow_prev=True, variant=variant
        )
        if clip is not None and clip.is_file() and clip.stat().st_size > 0:
            _export_clip_copy(idx)
            return
        source = _load_segment_export_source(
            node_id, segments[idx], plan, vae=vae,
            workflow_name=workflow_name, variant=variant,
            # Straight to an encoder that accepts uint8 — no expansion needed.
            dtype=torch.uint8,
        )
        if source is None:
            skipped.append({"index": idx, "reason": "no exportable cache"})
            return
        _export_one(segments[idx], source[0], source[1], tag=f"seg_{idx + 1:02d}")

    def _segment_audio(idx: int) -> dict[str, Any] | None:
        audio = load_segment_audio(
            node_id, segments[idx], plan, allow_stale=True,
            workflow_name=workflow_name, variant=variant,
        )
        return audio if isinstance(audio, dict) else None

    if normalized != "continuous":
        for idx in valid:
            _export_standalone(idx)
        return {
            "files": files,
            "skipped": skipped,
            "export_dir": export_dir,
            "mode": normalized,
            "source": variant,
        }

    # --- 连续导出: stitch runs of adjacent checked segments ------------------
    # A segment can be stitched when it has any frame source on disk — normally
    # the encoded clip, whose seam frames are restored from ``frames_ht``;
    # see :func:`_segment_can_stitch`. The merge streams the clips in from disk
    # (peak ≈ result + one segment, same as「全部导出」) instead of holding a whole
    # run in RAM. Latent-only segments were predecoded above, so they are on disk
    # by now and take part like any other cached segment.
    from .segment_continuity import concat_chunks_lazy

    stitchable = [
        idx
        for idx in valid
        if _segment_can_stitch(
            node_id, segments[idx], plan, workflow_name=workflow_name, variant=variant
        )
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
            merged = concat_chunks_lazy(
                node_id, plan, run_segs, workflow_name=workflow_name, variant=variant
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "分段导出 连续导出: stitching #%d–#%d failed (%s); "
                "falling back to standalone clips.",
                first + 1, last + 1, exc,
            )
            for i in run:
                _export_standalone(i)
            continue
        counts = _run_frame_counts(
            node_id, plan, run_segs, int(merged.shape[0]),
            workflow_name=workflow_name, variant=variant,
        )
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
        "source": variant,
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
    # No ``.float()``: that turns uint8 [0,255] into float [0,255] rather than
    # [0,1], and the encoder then clips nearly every pixel to white. The writer
    # handles both domains.
    _write_via_temp(
        Path(path),
        lambda tmp: write_frames_to_mp4(tmp, frames.detach().cpu(), fps=fps, audio=audio),
    )
    return path
