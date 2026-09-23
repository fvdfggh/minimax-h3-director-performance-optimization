"""Pieces the batch executor needs *around* its phases.

Extracted verbatim from :mod:`batch_executor`, which now holds only the
orchestrator (`execute_director_batch`) and the prepare phase
(:mod:`batch_prepare`). Three kinds of thing live here:

* **adapters** — ``_unpack_node_output``, ``_build_minimax_inputs``,
  ``_ref_tensor_from_seg_refs``: shape a segment's materials into what the
  official MiniMax nodes expect;
* **decode** — ``_decode_av_latent`` / ``_trim_decoded_to_export``: latent →
  (frames, audio), plus the trim that makes those frames match the export length.
  :mod:`second_sampling` re-uses this exact pair, so it must keep behaving
  identically for both callers;
* **intermediates IO** — the ``_save_*`` / ``_load_*`` pairs, ``_latent_for_cache``,
  ``_batch_cache_dir`` and ``_clear_batch_cache``: the phase boundary. Phase 1
  writes these, phase 2 reads the conditioning and writes latents, phase 3 reads
  the latents. Keeping the on-disk shape in one module is what makes a resumed
  run and a fresh run agree.

``_prev_context_available`` and ``_assemble_export_list`` are the two small
planners that do not belong to any phase above.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from . import cache_layout
from . import segment_slots
from .audio_export import empty_audio_dict
from .cache_files import probe_segment_cache_shape
from .cache_readback import load_segment_audio
from .cache_store import load_segment_av_latent
from .h3_motion_context import trim_context_prefix, video_from_latent
from .plan import (
    ref_audios_to_dict,
    ref_video_audios_to_dict,
    ref_videos_to_dict,
    refs_to_kwargs_for_context,
)
from .plan_types import DirectorPlan
from .segment_runtime import segment_passthrough_chunk

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.batch.helpers")


def _unpack_node_output(out):
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    raise RuntimeError(f"Unexpected node output: {type(out)!r}")


def _decode_av_latent(samples, vae, audio_vae, *, decode_audio=True):
    from comfy_extras.nodes_lt import LTXVSeparateAVLatent
    from nodes import VAEDecode

    sep = LTXVSeparateAVLatent.execute(samples)
    video_latent, audio_latent = _unpack_node_output(sep)[:2]
    images, = VAEDecode().decode(vae, video_latent)
    if not decode_audio or audio_vae is None:
        return images, empty_audio_dict()
    try:
        from comfy_extras.nodes_audio import VAEDecodeAudio
    except ImportError:
        from comfy_extras.nodes_lt import VAEDecodeAudio
    audio_out = VAEDecodeAudio.execute(audio_vae, audio_latent)
    audio = _unpack_node_output(audio_out)[0]
    return images, audio


def _trim_decoded_to_export(decoded, audio_dict, *, trim_frames, export_len, plan):
    if trim_frames > 0:
        decoded, audio_dict = trim_context_prefix(
            decoded, audio_dict, trim_frames,
            fps=float(plan.frame_rate or 24), match_tail=True,
        )
    if decoded.shape[0] > export_len:
        decoded = decoded[:export_len]
        if isinstance(audio_dict, dict) and audio_dict.get("waveform") is not None:
            sr = int(audio_dict.get("sample_rate") or 32000)
            want = int(round((export_len / float(plan.frame_rate or 24)) * sr))
            wf = audio_dict["waveform"]
            if int(wf.shape[-1]) > want:
                audio_dict = {"waveform": wf[..., :want], "sample_rate": sr}
    return decoded, audio_dict


def _ref_tensor_from_seg_refs(refs, index):
    for ref in refs or []:
        if int(getattr(ref, "index", -1)) == index and ref.tensor is not None:
            t = ref.tensor
            if t.shape[0] > 0:
                return t[:1]
    return None


def _build_minimax_inputs(plan, seg, *, clip_frames, ctx_w, ctx_h, prev_tail):
    task_key = seg.task_key
    first_frame = last_frame = ref_images = ref_videos = ref_audios = ref_video_audios = None

    if task_key == "fl2v":
        first_frame = _ref_tensor_from_seg_refs(seg.refs, 0)
        last_frame = _ref_tensor_from_seg_refs(seg.refs, 1)
        if first_frame is not None and last_frame is None and clip_frames is not None:
            if clip_frames.shape[0] >= 2:
                last_frame = clip_frames[-1:].clone()
    elif task_key == "i2v":
        if clip_frames is not None and clip_frames.shape[0] > 0:
            first_frame = clip_frames[:1]
        else:
            first_frame = _ref_tensor_from_seg_refs(seg.refs, 0)
        del prev_tail
    elif task_key == "r2v":
        ref_kwargs = refs_to_kwargs_for_context(task_key, seg.refs)
        ref_images = {}
        for key, tensor in ref_kwargs.items():
            ref_images[key] = tensor
        ref_videos_dict = ref_videos_to_dict(getattr(seg, "ref_videos", None) or [])
        if ref_videos_dict:
            ref_videos = ref_videos_dict
        ref_audios_dict = ref_audios_to_dict(getattr(seg, "ref_audios", None) or [])
        if ref_audios_dict:
            ref_audios = ref_audios_dict
        # Read seg.ref_video_audios (SegmentRefVideo has no `.audio`) and emit the
        # official `ref_video_audio_<N>` keys, so soundtracks actually pair up.
        ref_video_audios = ref_video_audios_to_dict(getattr(seg, "ref_video_audios", None))
    elif task_key == "v2v":
        ref_videos_dict = ref_videos_to_dict(getattr(seg, "ref_videos", None) or [])
        if ref_videos_dict:
            ref_videos = ref_videos_dict
    elif task_key == "rv2v":
        ref_kwargs = refs_to_kwargs_for_context(task_key, seg.refs)
        ref_images = {}
        for key, tensor in ref_kwargs.items():
            ref_images[key] = tensor
        ref_videos_dict = ref_videos_to_dict(getattr(seg, "ref_videos", None) or [])
        if ref_videos_dict:
            ref_videos = ref_videos_dict
        ref_audios_dict = ref_audios_to_dict(getattr(seg, "ref_audios", None) or [])
        if ref_audios_dict:
            ref_audios = ref_audios_dict
        ref_video_audios = ref_video_audios_to_dict(getattr(seg, "ref_video_audios", None))

    return first_frame, last_frame, ref_images, ref_videos, ref_audios, ref_video_audios


def _batch_cache_dir(node_id: int, workflow_name: str | None = None) -> Path:
    """Directory holding this run's intermediates.

    Shares the one Director cache folder with the durable segment artefacts, so
    scratch files are marked ``seg_XXXX_scratch_<kind>.pt`` and told apart by
    that prefix alone — see :mod:`cache_layout`.

    Namespaced by workflow name just like the encoding cache: node ids are
    per-graph and get reused across workflow files, so without this two
    workflows would overwrite each other's scratched latents.

    File names stay fixed per segment, so a re-run overwrites in place rather
    than accumulating.
    """
    return cache_layout.node_cache_dir(str(node_id), workflow_name)


def _clear_batch_cache(cache_dir: Path, reports: list[str]) -> None:
    """Delete this run's scratch cache (Phase 1/2/3 intermediates).

    Scratch files now live beside the durable segment artefacts, so this removes
    only the ``seg_*_scratch_*`` files — never the whole directory. Deleting the
    directory would take the rendered frames / AV latents with it, which is what
    motion context and「全部导出」read back.

    When the option is off, files keep fixed names and are simply overwritten by
    the next run.
    """
    try:
        if not cache_dir.is_dir():
            reports.append("Batch cache: nothing to clear")
            return
        removed = 0
        failed = 0
        for path in cache_layout.iter_scratch_files(cache_dir):
            try:
                path.unlink()
                removed += 1
            except OSError:
                failed += 1
        if failed:
            reports.append(f"Batch cache: removed {removed} file(s), {failed} locked")
        else:
            reports.append(f"Batch cache cleared: {removed} scratch file(s) in {cache_dir}")
    except Exception as exc:  # never fail the run over cache cleanup
        log.warning("Director batch: failed to clear batch cache %s: %s", cache_dir, exc)
        reports.append(f"Batch cache: cleanup failed ({exc})")


def _latent_for_cache(node_id, seg_index, completed_av_latents, cache_dir):
    """Return the AV latent to persist into seg_cache, back-filling from disk.

    The next segment pins its motion context from seg_cache's av latent, so this
    must never return None for a segment we just decoded. When Phase 3 loaded the
    latent from the batch scratch dir (fallback path), it was never inserted into
    ``completed_av_latents`` — back-fill it so save_segment_cache persists it.

    This also covers the final segment: nothing follows it in this run, but a
    later run may append segments after it, and those will pin from here.
    """
    lat = completed_av_latents.get(seg_index)
    if lat is not None:
        return lat
    lat = _load_batch_latent(node_id, seg_index, cache_dir)
    if lat is not None:
        completed_av_latents[seg_index] = lat
    return lat


def _save_batch_conditioning(node_id, seg_index, positive, negative, cache_dir):
    """Save pre-encoded conditioning to disk (per-run scratch).

    The initial AV latent is deliberately left out: it is always all zeros and
    a pure function of ``(ctx_w, ctx_h, sample_len)``, which the sampling loop
    already has, so it is rebuilt on read rather than costing ~6 MB of scratch
    I/O per segment per run.
    """
    path = cache_layout.scratch_path(cache_dir, seg_index, "cond")
    torch.save({
        "positive": positive,
        "negative": negative,
    }, path, _use_new_zipfile_serialization=True)
    return path


def _load_batch_conditioning(node_id, seg_index, cache_dir):
    """Load pre-encoded conditioning from disk (per-run scratch)."""
    path = cache_layout.scratch_path(cache_dir, seg_index, "cond")
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def _save_batch_ref(node_id, seg_index, ref_data, cache_dir):
    """Save pre-processed reference data to disk (per-run scratch)."""
    path = cache_layout.scratch_path(cache_dir, seg_index, "ref")
    torch.save(ref_data, path, _use_new_zipfile_serialization=True)
    return path


def _load_batch_ref(node_id, seg_index, cache_dir):
    """Load pre-processed reference data from disk (per-run scratch)."""
    path = cache_layout.scratch_path(cache_dir, seg_index, "ref")
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_batch_latent(node_id, seg_index, samples, cache_dir):
    """Persist the sampled AV latent to the durable, content-addressed segment cache.

    Writes ``seg_<hash>_latent.pt`` (the same slot that ``segment_cache.save_segment_cache``
    fills in Phase 3), NOT a per-run scratch file. This keeps a single on-disk copy of the
    latent and lets the next segment — same run or a later run — pin its motion context
    straight from the segment cache via ``load_segment_av_latent`` instead of from a scratch
    file that ``iter_scratch_files`` would delete.

    A latent-only file (no ``.meta.json`` yet) is deliberately *not* a "finished" segment:
    strict readers (``allow_stale=False``) still require the meta and ignore it, so it never
    spoofs a complete cache that continuity /「全部导出」would trust. ``save_segment_cache``
    in Phase 3 then completes the group with frames + meta.
    """
    stem = segment_slots.resolve_stem(cache_dir, seg_index)
    if not stem:
        log.warning("Director batch: no cache slot for segment %d, latent not persisted", seg_index + 1)
        return None
    path = cache_layout.segment_paths(cache_dir, stem)["latent"]
    # Move to CPU for disk storage
    cpu_samples = {}
    for k, v in samples.items():
        if isinstance(v, torch.Tensor):
            cpu_samples[k] = v.cpu()
        else:
            cpu_samples[k] = v
    torch.save(cpu_samples, path, _use_new_zipfile_serialization=True)
    return path


def _load_batch_latent(node_id, seg_index, cache_dir):
    """Load a sampled AV latent from the durable segment cache.

    Position-addressed (mirrors :func:`segment_cache.load_segment_av_latent`) and without the
    fingerprint gate, so a latent-only file written by ``_save_batch_latent`` in Phase 2 —
    before the meta exists — still loads. Used as the fallback when the stricter
    ``load_segment_av_latent`` did not return one.
    """
    stem = segment_slots.resolve_stem(cache_dir, seg_index)
    if not stem:
        return None
    path = cache_layout.segment_paths(cache_dir, stem)["latent"]
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _av_latent_canvas_matches(av_latent, width: int, height: int) -> bool:
    """True when apply_motion_context would take its *latent* path for this AV latent.

    Must mirror the check inside apply_motion_context exactly, otherwise the two
    disagree: the probe would report "pinnable", Phase 2 would fall through to
    the pixel path, and resolve_prev_segment_output would then raise because
    that path needs a frame cache a latent-only re-run does not have.

    Hence the reuse of video_from_latent — the AV latent is stored under
    "samples" (a NestedTensor), not "video"; indexing it by hand is what made
    an earlier version of this return False unconditionally. Any failure here
    degrades to False (treat as unpinnable) rather than propagating, since the
    probe only needs a conservative answer.
    """
    try:
        src = video_from_latent(av_latent)
        return (
            int(src.shape[4]) * 16 == int(width)
            and int(src.shape[3]) * 16 == int(height)
        )
    except Exception:
        return False


def _prev_context_available(
    node_id,
    plan: DirectorPlan,
    all_segments: list,
    seg_index: int,
    run_indices: set,
    completed_av_latents: dict,
    width: int = 0,
    height: int = 0,
    workflow_name: str | None = None,
) -> bool:
    """Can segment ``seg_index`` pin its timeline predecessor?

    Phase 1 must already know: ``sample_len`` is baked into the conditioning
    here, so the motion-context decision cannot be revisited in Phase 2. Without
    this probe,「选择运行」with a gap (e.g. run seg 1, then seg 4) reaches
    ``apply_motion_context`` with nothing to pin and raises
    "need previous segment latent or decoded frames". Degrade to a no-pin
    segment there instead of failing the run.

    The predecessor's AV latent is reachable in one of two ways:
      * it is sampled earlier in this very run (``run_list`` keeps timeline
        order, so any predecessor inside ``run_indices`` comes first), or
      * it is already on disk from a previous run.

    On a disk hit the latent is kept in ``completed_av_latents`` — Phase 2 would
    load it anyway, so this costs no extra I/O.
    """
    prev_idx = int(seg_index) - 1
    if prev_idx < 0:
        return False
    if prev_idx in completed_av_latents:
        return True
    if prev_idx in run_indices:
        return True
    prev_seg = next((s for s in all_segments if s.index == prev_idx), None)
    if prev_seg is None:
        return False
    prev_av = load_segment_av_latent(node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name)
    if prev_av is None:
        return False
    # A canvas mismatch would send apply_motion_context down its pixel path,
    # which now has nothing to pin (see _av_latent_canvas_matches).
    if width > 0 and height > 0 and not _av_latent_canvas_matches(prev_av, width, height):
        return False
    completed_av_latents[prev_idx] = prev_av
    return True


def _assemble_export_list(
    node_id,
    plan: DirectorPlan,
    all_segments: list,
    *,
    decoded_frames: dict[int, int],
    completed_audios: dict[int, dict],
    reports: list[str],
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> tuple[list, list[dict], list[int], dict[int, torch.Tensor]]:
    """Timeline-ordered export list for「全部导出」.

    Batch mode only samples「选择运行」, but the merge must still cover the whole
    timeline — unselected slots are restored from the segment cache (exact
    fingerprint first, then stale) or a source passthrough.
    Returns ``(segments, audios, frame_counts, memory_chunks)``
    aligned 1:1.

    ``variant`` picks which pass's cache fills an unselected slot — it is the
    picker's「缓存来源」toggle, so a「二采」merge must never pull a first-pass
    render in here (that would silently undo the upscale).

    ``memory_chunks`` holds the passthrough fills: those exist **only** in RAM,
    so they must be handed to ``concat_chunks_lazy`` as overrides instead of
    being dropped — dropping them makes the merge fail with a cache miss on the
    very segment that was just accepted. Stale cache hits are not stored here;
    they stay on disk and the merge re-reads them with the same stale policy.
    """
    segments: list = []
    audios: list[dict] = []
    frame_counts: list[int] = []
    memory_chunks: dict[int, torch.Tensor] = {}
    skipped: list[int] = []

    for seg in all_segments:
        if seg.index in decoded_frames:
            segments.append(seg)
            frame_counts.append(int(decoded_frames[seg.index]))
            audios.append(completed_audios.get(seg.index) or {})
            continue

        # Probe the header only: this branch used to load the whole clip just to
        # read ``shape[0]`` and then throw the pixels away, so every unselected
        # segment was read twice per run (once here, once by the merge).
        cached_shape = probe_segment_cache_shape(
            node_id, seg, plan, workflow_name=workflow_name, variant=variant
        )
        used_stale = False
        if cached_shape is None:
            cached_shape = probe_segment_cache_shape(
                node_id, seg, plan, allow_stale=True,
                workflow_name=workflow_name, variant=variant,
            )
            used_stale = cached_shape is not None
        if cached_shape is not None:
            n_frames = int(cached_shape[0])
            cached_audio = load_segment_audio(
                node_id, seg, plan, allow_stale=used_stale,
                workflow_name=workflow_name, variant=variant,
            )
            if not isinstance(cached_audio, dict):
                cached_audio = {}
            segments.append(seg)
            frame_counts.append(n_frames)
            audios.append(cached_audio)
            reports.append(
                f"  Seg #{seg.index + 1}: cache fill for「全部导出」({n_frames}f"
                f"{', +audio' if cached_audio else ', no audio cache'}"
                f"{', stale fingerprint' if used_stale else ''})"
            )
            continue

        fill = segment_passthrough_chunk(plan, seg)
        if fill is None:
            skipped.append(seg.index + 1)
            reports.append(
                f"  Seg #{seg.index + 1}: skipped — no cache "
                "(outside run selection; omitted from「全部导出」merge)"
            )
            continue
        segments.append(seg)
        frame_counts.append(int(fill.shape[0]))
        audios.append({})
        # Not on disk — keep it alive for the merge (see docstring).
        memory_chunks[int(seg.index)] = fill
        reports.append(
            f"  Seg #{seg.index + 1}: source passthrough ({int(fill.shape[0])}f, "
            "not sampled — outside run selection)"
        )

    if skipped:
        reports.append(
            f"Omitted from「全部导出」(no cache): segment(s) {skipped} — "
            "勾选重跑或先全跑可补上。"
        )
    return segments, audios, frame_counts, memory_chunks
