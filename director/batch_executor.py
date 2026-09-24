"""Batch mode executor: three-phase processing to minimize model loading.

Phase 1 (prepare):  Pre-encode text/audio/ref → disk
Phase 2 (sample):   UNet stays loaded, sequential sampling → latent to disk
Phase 3 (decode):   VAE stays loaded, sequential decode → mp4 export

Key constraint: segment N+1's conditioning depends on segment N's tail latent
(motion context), so sampling is sequential but models stay resident.
"""

from __future__ import annotations

import gc
import logging
import re
from typing import Any

import torch

from ..lib.image_prep import assert_minimax_canvas, fit_canvas, fit_video_long_edge
from ..lib.media_b64 import tensor_frame_to_jpeg_b64
from . import segment_slots
from .batch_helpers import (
    _assemble_export_list,
    _av_latent_canvas_matches,
    _batch_cache_dir,
    _build_minimax_inputs,
    _clear_batch_cache,
    _decode_av_latent,
    _latent_for_cache,
    _load_batch_conditioning,
    _load_batch_latent,
    _load_batch_ref,
    _prev_context_available,
    _save_batch_conditioning,
    _save_batch_latent,
    _save_batch_ref,
    _trim_decoded_to_export,
)
from .batch_prepare import (
    prepare_segment_materials,
    encode_text_batch,
    encode_video_vae_batch,
    encode_audio_vae_batch,
    assemble_conditioning,
    unload_model_group,
    _rebuild_empty_latent,
)
from .conditioning_cache import (
    clear_conditioning_cache,
    load_conditioning_cache,
    save_conditioning_cache,
    save_segment_second_params,
    slugify_workflow_name,
    text_cache_key,
)
from .core_sampling import sample_single_stage
from .frame_align import minimax_align_frame_count
from .audio_export import (
    AUDIO_MODE_GENERATE, AUDIO_MODE_MUTE, AUDIO_MODE_SOURCE,
    empty_audio_dict, resolve_audio_mode,
)
from .segment_runtime import frames_label, resolve_segment_raw_clip
from .plan import (
    prepare_segment_clip,
    ref_audios_to_dict, ref_video_audios_to_dict, ref_videos_to_dict,
    reference_video_for_segment,
    refs_to_kwargs_for_context, reinforce_r2v_prompt, reinforce_rv2v_prompt, reinforce_v2v_prompt,
)
from .plan_types import DirectorPlan, resolve_ref_image_size
from .progress import report_director_finish, report_director_progress, report_director_segment_preview
from .h3_motion_context import (
    DEFAULT_AUDIO_CONTEXT_FRAMES, apply_motion_context,
    generation_frame_budget, handoff_end_frame,
    resolve_tail_context_length,
    snap_context_frames, trim_context_prefix, trim_export_tail,
    video_from_latent,
    TAIL_CONTEXT_FRAMES,
)
from .h3_latent_continue import apply_latent_continue
from .segment_cache import (
    build_run_selection_clips, continuous_export_runs,
    load_next_segment_av_latent,
    load_segment_audio, load_segment_av_latent,
    load_segment_handoff_meta, probe_segment_cache_shape,
    save_segment_cache, slot_content_hash,
    sync_segment_slots,
)
from .segment_mp4_export import (
    maybe_export_segment_mp4, new_segment_mp4_run_dir, export_run_mp4, run_mp4_path,
)
from .segment_continuity import concat_chunks_lazy, is_continuity_active, resolve_prev_segment_output
from .vram_cleanup import cleanup_segment_vram
from .batch_source_audio import build_source_audio_cache
from .batch_phases import (
    _prepare_one_segment,
    _sample_one_segment,
    _decode_export_one_segment,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.batch")

# ---------------------------------------------------------------------------
# Main batch executor
# ---------------------------------------------------------------------------

def execute_director_batch(
    plan: DirectorPlan,
    *,
    node_id: int,
    model,
    vae,
    audio_vae,
    clip,
    cfg: float,
    seed: int,
    steps: int,
    sampler: str,
    scheduler: str,
    shift_video: float,
    shift_audio: float,
    sigmas=None,
    use_conditioning_cache: bool = False,
    clear_conditioning_cache_on_run: bool = False,
    clear_vram_between_segments: bool = True,
    workflow_name: str | None = None,
    progress_cb=None,
    # Connected-frame (r2v/v2v/rv2v) taper noise — applied to the pinned
    # reference frames before they are handed to H3 Motion Context. 0 disables.
    conn_noise: bool = False,
) -> tuple:
    """Three-phase batch execution.

    This is the only execution path. Returns
    ``(combined, segment_outputs, segment_audios, report, export_frame_counts)``.
    """
    # conn_noise 开关：开启 = 启用段间锥形重绘(continue 模式)，关闭 = 仅参考帧引导(guide)。
    if not conn_noise:
        plan.continuity_redraw = 0.0
    all_segments = plan.segments
    # Reconcile cache files with the current timeline first: a group deleted in
    # the middle takes its own files, everyone else keeps their own render.
    sync_segment_slots(node_id, plan, workflow_name=workflow_name)
    # ``ext_meta`` is the raw externalGroups payload — a plain dict, NOT a plan.
    # Audio mode must be resolved from the plan, which owns raw["output"]["audioMode"].
    ext_meta = (plan.raw or {}).get("externalGroups") or {}
    audio_mode = resolve_audio_mode(plan)
    decode_audio = audio_mode == AUDIO_MODE_GENERATE
    raw_live = (plan.raw or {}).get("liveTaePreview", (plan.raw or {}).get("live_tae_preview", True))
    live_tae_preview = False if raw_live in (False, 0, "0", "false", "False", "off") else True

    # UI card count drives progress labels; timeline rows are the floor because
    # unselected slots still fill「全部导出」from cache/source.
    timeline_seg_total = len(all_segments)
    try:
        timeline_seg_total = max(timeline_seg_total, int(ext_meta.get("count") or 0))
    except (TypeError, ValueError):
        pass

    run_indices = set(plan.run_indices) if plan.run_indices is not None else set(range(len(all_segments)))

    # 「分段导出」: the checked segments already live in the cache, so there is
    # NOTHING to sample here. Sampling would re-run the UNet on the whole
    # timeline (and apply seam luma on every segment), which is exactly what an
    # export-only run must avoid. We sample zero segments; the export reads
    # frames / latents from disk and decodes with the already-loaded VAE.
    seg_export = getattr(plan, "segment_export", None)
    # Trigger on the one-shot ``enabled`` flag AND the checked ``indices``. The
    #「分段导出」button sets both; a plain 运行 must not export.
    #
    # ``indices`` alone is NOT a valid trigger: it is persisted in the timeline so
    # the picker reopens with the last selection, so once anything was ever checked
    # every subsequent run became an export-only pass (run_indices emptied → zero
    # sampling). The flag is safe to trust: ``app.queuePrompt`` is patched to flush
    # every Director timeline synchronously *before* the prompt is built, so the
    # widget snapshot always carries the flag as it was when the button was pressed.
    seg_export_active = (
        seg_export is not None and bool(seg_export.enabled) and bool(seg_export.indices)
    )
    if seg_export is not None and seg_export.indices and not seg_export.enabled:
        # Checked but not triggered: a normal generation run. Log it so an
        # accidental no-op export is visible rather than silent.
        log.info(
            "分段导出: %d 个片段已勾选，但未由「分段导出」按钮触发 — 按正常生成运行。",
            len(seg_export.indices),
        )
    if seg_export_active:
        run_indices = set()

    run_list = [seg for seg in all_segments if seg.index in run_indices]

    if not run_list and not seg_export_active:
        raise ValueError("Batch mode: no segments to run.")

    # Lives from Phase 1 on: the motion-context probe in Phase 1 may back-fill a
    # predecessor's AV latent here, which Phase 2 then reuses instead of re-reading.
    completed_av_latents: dict[int, dict] = {}

    reports = [
        f"Batch mode: {len(run_list)}/{timeline_seg_total} segments",
        f"Phase 1: Pre-encode all conditioning → disk",
        f"Phase 2: Sequential sampling (UNet stays loaded)",
        f"Phase 3: Sequential decode (VAE stays loaded)",
    ]
    if audio_mode == AUDIO_MODE_MUTE:
        reports.append("Audio: muted — silent AUDIO output.")
    elif audio_mode == AUDIO_MODE_SOURCE:
        reports.append("Audio: source — extract source PCM once in Phase 1, reuse in Phase 3.")
    else:
        reports.append("Audio: generate — encode → sample → decode model audio.")

    # ===================================================================
    # SOURCE MODE: pre-extract the source PCM for every running segment
    # ===================================================================
    # The soundtrack of a source-mode run comes from the source video, not the
    # AV latent, so extract it once here (Phase 1) and reuse the tensor in
    # Phase 3 + the mp4 mux instead of re-reading the file per segment.
    #
    # Mute mode has no soundtrack at all and generate mode decodes model audio:
    # neither builds this cache, so the extraction only ever happens for source.
    source_audio_cache: dict[int, dict] = {}
    if audio_mode == AUDIO_MODE_SOURCE:
        try:
            source_audio_cache = build_source_audio_cache(
                run_list=run_list,
                plan=plan,
                fps=float(plan.frame_rate or 24),
            )
        except Exception as exc:
            log.warning("Source audio cache failed: %s — continuing without it", exc, exc_info=True)

    # 「保留音频」: every card whose 音频 tab has an entry ticked. Independent of
    # audio_mode — it is an explicit per-segment choice, so it works under
    # generate / source / mute alike.
    retain_audio_cache: dict[int, dict] = {}
    try:
        from .audio_retain import build_retain_audio_cache

        retain_audio_cache = build_retain_audio_cache(
            node_id=node_id, plan=plan, workflow_name=workflow_name, run_list=run_list,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("保留音频: 读取失败，本次全部按正常生成处理 (%s)。", exc)
    if retain_audio_cache:
        reports.append(
            f"Audio: 保留音频 — {len(retain_audio_cache)} 段使用提取音频"
            "（采样时锁定，输出直接复用原音频）"
        )

    cache_dir = _batch_cache_dir(node_id, workflow_name)

    # Report current memory
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        reports.append(f"GPU memory: {alloc:.1f}GB allocated, {reserved:.1f}GB reserved")

    # ===================================================================
    # PHASE 1: Pre-encode all conditioning → disk
    # ===================================================================
    # Opt-in wipe before anything is encoded: when a prompt/canvas changed but
    # the on-disk hash still matches an old variant, the run would otherwise be
    # silently served from that stale cache.
    if clear_conditioning_cache_on_run:
        try:
            removed = clear_conditioning_cache(node_id, workflow_name=workflow_name)
            reports.append(f"Conditioning cache cleared on run: {removed} file(s) removed")
        except Exception as exc:
            log.warning("Conditioning cache clear skipped (%s).", exc)

    reports.append("=" * 50)
    reports.append("PHASE 1: Preparing materials, then encoding by model group...")

    # Step 0 runs with no model resident: canvas fit, resize and frame-grid
    # alignment are plain tensor work. Only after every segment's pixels are
    # staged do we load CLIP, then the video VAE, then the audio VAE — one at a
    # time, released before the next is loaded.
    pending: list[dict] = []        # segments still needing an encoder
    pending_meta: list[dict] = []   # per-segment context for cache + reporting
    cache_hits = 0
    # Every text hash this run consumes — cache-served and freshly encoded alike.
    # Anything on disk outside this set is stale and gets pruned after Step 1.
    used_text_keys: set[str] = set()
    # Shared across all segments: in global edit mode every segment points at the
    # same reference tensors, so identical media encodes once instead of N times.
    vae_cache: dict = {}
    aud_cache: dict = {}

    for seg in run_list:
        seg_pos = run_list.index(seg)
        staged_entry = _prepare_one_segment(
            plan, seg, seg_pos,
            node_id=node_id, all_segments=all_segments, run_indices=run_indices,
            completed_av_latents=completed_av_latents, cache_dir=cache_dir,
            workflow_name=workflow_name, use_conditioning_cache=use_conditioning_cache,
            audio_vae=audio_vae, timeline_seg_total=timeline_seg_total,
            segment_total=len(run_list), reports=reports,
        )
        used_text_keys.add(staged_entry["text_key"])
        if staged_entry["cache_hit"]:
            cache_hits += 1
            continue
        pending.append(staged_entry["staged"])
        pending_meta.append(staged_entry["meta"])

    # ---- Step 1: text encoder -------------------------------------------------
    if pending:
        n_deduped = len(pending) - len(used_text_keys)
        reports.append(
            f"  text encoder: {len(used_text_keys)} encode(s) for {len(pending)} segment(s)"
            + (f", {n_deduped} deduped" if n_deduped > 0 else "")
            + (f", {cache_hits} cache hit(s)" if cache_hits else "")
        )
        n_reused = encode_text_batch(clip, pending)
        if n_reused:
            reports.append(f"  text encoder: {n_reused} segment(s) shared an encoding")
        unload_model_group(clip, reports=reports, label="text encoder (CLIP)")

        # ---- Step 2: video VAE ----------------------------------------------
        # One cache across every segment: shared reference media is encoded once.
        n_img_jobs = sum(len(p["image_jobs"]) for p in pending)
        n_aud_jobs = sum(len(p["audio_jobs"]) for p in pending)
        n_img = encode_video_vae_batch(vae, pending, vae_cache)
        if n_img:
            unload_model_group(vae, reports=reports, label="video VAE")
        shared = n_img_jobs - n_img
        reports.append(f"  video VAE: {n_img} encode(s)"
                       + (f", {shared} shared" if shared else ""))

        # ---- Step 3: audio VAE ----------------------------------------------
        # Encodes *reference* audio (seg.ref_audios, reference-video soundtracks)
        # into minimax_refs conditioning. This is input conditioning, required in
        # every mode — source mode only skips decoding the *output* audio from the
        # AV latent, not the reference encoding.
        n_aud = (encode_audio_vae_batch(audio_vae, pending, aud_cache)
                 if audio_vae is not None else 0)
        if n_aud:
            unload_model_group(audio_vae, reports=reports, label="audio VAE")
        shared = n_aud_jobs - n_aud
        reports.append(f"  audio VAE: {n_aud} encode(s)"
                       + (f", {shared} shared" if shared else ""))

        # ---- Step 4: assemble + persist --------------------------------------
        for prepared, meta in zip(pending, pending_meta):
            seg = meta["seg"]
            result = assemble_conditioning([prepared])[0]
            positive, negative, latent = result["positive"], result["negative"], result["latent"]
            _save_batch_conditioning(node_id, seg.index, positive, negative, cache_dir)
            if use_conditioning_cache and not prepared.get("text_reused"):
                # Only one segment per key writes the shared encoding; the
                # dedupe path above skips the redundant rewrite.
                save_conditioning_cache(
                    node_id=node_id, segment_index=seg.index,
                    positive=positive, negative=negative, latent=latent,
                    prompt=meta["positive_prompt"], width=meta["ctx_w"], height=meta["ctx_h"],
                    length=meta["sample_len"], task_key=seg.task_key,
                    ref_image_size=meta["ref_image_size"], ref_images=meta["ref_images"],
                    workflow_name=workflow_name,
                    ref_videos=meta.get("ref_videos"),
                    first_frame=meta.get("first_frame"), last_frame=meta.get("last_frame"),
                    frame_count=prepared.get("frame_count"),
                )
            reports.append(f"  Seg #{seg.index + 1}: conditioning encoded → disk")
            del positive, negative, latent
    elif cache_hits:
        reports.append(f"  all {cache_hits} segment(s) served from conditioning cache")

    # NOTE: unused conditioning files are NOT pruned automatically after a run.
    # The "Clear cache" button in the UI handles cleanup on demand
    # (minimax_clear_cache → clear_conditioning_cache), which is sufficient.

    # Release the staged pixels/captions; Phase 2 only needs the latents now.
    pending.clear()
    pending_meta.clear()
    gc.collect()

    reports.append(f"Phase 1 complete: {len(run_list)} segments prepared")
    gc.collect()

    # ===================================================================
    # PHASE 2: Sequential sampling (UNet stays loaded)
    # ===================================================================
    reports.append("=" * 50)
    reports.append("PHASE 2: Sequential sampling with UNet resident...")

    completed_av_handoff: dict[int, dict] = {}
    completed_audios: dict[int, dict] = {}
    # Phase-align gap per segment: how many trailing frames must be dropped from
    # the *previous* export before concat (h3_motion_context.apply_motion_context
    # returns it as prev_export_trim_tail). Applied in Phase 3 once the previous
    # segment has been decoded.
    pending_prev_trim: dict[int, int] = {}
    # Highest timeline index; its tail must never be phase-trimmed.
    last_timeline_index = max((s.index for s in all_segments), default=None)

    # new_segment_mp4_run_dir already returns None unless export_mode == "segments".
    # Do not invert the condition here: both branches would yield None and the
    # per-segment mp4 would never be written.
    mp4_run_dir = new_segment_mp4_run_dir(plan)
    if mp4_run_dir is not None:
        reports.append(f"Segment mp4 export dir: {mp4_run_dir}")

    # Two-pass sampling: Pass 1 builds every latent with head references only
    # (tail disabled); Pass 2 re-samples just the 对齐下段 (NEXT/BOTH) segments,
    # pinning the next segment's *trimmed* opening now that its latent is on disk.
    # The next segment must be generated and exported before it can be referenced.
    _tail_segs = [s for s in run_list if bool(getattr(s, "continuity_to_next", False))]
    _seg_pass_list = [(s, False) for s in run_list]
    if _tail_segs:
        _seg_pass_list += [(s, True) for s in _tail_segs]

    for seg, _enable_tail in _seg_pass_list:
        _sample_one_segment(
            _enable_tail=_enable_tail,
            _seg_pass_list=_seg_pass_list,
            all_segments=all_segments,
            audio_mode=audio_mode,
            audio_vae=audio_vae,
            cache_dir=cache_dir,
            cfg=cfg,
            clear_vram_between_segments=clear_vram_between_segments,
            completed_audios=completed_audios,
            completed_av_handoff=completed_av_handoff,
            completed_av_latents=completed_av_latents,
            live_tae_preview=live_tae_preview,
            model=model,
            node_id=node_id,
            pending_prev_trim=pending_prev_trim,
            plan=plan,
            reports=reports,
            run_list=run_list,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            seg=seg,
            shift_audio=shift_audio,
            shift_video=shift_video,
            sigmas=sigmas,
            steps=steps,
            timeline_seg_total=timeline_seg_total,
            vae=vae,
            workflow_name=workflow_name,
            retain_audio_cache=retain_audio_cache,
        )

    reports.append(f"Phase 2 complete: {len(run_list)} latents saved")
    gc.collect()

    # ===================================================================
    # PHASE 3: Sequential decode (VAE stays loaded)
    # ===================================================================
    # Unload UNet to free VRAM for VAE (UNet ~20GB + VAE ~5GB > 22GB GPU)
    reports.append("=" * 50)
    reports.append("PHASE 3: Sequential decoding with VAE resident...")
    cleanup_segment_vram(enabled=True, unload_models=True)
    reports.append("  UNet unloaded, loading VAE...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reports.append(f"  VRAM after UNet unload: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    segment_outputs: list[torch.Tensor] = []
    segment_audios: list[dict[str, Any]] = []
    export_frame_counts: list[int] = []
    # Per-segment bookkeeping keyed by timeline index, so the merge can walk the
    # whole timeline (including slots this run did not sample) without index math.
    # Frames are NOT kept here — only lengths — to avoid pinning every chunk.
    decoded_segments: list = []
    decoded_frames: dict[int, int] = {}
    # Only the immediate previous export is kept: the next iteration may need to
    # trim its tail (phase-align), and holding every chunk would double peak RAM.
    prev_export_seg = None
    prev_export_chunk: torch.Tensor | None = None
    prev_export_audio: dict | None = None
    # segment_outputs is ordered by run_list position; map timeline index -> pos
    # so a tail trim can patch the already-finished previous segment in place.
    run_pos_map: dict[int, int] = {}

    for seg in run_list:
        _decode_export_one_segment(
            audio_mode=audio_mode,
            audio_vae=audio_vae,
            cache_dir=cache_dir,
            completed_audios=completed_audios,
            completed_av_handoff=completed_av_handoff,
            completed_av_latents=completed_av_latents,
            decode_audio=decode_audio,
            decoded_frames=decoded_frames,
            decoded_segments=decoded_segments,
            export_frame_counts=export_frame_counts,
            last_timeline_index=last_timeline_index,
            mp4_run_dir=mp4_run_dir,
            node_id=node_id,
            pending_prev_trim=pending_prev_trim,
            plan=plan,
            reports=reports,
            run_list=run_list,
            run_pos_map=run_pos_map,
            seg=seg,
            segment_audios=segment_audios,
            segment_outputs=segment_outputs,
            source_audio_cache=source_audio_cache,
            timeline_seg_total=timeline_seg_total,
            vae=vae,
            workflow_name=workflow_name,
            retain_audio_cache=retain_audio_cache,
        )

    reports.append(f"Phase 3 complete: {len(run_list)} segments decoded and exported")

    # Free AV latents (no longer needed after decode)
    completed_av_latents.clear()
    completed_av_handoff.clear()
    gc.collect()

    # ===================================================================
    # MERGE
    # ===================================================================
    report_director_finish(node_id, len(run_list))

    # Trigger on the checked indices (``seg_export_active``), consistent with the
    # Phase-1 decision above — NOT on ``seg_export.enabled``. The picker clears
    # ``enabled`` right after queueing, so a run that already skipped sampling
    # (run_indices = set()) used to fall through to the「全部导出」branch here and
    # crash on the unbound ``combined``.
    # ``merge_done`` marks a branch that already produced ``combined``; the generic
    # merge below then stands down instead of stitching the timeline a second time.
    merge_done = False
    if seg_export_active:
        # 「分段导出」: only the checked segments participate — no full-timeline
        # merge. This keeps an export-only run fast (it never renders or encodes
        # the other 30+ segments) and returns a combined clip of just the checked
        # segments as the node output; the per-segment mp4 files come from
        # ``run_segment_export`` below.
        #
        # Unified decode first: latent-only segments (no frame cache) are decoded
        # once and written back to ``seg_XXXX.pt``, so ``_load_segment_export_source``
        # below reads frames directly — no repeated per-consumer VAE decode.
        from .segment_cache import predecode_latent_segments as _predecode

        # 「缓存来源」(一采 seg_* / 二采 seg2_*) decides which pass's cache the
        # whole export below reads. Without threading it through, a「二采」export
        # silently pulled the first-pass render and undid the upscale.
        _seg_variant = (
            segment_slots.VARIANT_SECOND
            if seg_export.normalized_source() == segment_slots.VARIANT_SECOND
            else segment_slots.VARIANT_FIRST
        )
        _predecode(
            node_id,
            plan,
            list(seg_export.indices),
            vae=(vae, audio_vae) if (vae is not None or audio_vae is not None) else None,
            workflow_name=workflow_name,
            variant=_seg_variant,
        )
        wanted_segs = [s for s in all_segments if int(s.index) in set(seg_export.indices)]
        export_segments_list, segment_audios, export_frame_counts, merge_overrides = (
            _assemble_export_list(
                node_id, plan, wanted_segs,
                decoded_frames=decoded_frames,
                completed_audios=completed_audios,
                reports=reports,
                workflow_name=workflow_name,
                variant=_seg_variant,
            )
        )
        # 「分段导出」must NOT merge the checked segments into one video on the
        # images output — each checked segment (piecewise) / each contiguous run
        # (continuous) is a separate clip. Route through the segments layout so
        # the node emits one IMAGE per clip instead of a stitched merge. The
        # per-segment mp4 files come from ``run_segment_export`` below.
        plan.export_mode = "segments"
        # ``export_segments_list`` is a list of SegmentPlan objects (not frames).
        # Load each checked segment's frames from cache / latent for the images
        # output, reusing the exact source the export uses so IMAGE == mp4.
        from .segment_cache import _load_segment_export_source as _seg_src

        seg_by_index = {int(s.index): s for s in wanted_segs}
        frame_by_index = {}
        audio_by_index = {}
        for idx in sorted(set(int(i) for i in seg_export.indices)):
            s = seg_by_index.get(idx)
            if s is None:
                continue
            src = _seg_src(
                node_id, s, plan,
                vae=(vae, audio_vae) if (vae is not None or audio_vae is not None) else None,
                workflow_name=workflow_name,
                variant=_seg_variant,
            )
            if src is not None:
                frame_by_index[idx] = src[0]
                audio_by_index[idx] = src[1] if isinstance(src[1], dict) else {}
        # ``segment_audios`` from the export list is aligned with
        # ``export_segments_list``; fall back to it when the frame source carried
        # no audio (e.g. a stale frame cache without an audio sidecar).
        for s, a in zip(export_segments_list or [], segment_audios or []):
            idx = int(getattr(s, "index", -1))
            if idx >= 0 and not audio_by_index.get(idx) and isinstance(a, dict):
                audio_by_index[idx] = a

        _seg_mode = seg_export.normalized_mode()
        log.info(
            "分段导出: mode=%s source=%s checked=%s frames_loaded=%s",
            _seg_mode, _seg_variant,
            sorted(set(int(i) for i in seg_export.indices)), sorted(frame_by_index),
        )
        if _seg_mode == "continuous":
            # 连续导出: one clip per contiguous run (stitched with the streaming
            # merge), standalone for a checked segment with no checked neighbour —
            # matching exactly what ``run_segment_export`` writes to disk.
            #
            # NOTE: only *timeline-adjacent* checked segments are stitched.
            # Checking #1,#5,#9,#13 yields four standalone clips by design —
            # the run breakdown is logged below so this is never a silent guess.
            from .segment_cache import continuous_export_runs as _runs
            from .segment_cache import merge_run_audio as _merge_audio

            outputs: list[torch.Tensor] = []
            counts: list[int] = []
            run_audios: list[dict] = []
            _all_runs = _runs(frame_by_index.keys())
            log.info("分段导出 连续导出: %d run(s) → %s", len(_all_runs), _all_runs)
            for run in _all_runs:
                run = [i for i in run if i in frame_by_index]
                if not run:
                    continue
                if len(run) == 1:
                    outputs.append(frame_by_index[run[0]])
                    counts.append(int(frame_by_index[run[0]].shape[0]))
                    run_audios.append(audio_by_index.get(run[0]) or {})
                    continue
                run_counts = [int(frame_by_index[i].shape[0]) for i in run]
                try:
                    merged = concat_chunks_lazy(
                        node_id,
                        plan,
                        [seg_by_index[i] for i in run],
                        overrides={i: frame_by_index[i] for i in run},
                        workflow_name=workflow_name,
                        variant=_seg_variant,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning(
                        "分段导出 连续导出: images merge of #%d–#%d failed (%s); "
                        "falling back to standalone clips.",
                        run[0] + 1, run[-1] + 1, exc,
                    )
                    for i in run:
                        outputs.append(frame_by_index[i])
                        counts.append(int(frame_by_index[i].shape[0]))
                        run_audios.append(audio_by_index.get(i) or {})
                    continue
                log.info(
                    "分段导出 连续导出: merged #%d–#%d (%df + %df + … → %df)",
                    run[0] + 1, run[-1] + 1, run_counts[0],
                    run_counts[1] if len(run_counts) > 1 else 0,
                    int(merged.shape[0]),
                )
                outputs.append(merged)
                counts.append(int(merged.shape[0]))
                run_audios.append(
                    _merge_audio(plan, [audio_by_index.get(i) or {} for i in run], run_counts)
                    or {}
                )
            log.info(
                "分段导出 连续导出: %d clip(s) on images output (frame counts %s)",
                len(outputs), counts,
            )
            segment_outputs = outputs
            export_frame_counts = counts
            segment_audios = run_audios
        else:
            ordered = sorted(frame_by_index)
            segment_outputs = [frame_by_index[i] for i in ordered]
            export_frame_counts = [int(t.shape[0]) for t in segment_outputs]
            # Keep audio 1:1 with the frames actually emitted (a segment whose
            # frames could not be loaded must not shift the alignment).
            segment_audios = [audio_by_index.get(i) or {} for i in ordered]
    elif run_list:
        # 「选择运行」in batch mode: export only what actually ran — one clip per
        # contiguous run, stitched exactly like「分段导出」continuous mode. No more
        # splicing unselected slots back from cache / source into a single merged
        # clip that the selection was supposed to avoid.
        from .segment_cache import build_run_selection_clips

        mp4_run_dir = new_segment_mp4_run_dir(plan, for_selection=True)
        seg_outputs, run_audios, seg_counts, run_mp4s = build_run_selection_clips(
            node_id, plan, [seg.index for seg in run_list], segment_outputs, segment_audios,
            all_segments=all_segments, mp4_run_dir=mp4_run_dir,
            workflow_name=workflow_name,
        )
        if not seg_outputs:
            raise ValueError("Batch mode: export list is empty.")
        if run_mp4s:
            reports.append(
                "选择运行导出: " + ", ".join(f"#{p.split('seg_')[-1]}" for p in run_mp4s)
            )
        else:
            reports.append(
                "选择运行导出: "
                + ", ".join(f"#{run[0] + 1}-{run[-1] + 1}" for run in continuous_export_runs([seg.index for seg in run_list]))
            )
        segment_outputs = seg_outputs
        export_frame_counts = seg_counts
        segment_audios = run_audios
        merge_overrides = None
        # The runs are already stitched by build_run_selection_clips above. Bind
        # ``combined`` to the first clip and clear ``export_segments_list`` so the
        # generic merge below is skipped — otherwise the whole timeline was stitched
        # a SECOND time (the duplicated "additive opening luma" seams) and the run
        # paid for two full merges.
        combined = seg_outputs[0]
        export_segments_list = None
        merge_done = True
    elif plan.export_mode == "all" or (
        not run_list
        and seg_export is not None
        and bool(seg_export.enabled)
        and bool(seg_export.indices)
    ):
        # Batch mode only samples「选择运行」—「全部导出」still needs the whole
        # timeline, so unselected slots are filled from cache / source.
        # (An export-only run with every selected segment cached has an empty
        # ``run_list``; we still assemble the full merged clip from cache so the
        # node returns a valid output.)
        # Index them by seg.index: run_list positions are not contiguous, so the
        # old `all_segments[run_list[0].index + i]` mapped to the wrong segment
        # whenever the selection had gaps.
        export_segments_list, segment_audios, export_frame_counts, merge_overrides = (
            _assemble_export_list(
                node_id, plan, all_segments,
                decoded_frames=decoded_frames,
                completed_audios=completed_audios,
                reports=reports,
                workflow_name=workflow_name,
            )
        )
    else:
        export_segments_list = list(decoded_segments)
        export_frame_counts = [int(t.shape[0]) for t in segment_outputs]
        merge_overrides = None

    # Under「全部导出」the node emits the merged clip only — every segment frame
    # is re-read from disk by the merge, so the in-memory chunks are now a second
    # full copy of the same video. Release them *before* allocating the merge
    # result: keeping both is what pushed peak RAM to 2x the final video (the
    # long-timeline OOM).
    if plan.export_mode == "all":
        del segment_outputs[:]
        gc.collect()

    log.info("分段导出: export_mode=%r seg_export_active=%s export_segments_list_len=%d segment_outputs_len=%d",
             plan.export_mode, seg_export_active,
             len(export_segments_list) if export_segments_list else 0,
             len(segment_outputs) if segment_outputs else 0)
    # Streaming merge: one allocation for the result, one copy-in per segment.
    # Every branch must bind ``combined`` — the old ``pass`` on the「全部导出」
    # path left it unbound and killed the node with UnboundLocalError.
    if merge_done:
        # Already stitched (e.g. the「选择运行」runs above). Merging a second time
        # duplicated every seam pass and doubled the merge cost.
        pass
    elif seg_export_active and plan.export_mode == "segments":
        # 分段导出: the split layout emits every clip straight from
        # ``segment_outputs`` and never reads ``combined`` — a second merge would
        # just double peak RAM. Bind it to a real clip (no placeholder) so the
        # slot always carries actual frames.
        if segment_outputs:
            combined = segment_outputs[0]
        else:
            hh = int(getattr(plan, "height", 480) or 480)
            ww = int(getattr(plan, "width", 864) or 864)
            combined = torch.zeros((1, hh, ww, 3), dtype=torch.float32)
        merge_overrides = None
    elif not export_segments_list:
        # 「分段导出」of latent-only segments (no .pt) has nothing to merge here —
        # the per-segment files are still produced by run_segment_export below.
        # Emit a 1-frame placeholder so the node never fails on an empty list.
        if not seg_export_active:
            raise ValueError("Batch mode: export list is empty.")
        hh = int(getattr(plan, "height", 480) or 480)
        ww = int(getattr(plan, "width", 864) or 864)
        combined = torch.zeros((1, hh, ww, 3), dtype=torch.float32)
        merge_overrides = None
    else:
        combined = concat_chunks_lazy(
            node_id, plan, export_segments_list, overrides=merge_overrides,
            workflow_name=workflow_name,
        )
        merge_overrides = None

    # 「分段导出」: the user checked segments + a mode. The VAE is loaded here
    # (and the clip cache was just written during decode), so latent-only
    # segments can be decoded on demand. Best-effort: a failed export only logs.
    seg_export = getattr(plan, "segment_export", None)
    if seg_export is not None:
        log.info(
            "分段导出 config: enabled=%s mode=%s indices=%s",
            seg_export.enabled,
            seg_export.mode,
            seg_export.indices,
        )
    # Node-output-only export: the merged video is already on ``combined`` (see the
    # seg_export_active branch above). We intentionally do NOT write files to disk —
    # the user wants the result directly on the node's OUTPUT slot, same as a normal
    # run. Disk export via run_segment_export is disabled on purpose.
    if seg_export is not None and seg_export.enabled:
        log.info(
            "分段导出 -> 节点输出模式: mode=%s indices=%s (不写磁盘)",
            seg_export.normalized_mode(),
            seg_export.indices,
        )

    # Scratch intermediates (seg_*_scratch_*.pt) live in the same flat node cache dir
    # as the durable segments and are cleaned via _clear_batch_cache / iter_scratch_files,
    # driven by the node's 「清空缓存」button rather than a per-run flag, so they survive
    # failed runs for debugging.

    # Now — and only now — advance the cache generation: this run has just written
    # fresh renders, so the file groups it superseded are expendable. The pre-run
    # sync and every read-only HTTP probe leave them in place, otherwise re-wording
    # a prompt would delete the last render before a replacement exists.
    try:
        sync_segment_slots(node_id, plan, workflow_name=workflow_name, gc=True)
    except Exception as exc:  # pragma: no cover - GC is best-effort
        log.warning("Segment cache cleanup after batch run skipped (%s).", exc)

    # Drop per-run scratch intermediates (seg_*_scratch_*.pt, e.g. the position-
    # numbered seg_0000_scratch_ref.pt) now that the run finished cleanly. These
    # are regenerate-in-place working state — NOT the content-hash durable files
    # (seg_<hash>_*.pt) that「分段导出」/ motion-context read back — so keeping
    # them around only duplicates data on disk and clutters the cache dir. Failures
    # above skip this block (via the early raises) and leave scratch intact for
    # debugging, which is the intended behaviour.
    try:
        _clear_batch_cache(cache_dir, reports)
    except Exception as exc:  # pragma: no cover - cleanup is best-effort
        log.warning("Batch scratch cleanup skipped (%s).", exc)

    return (
        combined,
        segment_outputs,
        segment_audios,
        "\n".join(reports),
        export_frame_counts,
    )
