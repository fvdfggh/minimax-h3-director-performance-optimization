"""Per-segment bodies of the three batch phases.

execute_director_batch is the orchestrator: it resolves modes, builds the run list
and owns the accumulators. The functions here are the per-segment work of each
phase, moved out verbatim so the orchestrator stays readable. Mutable containers
are passed in and updated in place, which is why only _prepare_one_segment returns
anything (what Phase 1 staged for this segment).
"""

from __future__ import annotations
import logging

import torch
log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.batch")

from .audio_export import AUDIO_MODE_MUTE, AUDIO_MODE_SOURCE
from .batch_source_audio import align_pcm_to_frames
from .batch_helpers import _av_latent_canvas_matches, _build_minimax_inputs, _decode_av_latent, _latent_for_cache, _load_batch_conditioning, _load_batch_latent, _load_batch_ref, _prev_context_available, _save_batch_conditioning, _save_batch_latent, _save_batch_ref, _trim_decoded_to_export
from .batch_prepare import _rebuild_empty_latent, prepare_segment_materials
from .cache_paths import slot_content_hash
from .cache_readback import load_segment_audio
from .cache_store import load_next_segment_av_latent, load_segment_av_latent, load_segment_handoff_meta, save_segment_cache
from .conditioning_keys import text_cache_key
from .conditioning_params import save_segment_second_params
from .conditioning_store import load_conditioning_cache
from .continuity_settings import is_continuity_active, resolve_prev_segment_output
from .core_sampling import sample_single_stage
from .frame_align import minimax_align_frame_count
from .h3_latent_continue import apply_latent_continue
from .h3_motion_context import DEFAULT_AUDIO_CONTEXT_FRAMES, TAIL_CONTEXT_FRAMES, apply_motion_context, generation_frame_budget, handoff_end_frame, resolve_tail_context_length, snap_context_frames, trim_export_tail
from .plan_prompts import reinforce_r2v_prompt, reinforce_rv2v_prompt, reinforce_v2v_prompt
from .plan_ranges import prepare_segment_clip
from .plan_types import DirectorPlan, resolve_ref_image_size
from .progress import report_director_progress, report_director_segment_preview
from .segment_mp4_export import maybe_export_segment_mp4
from .segment_runtime import frames_label, resolve_segment_raw_clip
from ..lib.image_prep import assert_minimax_canvas, fit_canvas, fit_video_long_edge
from ..lib.media_b64 import tensor_frame_to_jpeg_b64

#: Report wording for material dropped by the strict prompt filter
#: (see ``batch_prepare.prepare_segment_materials`` -> ``ref_drops``).
_IGNORED_REF_LABEL = {
    "Picture": "参考图",
    "Video": "参考视频",
    "Audio": "参考音频",
    "VideoAudio": "参考视频音轨",
}

def _prepare_one_segment(
    plan: DirectorPlan,
    seg,
    seg_pos: int,
    *,
    node_id: int,
    all_segments,
    run_indices: set,
    completed_av_latents: dict,
    cache_dir,
    workflow_name: str | None,
    use_conditioning_cache: bool,
    audio_vae,
    timeline_seg_total: int,
    segment_total: int,
    reports: list,
) -> dict:
    """Per-segment body of the Phase 1 loop.

    Stages this segment's pixels and conditioning and writes its cache entries.
    Returns ``{"text_key", "cache_hit", "staged", "meta"}``; on a conditioning
    cache hit there is nothing left to encode, so ``staged``/``meta`` are None and
    the caller only counts the hit.
    """
    ui_idx = seg.timeline_index
    report_director_progress(
        node_id, segment_index=seg_pos, segment_total=segment_total,
        phase="batch_prepare", phase_value=0, phase_max=1,
        frames_label=frames_label(seg), task_key=seg.task_key,
        timeline_segment_index=ui_idx, timeline_segment_total=timeline_seg_total,
    )
    target_len = max(1, int(seg.frame_count or plan.total_frames or 124))
    raw_clip = resolve_segment_raw_clip(plan, seg)
    if seg.source_clip is not None:
        body_raw = seg.source_clip
        target_len = max(target_len, int(body_raw.shape[0]))
    else:
        body_raw = raw_clip[:target_len] if int(raw_clip.shape[0]) > target_len else raw_clip
    if body_raw is not None and body_raw.shape[0] > 0:
        if plan.output_mode == "fixed":
            clip_frames = fit_canvas(body_raw, plan.width, plan.height)
        else:
            clip_frames = fit_video_long_edge(body_raw, plan.ref_max_size)
            if int(clip_frames.shape[1]) != int(plan.height) or int(clip_frames.shape[2]) != int(plan.width):
                clip_frames = fit_canvas(clip_frames, plan.width, plan.height)
    else:
        clip_frames = None
    num_frames = minimax_align_frame_count(target_len)
    if clip_frames is not None:
        clip_frames, _ = prepare_segment_clip(clip_frames, num_frames)
    continuity_active = is_continuity_active(plan, seg)
    ctx_w = int(plan.width)
    ctx_h = int(plan.height)
    if clip_frames is not None and clip_frames.shape[0] > 0:
        ctx_h, ctx_w = int(clip_frames.shape[1]), int(clip_frames.shape[2])
    assert_minimax_canvas(ctx_w, ctx_h)
    positive_prompt = seg.prompt
    if seg.task_key == "fl2v":
        from .fl2v_timeline import reinforce_fl2v_prompt
        has_start = any(getattr(r, "index", None) == 0 for r in (seg.refs or []))
        has_end = any(getattr(r, "index", None) == 1 for r in (seg.refs or []))
        if not has_start and not has_end and seg.refs:
            has_start = True
            has_end = len(seg.refs) >= 2
        positive_prompt = reinforce_fl2v_prompt(positive_prompt, has_end_frame=has_end, has_start_frame=has_start)
    elif seg.task_key == "r2v":
        # Reference tags are the user's alone: nothing is auto-added here, so the
        # reference filter can drop every un-referenced picture / video / audio.
        positive_prompt = reinforce_r2v_prompt(positive_prompt)
    elif seg.task_key == "v2v":
        positive_prompt = reinforce_v2v_prompt(positive_prompt)
    elif seg.task_key == "rv2v":
        positive_prompt = reinforce_rv2v_prompt(positive_prompt)
    # Build inputs (without prev_tail — we don't have it yet in phase 1)
    first_frame, last_frame, ref_images, ref_videos, ref_audios, ref_video_audios = _build_minimax_inputs(
        plan, seg, clip_frames=clip_frames, ctx_w=ctx_w, ctx_h=ctx_h, prev_tail=None,
    )
    i2v_new_anchor = seg.task_key == "i2v" and first_frame is not None
    # Probe availability here, not in Phase 2: sample_len is baked into the
    # conditioning below, so the pin decision cannot be revised later.
    prev_context_available = _prev_context_available(
        node_id, plan, all_segments, seg.index, run_indices, completed_av_latents,
        width=ctx_w, height=ctx_h, workflow_name=workflow_name,
    )
    use_motion_context = (
        continuity_active
        and not i2v_new_anchor
        and seg.index > 0
        and prev_context_available
    )
    if use_motion_context:
        first_frame = None  # context owns the head
    if continuity_active and seg.index > 0 and not use_motion_context:
        reports.append(
            f"  Seg #{seg.index + 1}: 无上段可引用（前段未采样且无缓存），"
            "本段不继承运动——请按顺序重跑该段以恢复接缝连贯。"
        )
    # The head pin *replays* the previous segment's tail, so it lengthens
    # the sample and is trimmed after decode. The tail pin needs its room
    # reserved up front: it is replayed past the export and dropped, and
    # taking that room out of the free zone instead is what used to clip a
    # segment's ending.
    context_n = (
        snap_context_frames(plan.continuity_overlap_frames) if use_motion_context else 0
    )
    next_pin = bool(getattr(seg, "continuity_to_next", False))
    tail_room = int(TAIL_CONTEXT_FRAMES) if (use_motion_context and next_pin) else 0
    role = (
        "both" if (context_n > 0 and next_pin) else
        "prev" if context_n > 0 else
        "next" if next_pin else
        "none"
    )
    # sample_len only here; export length is recomputed in Phase 2 from the
    # same role so the three-zone budget stays the single source of truth.
    sample_len, _, _, _ = generation_frame_budget(
        num_frames, context_n, role, tail_room,
    )
    if seg.task_key in {"r2v", "v2v", "rv2v"} and (ref_images or ref_videos or ref_audios or ref_video_audios) and audio_vae is None:
        raise ValueError("r2v/v2v/rv2v requires audio_vae input.")
    # Text-side identity of this segment. Segments sharing a key produce
    # interchangeable encodings, so the expensive Qwen prefill runs once per
    # distinct key rather than once per segment.
    ref_image_size = resolve_ref_image_size(seg, plan)
    # The canvas only enters the key when this segment actually feeds pixels to
    # the text encoder (refs / videos / first-last frames); the sample length
    # never does, so a duration tweak reuses the encoding instead of paying for
    # another Qwen prefill.
    text_key = text_cache_key(
        positive_prompt, ctx_w, ctx_h, sample_len, seg.task_key,
        ref_image_size, ref_images,
        ref_videos=ref_videos, first_frame=first_frame, last_frame=last_frame,
    )
    # Remember this segment's first-pass text/context identity on disk so a
    # later「二次采样」can load the *same* conditioning file without re-hashing
    # refs (pixel tensors that no longer exist once this run ends).
    # Keyed by content hash, not by position: a later timeline reorder must
    # not let one segment pick up another's encoding.
    save_segment_second_params(
        node_id=node_id,
        workflow_name=workflow_name,
        segment_index=seg.index,
        slot_key=slot_content_hash(seg, plan),
        text_key=text_key,
        ctx_w=ctx_w,
        ctx_h=ctx_h,
        sample_len=sample_len,
        num_frames=num_frames,
        frame_count=getattr(seg, "frame_count", 0) or num_frames,
        context_n=context_n,
        task_key=seg.task_key,
        ref_image_size=ref_image_size,
        positive_prompt=positive_prompt,
    )
    # Try conditioning cache first
    cached_conditioning = None
    if use_conditioning_cache:
        cached_conditioning = load_conditioning_cache(
            node_id=node_id, segment_index=seg.index, prompt=positive_prompt,
            width=ctx_w, height=ctx_h, length=sample_len, task_key=seg.task_key,
            ref_image_size=ref_image_size, ref_images=ref_images,
            workflow_name=workflow_name,
            ref_videos=ref_videos, first_frame=first_frame, last_frame=last_frame,
            # Two-level hit: a canvas we have never encoded at still has a usable
            # text encoding, so take it and leave the latents to the VAEs.
            require_latents=False,
        )
    # Ref data feeds Phase 2; write it before any cache short-circuit.
    ref_data = {
        "first_frame": first_frame,
        "last_frame": last_frame,
        "ref_images": ref_images,
        "ref_videos": ref_videos,
        "ref_audios": ref_audios,
        "ref_video_audios": ref_video_audios,
        "clip_frames": clip_frames,
        "num_frames": num_frames,
        "target_len": target_len,
        "sample_len": sample_len,
        "ctx_w": ctx_w,
        "ctx_h": ctx_h,
        "positive_prompt": positive_prompt,
        "context_n": context_n,
        "use_motion_context": use_motion_context,
    }
    _save_batch_ref(node_id, seg.index, ref_data, cache_dir)
    if cached_conditioning is not None and not cached_conditioning.get("latents_pending"):
        # Full hit: text and latents, so this segment never touches a model.
        _save_batch_conditioning(
            node_id, seg.index,
            cached_conditioning["positive"], cached_conditioning["negative"],
            cache_dir,
        )
        reports.append(f"  Seg #{seg.index + 1}: conditioning CACHE HIT")
        return {"text_key": text_key, "cache_hit": True, "staged": None, "meta": None}
    # Stage the pixels now; the encoders consume them in the grouped passes.
    staged = prepare_segment_materials(
        prompt=positive_prompt, width=ctx_w, height=ctx_h,
        length=sample_len, task_key=seg.task_key,
        first_frame=first_frame, last_frame=last_frame,
        ref_images=ref_images, ref_videos=ref_videos,
        ref_video_audios=ref_video_audios, ref_audios=ref_audios,
        ref_image_size=ref_image_size,
    )
    staged["text_key"] = text_key
    if cached_conditioning is not None:
        # Text half was cached but the latents were never encoded at this canvas.
        # Carry the encoding over so the Qwen prefill is skipped and only the
        # reference VAEs run.
        staged["cond"] = cached_conditioning["positive"]
        staged["text_reused"] = True
        reports.append(f"  Seg #{seg.index + 1}: text CACHE HIT, latents to encode")
    # Surface the strict prompt filter: material the prompt never referenced is
    # not sent, and the user has to be able to see that happen.
    for drop in staged.get("ref_drops") or []:
        reports.append(
            f"  Seg #{seg.index + 1}: 提示词未引用 {_IGNORED_REF_LABEL.get(drop['token'], drop['token'])}"
            f" ×{drop['dropped']}，已忽略（实际使用 {drop['kept']} 个）"
        )
    meta = {
        "seg": seg, "positive_prompt": positive_prompt,
        "ctx_w": ctx_w, "ctx_h": ctx_h, "sample_len": sample_len,
        "ref_images": ref_images, "ref_image_size": ref_image_size,
        "ref_videos": ref_videos, "first_frame": first_frame, "last_frame": last_frame,
        "text_key": text_key,
    }

    return {"text_key": text_key, "cache_hit": False, "staged": staged, "meta": meta}

def _sample_one_segment(
    _enable_tail,
    _seg_pass_list,
    all_segments,
    audio_mode,
    audio_vae,
    cache_dir,
    cfg,
    clear_vram_between_segments,
    completed_audios,
    completed_av_handoff,
    completed_av_latents,
    live_tae_preview,
    model,
    node_id,
    pending_prev_trim,
    plan,
    reports,
    run_list,
    sampler,
    scheduler,
    seed,
    seg,
    shift_audio,
    shift_video,
    sigmas,
    steps,
    timeline_seg_total,
    vae,
    workflow_name,
    retain_audio_cache=None,
) -> None:
    """Per-segment body of a %s loop in execute_director_batch.

    Moved verbatim (re-indented only); mutable containers passed in are
    updated in place, which is why it returns nothing.
    """
    ui_idx = seg.timeline_index
    seg_pos = run_list.index(seg)

    report_director_progress(
        node_id, segment_index=seg_pos, segment_total=len(run_list),
        phase="batch_sample", phase_value=0, phase_max=1,
        frames_label=frames_label(seg), task_key=seg.task_key,
        timeline_segment_index=ui_idx, timeline_segment_total=timeline_seg_total,
    )

    # Load ref data from disk first — it carries the canvas (ctx_w/ctx_h/
    # sample_len) that the zero AV latent is rebuilt from.
    ref_data = _load_batch_ref(node_id, seg.index, cache_dir)
    if ref_data is None:
        raise RuntimeError(f"Batch mode: ref cache miss for segment {seg.index}")

    sample_len = ref_data["sample_len"]
    num_frames = ref_data["num_frames"]
    target_len = ref_data["target_len"]
    ctx_w = ref_data["ctx_w"]
    ctx_h = ref_data["ctx_h"]
    use_motion_context = ref_data["use_motion_context"]
    context_n = ref_data["context_n"]

    # Continuity role + export length — single source of truth for Phase 2.
    next_pin = bool(getattr(seg, "continuity_to_next", False))
    role = (
        "both" if (context_n > 0 and next_pin) else
        "prev" if context_n > 0 else
        "next" if next_pin else
        "none"
    )
    _tail_room = int(TAIL_CONTEXT_FRAMES) if (context_n > 0 and role in ("next", "both")) else 0
    _, _tf, _tb, export_len = generation_frame_budget(num_frames, context_n, role, _tail_room)

    # Load pre-encoded conditioning from disk
    cond_data = _load_batch_conditioning(node_id, seg.index, cache_dir)
    if cond_data is None:
        raise RuntimeError(f"Batch mode: conditioning cache miss for segment {seg.index}")
    positive = cond_data["positive"]
    negative = cond_data["negative"]
    latent = cond_data.get("latent")
    if latent is None:
        # New-style scratch: the all-zero canvas was never written.
        # (ctx_w, ctx_h, sample_len) are exactly what prepare_segment_materials
        # used to build the original, so this is identical.
        latent = _rebuild_empty_latent(ctx_w, ctx_h, sample_len)
    del cond_data

    # Load previous segment's tail for motion context
    trim_frames = 0
    if use_motion_context and seg.index > 0:
        prev_idx = seg.index - 1
        # Motion context always reads the *timeline* predecessor (seg.index - 1).
        # Lookup order: this run's memory -> seg_cache av latent -> durable segment
        # cache left by an earlier run. The last hop keeps a partial「选择运行」
        # working even when the predecessor was never re-sampled this run.
        prev_av = completed_av_latents.get(prev_idx)
        if prev_av is None:
            prev_seg = next((s for s in all_segments if s.index == prev_idx), None)
            if prev_seg is not None:
                prev_av = load_segment_av_latent(node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name)
        if prev_av is None:
            prev_av = _load_batch_latent(node_id, prev_idx, cache_dir)
            if prev_av is not None:
                log.info(
                    "Director batch: seg #%d uses seg #%d latent from the durable segment cache; "
                    "if its prompt changed, re-run that segment to refresh.",
                    seg.index + 1, prev_idx + 1,
                )
        if prev_av is not None:
            completed_av_latents[prev_idx] = prev_av

        # Pixel tail is a fallback, not a prerequisite: apply_motion_context
        # prefers context_latent and then never touches context_frames.
        # Resolving it eagerly made a missing *frame* cache fatal even though
        # the latent was sitting right there — resolve_prev_segment_output
        # raises whenever the predecessor is unavailable and continuity is on.
        # Only needed when the latent is missing, or when its canvas does not
        # match and the pin would fall through to the pixel path anyway.
        prev_tail = None
        if prev_av is None or not _av_latent_canvas_matches(prev_av, ctx_w, ctx_h):
            # Passed an empty dict on purpose: Phase 2 runs entirely before Phase 3,
            # so no segment is decoded to pixels yet — the predecessor's tail can
            # only come from the segment cache on disk.
            prev_tail = resolve_prev_segment_output(
                plan, all_segments, seg.index, {}, node_id, workflow_name=workflow_name
            )

        prev_audio = completed_audios.get(prev_idx)
        if prev_audio is None:
            prev_seg = next((s for s in all_segments if s.index == prev_idx), None)
            if prev_seg is not None:
                prev_audio = load_segment_audio(node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name)
                if prev_audio is not None:
                    completed_audios[prev_idx] = prev_audio

        prev_handoff = completed_av_handoff.get(prev_idx)
        if prev_handoff is None:
            prev_seg = next((s for s in all_segments if s.index == prev_idx), None)
            if prev_seg is not None:
                prev_handoff = load_segment_handoff_meta(node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name)
                if prev_handoff is not None:
                    completed_av_handoff[prev_idx] = prev_handoff

        prev_end_frame = None
        if prev_handoff:
            prev_end_frame = handoff_end_frame(
                trim_frames=int(prev_handoff.get("trim_frames") or 0),
                export_frames=int(prev_handoff.get("export_frames") or 0),
            )
            sample_f = int(prev_handoff.get("sample_frames") or 0)
            if sample_f > 0 and prev_end_frame >= sample_f:
                prev_end_frame = None

        pin_audio = audio_mode != AUDIO_MODE_MUTE and (prev_av is not None or prev_audio is not None)
        # ------------------------------------------------------------------
        # 对齐下段 (align-to-next): pin the next segment's opening into this
        # segment's tail. Cache-driven middle-out mode — only runs when that
        # neighbour already holds an AV latent, otherwise it silently no-ops.
        # ------------------------------------------------------------------
        tail_context_latent = None
        tail_context_length = 0
        tail_context_offset = 0
        if _enable_tail and bool(getattr(seg, "continuity_to_next", False)):
            tail_n = resolve_tail_context_length(
                latent, export_len, context_n=context_n
            )
            if tail_n > 0:
                next_latent = load_next_segment_av_latent(node_id, seg.index, workflow_name=workflow_name)
                if next_latent is None:
                    log.info(
                        "Seg #%d: 对齐下段 skipped — next segment has no cached AV latent.",
                        seg.index + 1,
                    )
                else:
                    tail_context_latent = next_latent
                    tail_context_length = tail_n
                    # Read the next segment's trimmed opening: skip its own head
                    # pin so we reference what it actually exports, per the
                    # "先裁切完才可被用于下一段进行参照" rule.
                    tail_context_offset = int(
                        completed_av_handoff.get(seg.index + 1, {}).get("trim_frames", 0) or 0
                    )
                    log.info(
                        "Seg #%d: 对齐下段 active — pinning next segment head %df (offset %df).",
                        seg.index + 1,
                        tail_n,
                        tail_context_offset,
                    )
        # 段间连续性：重绘幅度 > 0 时走「锥形重绘」(continue 模式)——body 前缀被重绘，
        # 解码后同样要裁掉（前缀 = 上一段尾巴的重放，不是本段内容）；否则保留原参考帧
        # 引导 (guide)。二者不叠加（上游语义）。
        trim_frames = 0
        prev_export_trim = 0
        if plan.continuity_redraw > 0:
            try:
                latent, _c_span, _c_trim = apply_latent_continue(
                    latent, prev_av=prev_av, prev_tail=prev_tail, vae=vae,
                    context_length=context_n or int(plan.continuity_overlap_frames),
                    context_end_frame=prev_end_frame,
                    pin_audio=pin_audio, context_audio=prev_audio, audio_vae=audio_vae,
                    audio_context_length=DEFAULT_AUDIO_CONTEXT_FRAMES,
                    seam_min_mask=float(plan.continuity_redraw),
                )
                # 第二个返回值就是「写入的前缀帧数」，与 guide 路径 apply_motion_context
                # 的 trim_frames 同义：it 是解码后要丢掉的头部。写 0 会让上一段尾巴
                # （重绘后仍近似保留）留在成片开头 → 接缝画面重复 + 时长/相位错位，
                # 下一段的 prev_end_frame 也会短 span 帧。
                trim_frames = int(_c_span)
                prev_export_trim = int(_c_trim)
                reports.append(
                    f"  Seg #{seg.index + 1}: 锥形重绘 ON — 前缀 {_c_span}f "
                    f"(重绘幅度 {plan.continuity_redraw:.2f})"
                )
            except Exception as exc:
                log.warning(
                    "Director batch: seg #%d 锥形重绘失败，回退为参考帧引导 (%s)。",
                    seg.index + 1, exc,
                )
                positive, trim_frames, prev_export_trim = apply_motion_context(
                    positive, latent, vae=vae,
                    context_length=context_n,
                    context_latent=prev_av,
                    context_frames=prev_tail,
                    context_audio=prev_audio,
                    audio_vae=audio_vae,
                    continue_audio=pin_audio,
                    keep_existing_keyframes=(seg.task_key == "fl2v"),
                    context_end_frame=prev_end_frame,
                    audio_context_length=DEFAULT_AUDIO_CONTEXT_FRAMES,
                    tail_context_latent=tail_context_latent,
                    tail_context_offset=tail_context_offset,
                    tail_context_length=tail_context_length,
                    seed=int(seed),
                )
        else:
            positive, trim_frames, prev_export_trim = apply_motion_context(
                positive, latent, vae=vae,
                context_length=context_n,
                context_latent=prev_av,
                context_frames=prev_tail,
                context_audio=prev_audio,
                audio_vae=audio_vae,
                continue_audio=pin_audio,
                keep_existing_keyframes=(seg.task_key == "fl2v"),
                context_end_frame=prev_end_frame,
                audio_context_length=DEFAULT_AUDIO_CONTEXT_FRAMES,
                tail_context_latent=tail_context_latent,
                tail_context_offset=tail_context_offset,
                tail_context_length=tail_context_length,
                seed=int(seed),
            )
        if prev_export_trim > 0:
            pending_prev_trim[seg.index] = int(prev_export_trim)
        reports.append(
            f"  Seg #{seg.index + 1}: motion context ON — pin {trim_frames}f from seg #{seg.index}"
            + (f", prev tail trim {prev_export_trim}f" if prev_export_trim else "")
        )

    # UNet sampling
    def _report_sample_phase(phase, value):
        report_director_progress(
            node_id, segment_index=seg_pos, segment_total=len(run_list),
            phase=phase, phase_value=value, phase_max=1,
            frames_label=frames_label(seg), task_key=seg.task_key,
            timeline_segment_index=ui_idx, timeline_segment_total=timeline_seg_total,
        )

    def _report_step_preview(step: int, total_steps: int, x0) -> None:
        # Live frame for the batch-card preview slot.
        try:
            from ..lib.media_b64 import pil_to_jpeg_b64
            from .tae_preview import x0_to_preview_pil
            pil = x0_to_preview_pil(x0, max_side=512)
            if pil is None:
                return
            report_director_segment_preview(
                node_id,
                segment_index=ui_idx,
                image_b64=pil_to_jpeg_b64(pil),
                width=pil.width,
                height=pil.height,
                live=True,
                step=step + 1,
                total_steps=total_steps,
            )
        except Exception as exc:
            log.debug("Live TAE preview skipped: %s", exc)

    # 保留音频: overwrite the whole audio stream with the retained clip and lock
    # it (audio noise_mask = 0) so the UNet conditions on it without re-drawing
    # it. Runs AFTER continuity on purpose —「段间引导」only pins the audio head,
    # and a retained clip must win over that everywhere.
    retain_active = False
    retain_entry = (retain_audio_cache or {}).get(seg.index)
    if retain_entry is not None:
        _fps = float(getattr(plan, "frame_rate", 24) or 24)
        try:
            from .audio_retain import apply_retain_audio

            latent = apply_retain_audio(
                latent, audio_vae, retain_entry.get("pcm"),
                sample_len=sample_len, fps=_fps,
                # 头部 pin 会被裁掉，音频要从导出帧 0 开始，否则成片音轨比模型
                # 听到的内容提前 trim_frames 帧。
                head_seconds=float(trim_frames or 0) / _fps,
            )
            retain_active = True
            reports.append(
                f"  Seg #{seg.index + 1}: 保留音频 ON — 锁定提取音频 "
                f"({retain_entry.get('entry_id') or '?'})"
            )
        except Exception as exc:
            log.warning(
                "Director batch: seg #%d 保留音频失败，按正常生成处理 (%s)。",
                seg.index + 1, exc,
            )
            retain_active = False
            # 注入没成 → 这一段就是普通生成段。把缓存条目摘掉，Phase 3 才会正常
            # 解码音频，而不是拿一条模型根本没听过的 PCM 直接合成。
            try:
                retain_audio_cache.pop(seg.index, None)
            except Exception:  # pragma: no cover - defensive
                pass

    samples = sample_single_stage(
        model=model, positive=positive, negative=negative,
        latent=latent, seed=seed, cfg=cfg, steps=steps,
        sampler_name=sampler, scheduler=scheduler,
        shift_video=shift_video, shift_audio=shift_audio,
        on_phase=_report_sample_phase,
        on_step_preview=_report_step_preview if live_tae_preview else None,
        preview_every=1,
        sigmas=sigmas,
    )

    if retain_active:
        # The lock mask is sampling-only bookkeeping; keeping it would write a
        # stale mask into the cached latent the next segment reads.
        samples.pop("noise_mask", None)

    # Save AV latent to disk
    _save_batch_latent(node_id, seg.index, samples, cache_dir)

    # Build handoff for next segment.
    # export_frames is the segment's own clean middle length: 17k for any
    # referenced segment (the +5 VAE phase is carried by the head reference),
    # 17k+5 for a standalone one. The next segment's pin end limit is
    # (trim + export), which is exactly where this segment's last exported
    # frame sits.
    handoff = {
        "trim_frames": int(trim_frames),
        "export_frames": int(export_len),
        "sample_frames": int(sample_len),
        "official_mc_length": False,
    }
    completed_av_latents[seg.index] = samples
    completed_av_handoff[seg.index] = handoff

    # Free conditioning from GPU (UNet stays!)
    del positive, negative, latent, samples
    if clear_vram_between_segments and torch.cuda.is_available():
        torch.cuda.empty_cache()

    reports.append(
        f"  Seg #{seg.index + 1}: sampled {sample_len}f → latent saved to disk"
    )

def _decode_export_one_segment(
    audio_vae,
    audio_mode,
    cache_dir,
    completed_audios,
    completed_av_handoff,
    completed_av_latents,
    decode_audio,
    decoded_frames,
    decoded_segments,
    export_frame_counts,
    last_timeline_index,
    mp4_run_dir,
    node_id,
    pending_prev_trim,
    plan,
    reports,
    run_list,
    run_pos_map,
    seg,
    segment_audios,
    segment_outputs,
    source_audio_cache,
    timeline_seg_total,
    vae,
    workflow_name,
    retain_audio_cache=None,
) -> None:
    """Per-segment body of a %s loop in execute_director_batch.

    Moved verbatim (re-indented only); mutable containers passed in are
    updated in place, which is why it returns nothing.
    """
    ui_idx = seg.timeline_index
    seg_pos = run_list.index(seg)

    report_director_progress(
        node_id, segment_index=seg_pos, segment_total=len(run_list),
        phase="batch_decode", phase_value=0, phase_max=1,
        frames_label=frames_label(seg), task_key=seg.task_key,
        timeline_segment_index=ui_idx, timeline_segment_total=timeline_seg_total,
    )

    # Use AV latent from Phase 2 (in memory) instead of reloading from disk
    samples = completed_av_latents.get(seg.index)
    if samples is None:
        # Fallback: load from disk. Back-fill it too — save_segment_cache below
        # writes the latent into seg_cache, and the next segment pins from
        # there rather than from this dict.
        samples = _load_batch_latent(node_id, seg.index, cache_dir)
        if samples is not None:
            completed_av_latents[seg.index] = samples
    if samples is None:
        raise RuntimeError(f"Batch mode: latent cache miss for segment {seg.index}")

    # No manual device move here: ``samples["samples"]`` is a
    # comfy.nested_tensor.NestedTensor, which is NOT a torch.Tensor subclass,
    # so an isinstance guard silently skipped it. ComfyUI's VAEDecode already
    # places inputs on the right device internally.

    # Load ref data for trim/export info
    ref_data = _load_batch_ref(node_id, seg.index, cache_dir)
    num_frames = ref_data["num_frames"]
    target_len = ref_data["target_len"]
    context_n = ref_data["context_n"]
    use_motion_context = ref_data["use_motion_context"]
    del ref_data

    # Determine trim_frames / export_len from the Phase 2 handoff: that is
    # the same pair the next segment uses to place its pin window, so the two
    # must never disagree. Recompute only when the handoff is missing.
    handoff = completed_av_handoff.get(seg.index, {})
    trim_frames = int(handoff.get("trim_frames") or 0)
    export_len = int(handoff.get("export_frames") or 0)
    if export_len <= 0:
        # num_frames is already on the 17k+5 grid and continuity no longer
        # cuts the export down to 17k.
        export_len = int(num_frames)

    # VAE decode
    # Two "the soundtrack already exists" cases reuse their PCM and skip the
    # audio VAE decode entirely: source mode (source video's track, extracted in
    # Phase 1) and「保留音频」(the clip the user pinned in the 音频 tab). The
    # latter wins — it is the explicit per-segment choice.
    source_pcm = (retain_audio_cache or {}).get(seg.index)
    if source_pcm is None and audio_mode == AUDIO_MODE_SOURCE:
        source_pcm = source_audio_cache.get(seg.index)
    if source_pcm is not None:
        decoded, _ = _decode_av_latent(samples, vae, audio_vae, decode_audio=False)
        del samples
        # Same video trim as the generate path, then align the source PCM to the
        # exported frame count. The PCM is anchored at seg.start_frame, so it is
        # only tail-trimmed — never head-trimmed like decoded model audio, whose
        # head belongs to the previous segment's pin.
        decoded, _ = _trim_decoded_to_export(
            decoded, None, trim_frames=trim_frames, export_len=export_len, plan=plan,
        )
        audio_dict = align_pcm_to_frames(
            source_pcm["pcm"], int(decoded.shape[0]), float(plan.frame_rate or 24),
        )
        log.debug(
            "Seg #%d: source audio from Phase 1 cache (%.2fs for %df)",
            seg.index + 1,
            audio_dict["waveform"].shape[-1] / max(1, int(audio_dict["sample_rate"])),
            int(decoded.shape[0]),
        )
    else:
        # Generate mode (or a source segment whose extraction failed): decode
        # audio from the AV latent exactly as before.
        decoded, audio_dict = _decode_av_latent(samples, vae, audio_vae, decode_audio=decode_audio)
        del samples
        decoded, audio_dict = _trim_decoded_to_export(
            decoded, audio_dict, trim_frames=trim_frames, export_len=export_len, plan=plan,
        )

    chunk = decoded.cpu().float()

    # The head/tail window and the segment's clip.mp4 are written together by
    # save_segment_cache, so a piecewise export is byte-equivalent to this
    # trimmed chunk — no re-decode, no re-trim, and no chance of the two
    # files describing different renders.
    save_segment_cache(
        node_id, seg, plan, chunk,
        av_latent=_latent_for_cache(node_id, seg.index, completed_av_latents, cache_dir),
        handoff=handoff,
        audio=audio_dict if isinstance(audio_dict, dict) else None,
        workflow_name=workflow_name,
    )

    # Export mp4
    if mp4_run_dir is not None:
        mp4_path = maybe_export_segment_mp4(
            mp4_run_dir, plan, seg, chunk,
            audio_dict if isinstance(audio_dict, dict) else None,
        )
        if mp4_path:
            reports.append(f"  Segment {ui_idx + 1}/{timeline_seg_total}: mp4 → {mp4_path}")

    # Preview
    if seg.task_key in {"t2v", "i2v", "r2v", "fl2v", "v2v", "rv2v"} and chunk.shape[0] >= 1:
        try:
            frames_b64 = [tensor_frame_to_jpeg_b64(chunk[i]) for i in range(min(int(chunk.shape[0]), 10))]
            h, w = int(chunk.shape[1]), int(chunk.shape[2])
            report_director_segment_preview(
                node_id, segment_index=ui_idx,
                image_b64=frames_b64[0], width=w, height=h,
                frames=frames_b64, fps=float(plan.frame_rate or 24),
            )
        except Exception as exc:
            log.debug("Segment preview skipped: %s", exc)

    run_pos_map[seg.index] = len(segment_outputs)
    segment_outputs.append(chunk)
    audio_out = audio_dict if isinstance(audio_dict, dict) else {}
    segment_audios.append(audio_out)
    decoded_segments.append(seg)
    decoded_frames[seg.index] = int(chunk.shape[0])
    completed_audios[seg.index] = audio_out
    export_frame_counts.append(int(chunk.shape[0]))

    # Phase-align orphaned tail: drop gap_after_pin frames from the *previous*
    # export so this segment's pin window abuts it exactly. Skipping this
    # replays the gap at every seam (visible stutter) and leaves the merge
    # longer by (seams x gap).
    # Now that export == sample and trim == 0, the pin window ends on the
    # previous export's last frame, so gap_after_pin is 0 and this block is
    # dormant. It stays as a safety net for hand-offs from older caches,
    # whose stored trim/export pair predates this layout.
    # ``pending_prev_trim[seg.index]`` must be applied to timeline segment
    # ``seg.index - 1``.  ``prev_export_seg`` is merely the previous *iterated*
    # segment, so with a sparse「选择运行」(e.g. 0,3,7) it would shave segment
    # 0's tail to pay for segment 3's gap.  Only trim when the predecessor was
    # actually decoded in this run.
    pending_trim = int(pending_prev_trim.get(seg.index) or 0)
    if (
        pending_trim > 0
        and prev_export_seg is not None
        and prev_export_seg.index != seg.index - 1
    ):
        log.warning(
            "Director batch: skip %df phase-align trim for seg #%d — "
            "timeline predecessor #%d was not decoded in this run "
            "(its cache is untrimmed, so this seam keeps the gap).",
            pending_trim, seg.index + 1, seg.index,
        )
        pending_trim = 0
    # Never trim a segment with no successor on the timeline. With no following
    # segment to pin it, those trailing frames are genuine content rather than
    # a handoff gap, so dropping them would be a real loss. The final segment's
    # AV latent is still persisted above, so a later run can append a segment
    # after it and pin normally.
    if (
        pending_trim > 0
        and prev_export_seg is not None
        and last_timeline_index is not None
        and prev_export_seg.index >= last_timeline_index
    ):
        log.warning(
            "Director batch: skip %df phase-align trim on seg #%d — it is the "
            "last timeline segment, so its tail is content, not a gap.",
            pending_trim, prev_export_seg.index + 1,
        )
        pending_trim = 0
    if pending_trim > 0 and prev_export_seg is not None and prev_export_chunk is not None:
        fps = float(plan.frame_rate or 24)
        if int(prev_export_chunk.shape[0]) > pending_trim:
            new_chunk, new_audio = trim_export_tail(
                prev_export_chunk, prev_export_audio, pending_trim, fps=fps
            )
            prev_export_chunk = new_chunk
            prev_export_audio = new_audio
            decoded_frames[prev_export_seg.index] = int(new_chunk.shape[0])
            new_audio_dict = new_audio if isinstance(new_audio, dict) else {}
            completed_audios[prev_export_seg.index] = new_audio_dict
            # Patch the already-finished lists (trimmed chunk must replace
            # the untrimmed one already appended above).
            run_pos = run_pos_map.get(prev_export_seg.index)
            if run_pos is not None and run_pos < len(segment_outputs):
                segment_outputs[run_pos] = new_chunk
                segment_audios[run_pos] = new_audio_dict
                export_frame_counts[run_pos] = int(new_chunk.shape[0])
            # concat_chunks_lazy reads from disk — the cache must match.
            # save_segment_cache rewrites the head/tail window *and* the
            # segment's clip.mp4, so both follow the shortened export.
            ph = dict(completed_av_handoff.get(prev_export_seg.index) or {})
            ph["export_frames"] = int(new_chunk.shape[0])
            ph["phase_align_trim"] = pending_trim
            completed_av_handoff[prev_export_seg.index] = ph
            # Re-persist with the same AV latent: the rewritten export must
            # not lose the latent the following segment pins from.
            save_segment_cache(
                node_id, prev_export_seg, plan, new_chunk,
                av_latent=_latent_for_cache(
                    node_id, prev_export_seg.index, completed_av_latents, cache_dir
                ),
                handoff=ph,
                audio=new_audio if isinstance(new_audio, dict) else None,
                replace_audio=False,
                workflow_name=workflow_name,
            )
            if mp4_run_dir is not None:
                maybe_export_segment_mp4(
                    mp4_run_dir, plan, prev_export_seg, new_chunk,
                    new_audio_dict,
                )
            reports.append(
                f"  Seg #{prev_export_seg.index + 1}: phase-align trim — dropped "
                f"{pending_trim}f orphaned tail (export {int(new_chunk.shape[0])}f)"
            )
            del new_chunk
        else:
            log.warning(
                "Director batch: skip %df phase-align trim on seg #%d "
                "(%df export is too short).",
                pending_trim, prev_export_seg.index + 1,
                int(prev_export_chunk.shape[0]),
            )

    prev_export_seg = seg
    prev_export_chunk = chunk
    prev_export_audio = audio_out

    # Free decoded frames from GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reports.append(
        f"  Seg #{seg.index + 1}: decoded {chunk.shape[0]}f → mp4 exported"
        f" (VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB)"
    )
