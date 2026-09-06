"""Run MiniMax H3 Director segments through the official ComfyUI core pipeline."""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..lib.image_prep import assert_minimax_canvas, fit_canvas, fit_video_long_edge
from ..lib.task_modes import SUPPORTED_TASK_KEYS
from ..nodes.conditioning import run_minimax_conditioning
from .conditioning_cache import (
    clear_conditioning_cache,
    get_cache_stats,
    load_conditioning_cache,
    save_conditioning_cache,
)
from .core_sampling import sample_single_stage
from .refine_pack import (
    confirm_first_pass_enabled,
    refine_needs_canvas,
    refine_passes_for,
    refine_will_sample,
)
from .refine_sampling import apply_segment_refine
from .frame_align import (
    minimax_align_frame_count, minimax_phase_aligned_export_frames, pad_or_trim_frames,
)
from .audio_export import (
    AUDIO_MODE_GENERATE,
    AUDIO_MODE_MUTE,
    AUDIO_MODE_SOURCE,
    empty_audio_dict,
    resolve_audio_mode,
)
from .segment_runtime import (
    frames_label,
    resolve_segment_raw_clip,
    segment_passthrough_chunk,
    tensor_frame_to_jpeg_b64,
)
from .plan import (
    DirectorPlan,
    plan_summary,
    prepare_segment_clip,
    resolve_ref_image_size,
    ref_audios_to_dict,
    ref_video_audios_to_dict,
    ref_videos_to_dict,
    reference_video_for_segment,
    refs_to_kwargs_for_context,
    reinforce_r2v_prompt,
    reinforce_rv2v_prompt,
    reinforce_v2v_prompt,
)
from .progress import report_director_finish, report_director_progress, report_director_segment_preview
from .h3_motion_context import (
    DEFAULT_AUDIO_CONTEXT_FRAMES,
    apply_motion_context,
    generation_frame_budget,
    handoff_end_frame,
    resolve_tail_context_length,
    snap_context_frames,
    trim_context_prefix,
    trim_export_tail,
)
from .segment_cache import (
    build_run_selection_clips,
    continuous_export_runs,
    load_first_pass_cache,
    load_next_segment_av_latent,
    load_segment_audio,
    load_segment_av_latent,
    load_segment_cache,
    load_segment_handoff_meta,
    save_first_pass_cache,
    save_segment_cache,
    sync_segment_slots,
)
from .segment_mp4_export import (
    copy_segment_mp4_suffix,
    maybe_export_segment_mp4,
    maybe_export_segment_mp4s,
    mp4_export_kind,
    new_segment_mp4_run_dir,
)
from .segment_continuity import (
    concat_chunks_lazy,
    concat_continuous_chunks,
    is_continuity_active,
    resolve_prev_segment_output,
)
from .vram_cleanup import cleanup_segment_vram

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.core")


def _resolve_tail_context_length(latent, seg, *, context_n: int) -> int:
    """Frames of the next segment to pin into this segment's tail.

    Shared helper lives in :mod:`h3_motion_context` so the batch path computes
    the identical value; this wrapper only logs segment context on failure.
    """
    n = resolve_tail_context_length(latent, int(seg.frame_count), context_n=context_n)
    if n <= 0:
        log.info(
            "Seg #%d: 对齐下段 inactive — sample has no spare tail capacity.",
            int(seg.index) + 1,
        )
    return n


def _unpack_node_output(out):
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    raise RuntimeError(f"Unexpected node output: {type(out)!r}")


def _decode_av_latent(samples, vae, audio_vae, *, decode_audio: bool = True):
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
        from comfy_extras.nodes_lt import VAEDecodeAudio  # type: ignore

    audio_out = VAEDecodeAudio.execute(audio_vae, audio_latent)
    audio = _unpack_node_output(audio_out)[0]
    return images, audio


def _trim_decoded_to_export(
    decoded: torch.Tensor,
    audio_dict: dict[str, Any] | None,
    *,
    trim_frames: int,
    export_len: int,
    plan: DirectorPlan,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    """Drop motion-context prefix and crop to the UI export length."""
    if trim_frames > 0:
        decoded, audio_dict = trim_context_prefix(
            decoded,
            audio_dict,
            trim_frames,
            fps=float(plan.frame_rate or 24),
            match_tail=True,
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


def _ref_tensor_from_seg_refs(refs, index: int) -> torch.Tensor | None:
    for ref in refs or []:
        if int(getattr(ref, "index", -1)) == index and ref.tensor is not None:
            t = ref.tensor
            if t.shape[0] > 0:
                return t[:1]
    return None


def _extract_referenced_picture_indices(prompt: str) -> set[int]:
    """Extract all <Picture N> indices referenced in the prompt.
    
    Searches for patterns like:
    - <Picture 1>, <Picture 2>, etc.
    - 图片1, 图片2, etc. (Chinese variant)
    
    Returns a set of 0-based indices (so <Picture 1> → {0}).
    """
    import re
    
    if not prompt:
        return set()
    
    indices = set()
    
    # English pattern: <Picture 1>, <Picture 2>, etc.
    for match in re.finditer(r'<Picture\s+(\d+)>', prompt):
        idx = int(match.group(1)) - 1  # Convert to 0-based
        if idx >= 0:
            indices.add(idx)
    
    # Chinese pattern: 图片1, 图片2, etc.
    for match in re.finditer(r'图片\s*(\d+)', prompt):
        idx = int(match.group(1)) - 1  # Convert to 0-based
        if idx >= 0:
            indices.add(idx)
    
    return indices


def _filter_ref_images_by_prompt(
    ref_images: dict[str, Any] | None,
    prompt: str,
) -> dict[str, Any] | None:
    """Filter reference images to only those referenced in the prompt.
    
    ref_images format: {"ref_image_0": tensor, "ref_image_1": tensor, ...}
    
    This reduces VRAM usage and encoding time by skipping unreferenced materials.
    """
    if not ref_images:
        return None
    
    referenced_indices = _extract_referenced_picture_indices(prompt)
    if not referenced_indices:
        # No references found in prompt — keep all for backward compatibility
        return ref_images
    
    filtered = {}
    for key, value in ref_images.items():
        if value is None:
            continue
        # Extract index from "ref_image_0" → 0
        try:
            idx_str = key.split("_")[-1]
            idx = int(idx_str)
            if idx in referenced_indices:
                filtered[key] = value
        except (ValueError, IndexError):
            # Malformed key — skip
            continue
    
    return filtered if filtered else None


def _build_minimax_inputs(
    plan: DirectorPlan,
    seg,
    *,
    clip_frames: torch.Tensor | None,
    ctx_w: int,
    ctx_h: int,
    prev_tail: torch.Tensor | None,
):
    """Map segment task + refs to MiniMax ImageToVideo / ReferenceToVideo inputs."""
    task_key = seg.task_key
    first_frame = None
    last_frame = None
    ref_images = None
    ref_videos = None
    ref_audios = None
    ref_video_audios = None

    if task_key == "fl2v":
        # Prefer explicit shot refs (index 0=start, 1=end). Official FL2VA allows
        # end-only — never invent a first_frame from the placeholder gen source_video
        # (1×16×16 gray) or a held clip when refs only carry image1.
        first_frame = _ref_tensor_from_seg_refs(seg.refs, 0)
        last_frame = _ref_tensor_from_seg_refs(seg.refs, 1)
        # Empty fl2v shot = text-to-video. Do not invent keyframes from the
        # 1×16×16 gray placeholder or a held clip. End-only also must not
        # promote clip_frames[0] into first_frame.
        if first_frame is not None and last_frame is None and clip_frames is not None:
            # Start+end endpoint hold: last may only live on the clip tail.
            if clip_frames.shape[0] >= 2:
                last_frame = clip_frames[-1:].clone()
    elif task_key == "i2v":
        # Explicit per-segment image wins; motion-context path leaves first_frame empty
        # so the previous tail can be pinned as a multi-frame head instead.
        if clip_frames is not None and clip_frames.shape[0] > 0:
            first_frame = clip_frames[:1]
        else:
            first_frame = _ref_tensor_from_seg_refs(seg.refs, 0)
        del prev_tail
    elif task_key == "r2v":
        ref_kwargs = refs_to_kwargs_for_context(task_key, seg.refs)
        ref_images = {}
        for key, tensor in ref_kwargs.items():
            if tensor is None:
                continue
            idx = key.removeprefix("reference_image_")
            ref_images[f"ref_image_{idx}"] = tensor[:1] if tensor.ndim == 4 else tensor
        if not ref_images:
            ref_images = None
        else:
            # Filter to only those referenced in the prompt (R2V optimization)
            filtered = _filter_ref_images_by_prompt(ref_images, seg.prompt or "")
            if filtered:
                ref_images = filtered
                log.debug(
                    "Seg #%d R2V ref_images filtered by prompt: %d → %d items",
                    seg.index + 1,
                    len(ref_kwargs),
                    len(filtered),
                )
        # Prefer multi-slot ref_videos (r2v batch cards); fall back to legacy single meta.
        ref_videos = ref_videos_to_dict(getattr(seg, "ref_videos", None) or [])
        if not ref_videos:
            nframes = max(5, int(getattr(seg, "frame_count", 0) or plan.total_frames or 124))
            ref_video = reference_video_for_segment(plan, seg, num_frames=nframes)
            if ref_video is not None and ref_video.shape[0] > 0:
                ref_videos = {"ref_video_0": ref_video}
        ref_audios = ref_audios_to_dict(getattr(seg, "ref_audios", None) or [])
        ref_video_audios = _ref_video_audios_to_dict(getattr(seg, "ref_video_audios", None) or [])
    elif task_key in {"v2v", "rv2v"}:
        # Bernini-style video edit: each timeline segment's source clip → <Video 1>.
        # rv2v additionally injects 图片1–9 / 音频1–3 as <Picture N> / <Audio J>.
        if clip_frames is None or clip_frames.shape[0] <= 0:
            raise ValueError(
                f"{task_key} segment #{seg.index + 1} has no source frames. "
                "Upload a video in the Director timeline before running."
            )
        ref_videos = {"ref_video_0": clip_frames}
        if task_key == "rv2v":
            # Refs are optional per segment: with refs → <Video 1>+<Picture N>;
            # without refs → same as v2v (source edit only).
            ref_kwargs = refs_to_kwargs_for_context(task_key, seg.refs)
            ref_images = {}
            for key, tensor in ref_kwargs.items():
                if tensor is None:
                    continue
                idx = key.removeprefix("reference_image_")
                ref_images[f"ref_image_{idx}"] = tensor[:1] if tensor.ndim == 4 else tensor
            if not ref_images:
                ref_images = None
            else:
                # Filter to only those referenced in the prompt (RV2V optimization)
                filtered = _filter_ref_images_by_prompt(ref_images, seg.prompt or "")
                if filtered:
                    ref_images = filtered
                    log.debug(
                        "Seg #%d RV2V ref_images filtered by prompt: %d → %d items",
                        seg.index + 1,
                        len(ref_kwargs),
                        len(filtered),
                    )
            ref_audios = ref_audios_to_dict(getattr(seg, "ref_audios", None) or [])

    return first_frame, last_frame, ref_images, ref_videos, ref_audios, ref_video_audios


def _ref_video_audios_to_dict(items) -> dict | None:
    # Shared with batch mode — see plan.ref_video_audios_to_dict.
    return ref_video_audios_to_dict(items)


def execute_director_plan_core(
    plan: DirectorPlan,
    *,
    node_id: str | None = None,
    workflow_name: str | None = None,
    model,
    vae,
    audio_vae,
    clip,
    cfg: float = 1.0,
    seed: int = 0,
    steps: int = 25,
    sampler: str = "res_multistep",
    scheduler: str = "simple",
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    clear_vram_between_segments: bool = True,
    use_conditioning_cache: bool = False,
    clear_conditioning_cache_on_run: bool = False,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[dict[str, Any]],
    str,
    list[int],
    torch.Tensor,
    list[torch.Tensor],
    bool,
]:
    """Process every segment with MiniMax H3 conditioning + single-stage sampling."""
    plan.sample_seed = int(seed)
    plan.sample_cfg = float(cfg)
    plan.sample_steps = int(steps)
    plan.sample_sampler = str(sampler or "")
    plan.sample_scheduler = str(scheduler or "")
    plan.sample_shift_video = float(shift_video)
    plan.sample_shift_audio = float(shift_audio)
    audio_mode = resolve_audio_mode(plan)
    decode_audio = audio_mode == AUDIO_MODE_GENERATE
    # UI toggle on the player bar (timeline.liveTaePreview); default on.
    raw_live = (plan.raw or {}).get("liveTaePreview", (plan.raw or {}).get("live_tae_preview", True))
    live_tae_preview = False if raw_live in (False, 0, "0", "false", "False", "off") else True

    # Conditioning cache management
    if clear_conditioning_cache_on_run:
        cleared = clear_conditioning_cache(node_id, workflow_name=workflow_name)
        if cleared > 0:
            reports_init = [f"Conditioning cache: cleared {cleared} files."]
        else:
            reports_init = ["Conditioning cache: no files to clear."]
    else:
        reports_init = []
    
    if use_conditioning_cache:
        cache_stats = get_cache_stats(node_id)
        if cache_stats["exists"]:
            reports_init.append(
                f"Conditioning cache: {cache_stats['files']} files, "
                f"{cache_stats['total_size_mb']} MB"
            )

    all_segments = plan.segments
    # Reconcile cache files with the current timeline *before* anything reads or
    # writes them. Content-addressed, so a group deleted in the middle takes its
    # own files and every other group keeps (or re-adopts) the render that
    # matches its content — no positional reshuffle. Uses every segment (not
    # run_indices): unselected「选择运行」slots still fill merge/export from disk.
    sync_segment_slots(node_id, plan, workflow_name=workflow_name)
    # Strictly honor「选择运行」— never force-sample unselected segments.
    run_indices = plan.run_indices if plan.run_indices is not None else frozenset(range(len(all_segments)))

    run_list = sorted(run_indices)
    seg_total = len(run_list)
    # A「选择运行」that leaves segments out. Under「全部导出」those slots used to be
    # re-read from cache / source and spliced back into one full-timeline clip;
    # instead we export one clip per contiguous run of what actually ran.
    partial_run = plan.run_indices is not None and len(run_list) < len(all_segments)
    progress_pos = {idx: pos for pos, idx in enumerate(run_list)}
    passthrough_indices: list[int] = []
    # External groups may compact selected packs to 0..N-1 while UI still shows
    # the full group list — prefer original timeline card count for progress UI.
    ext_meta = (plan.raw or {}).get("externalGroups") or {}
    try:
        timeline_seg_total = int(ext_meta.get("count") or 0) or len(all_segments)
    except (TypeError, ValueError):
        timeline_seg_total = len(all_segments)
    timeline_seg_total = max(timeline_seg_total, len(all_segments))

    output_chunks: list[torch.Tensor] = []
    output_pre_chunks: list[torch.Tensor] = []
    output_segments: list = []  # plans aligned 1:1 with output_chunks (skips omitted)
    segment_outputs: list[torch.Tensor] = []
    segment_pre_refine: list[torch.Tensor] = []
    segment_audios: list[dict[str, Any]] = []
    skipped_no_cache: list[int] = []
    # In-memory chunks keyed by timeline index for segments with no disk cache
    # (source passthrough). The merge reads everything else from disk.
    merge_overrides: dict[int, torch.Tensor] = {}
    reports: list[str] = [plan_summary(plan), "", "Execution path: ComfyUI official MiniMax H3"]
    # Add conditioning cache reports
    reports.extend(reports_init)
    if use_conditioning_cache:
        reports.append("Conditioning cache: ENABLED — skip CLIP encoding for cached segments.")
    # One timestamp folder per execute so all segments of this run stay together.
    # A partial「选择运行」also gets one: those VAE-decoded clips are the output.
    mp4_run_dir = new_segment_mp4_run_dir(plan, for_selection=partial_run)
    if mp4_run_dir is not None:
        reports.append(f"Segment mp4 export dir: {mp4_run_dir}")
    if clear_vram_between_segments:
        reports.append("VRAM: 段间清理显存已开启。")
    if audio_mode == AUDIO_MODE_MUTE:
        reports.append("Audio: muted — skip audio VAE decode, silent AUDIO output.")
    elif audio_mode == AUDIO_MODE_SOURCE:
        reports.append("Audio: source — skip audio VAE decode, use original timeline audio.")
    else:
        reports.append("Audio: generate — decode MiniMax H3 AV latent audio.")
    selected_ui = ext_meta.get("selected")
    if selected_ui is not None:
        selected_set = {int(x) for x in selected_ui}
        run_ui = [i + 1 for i in sorted(selected_set)]
        skipped = [i + 1 for i in range(timeline_seg_total) if i not in selected_set]
        reports.append(
            f"Run selection: {len(run_list)}/{timeline_seg_total} segment(s) "
            f"(indices {run_ui}; skipped {skipped or 'none'})"
        )
    elif plan.run_indices is not None:
        skipped = [i + 1 for i in range(len(all_segments)) if i not in run_indices]
        reports.append(
            f"Run selection: {len(run_list)}/{len(all_segments)} segment(s) "
            f"(indices {[i + 1 for i in run_list]}; skipped {skipped or 'none'})"
        )

    if plan.continuity_enabled:
        pinned = [
            seg.index + 1
            for seg in all_segments
            if seg.index > 0 and getattr(seg, "continuity_from_prev", True)
        ]
        skipped_pin = [
            seg.index + 1
            for seg in all_segments
            if seg.index > 0 and not getattr(seg, "continuity_from_prev", True)
        ]
        reports.append(
            "Segment continuity: ON — motion context "
            f"{snap_context_frames(plan.continuity_overlap_frames)}f "
            "(pin previous AV tail + trim prefix; t2v/i2v/fl2v/r2v/v2v/rv2v)."
        )
        if pinned:
            reports.append("  Pin from prev: #" + ", #".join(str(i) for i in pinned))
        if skipped_pin:
            reports.append(
                "  Hard cut (per-segment off): #"
                + ", #".join(str(i) for i in skipped_pin)
            )
    else:
        reports.append(
            "Segment continuity: OFF — official MiniMax H3 per-segment path "
            "(no motion-context pin/patch/trim; r2v/v2v/rv2v use stock ReferenceToVideo)."
        )

    completed_outputs: dict[int, torch.Tensor] = {}
    completed_pre_refine: dict[int, torch.Tensor] = {}
    completed_refine_passes: dict[int, list[tuple[str, torch.Tensor]]] = {}
    completed_av_latents: dict[int, dict] = {}
    completed_av_handoff: dict[int, dict] = {}
    completed_audios: dict[int, dict] = {}
    held_for_confirmation = False

    def _run_one_segment(
        seg, *, progress_index: int
    ) -> tuple[torch.Tensor, dict[str, Any] | None, torch.Tensor]:
        nonlocal held_for_confirmation
        if seg.task_key not in SUPPORTED_TASK_KEYS:
            raise ValueError(
                f"Task '{seg.task_key}' is not supported on MiniMax H3 Director. "
                f"Supported: {', '.join(sorted(SUPPORTED_TASK_KEYS))}."
            )

        ui_idx = seg.timeline_index
        will_refine = refine_will_sample(plan, seg)
        confirm_first = confirm_first_pass_enabled(plan)
        pre_cache = (
            load_first_pass_cache(node_id, seg, plan, workflow_name=workflow_name)
            if confirm_first and will_refine
            else None
        )
        skip_first_sample = pre_cache is not None
        hold_after_first = confirm_first and will_refine and not skip_first_sample
        held_for_confirmation = held_for_confirmation or hold_after_first
        meta = {
            "frames_label": frames_label(seg),
            "task_key": seg.task_key,
            "timeline_segment_index": ui_idx,
            "timeline_segment_total": timeline_seg_total,
        }

        report_director_progress(
            node_id, segment_index=progress_index, segment_total=seg_total,
            phase="prepare", phase_value=0, phase_max=1, **meta,
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
                # long_edge may leave storage-sized frames (e.g. 496) that are not
                # 32-aligned; lock to the resolved plan canvas after the long-edge fit.
                clip_frames = fit_video_long_edge(body_raw, plan.ref_max_size)
                if (
                    int(clip_frames.shape[1]) != int(plan.height)
                    or int(clip_frames.shape[2]) != int(plan.width)
                ):
                    clip_frames = fit_canvas(clip_frames, plan.width, plan.height)
        else:
            clip_frames = None

        num_frames = minimax_align_frame_count(target_len)
        if clip_frames is not None:
            clip_frames, _ = prepare_segment_clip(clip_frames, num_frames)

        # ── Continuity gate ─────────────────────────────────────────────
        # OFF → official MiniMax H3 path only (no prev load / pin / patch).
        # ON  → after stock conditioning, pin previous AV tail (incl. r2v/v2v/rv2v).
        continuity_active = is_continuity_active(plan, seg)
        prev_tail = None
        prev_av = None
        prev_audio = None
        prev_end_frame = None
        prev_idx = seg.index - 1
        if continuity_active:
            if prev_idx in passthrough_indices:
                raise ValueError(
                    f"段间连贯：片段 #{seg.index + 1} 的前一段 #{prev_idx + 1} "
                    "是源视频透传（未采样/无有效缓存），不能作为 motion context。"
                    "请先运行该段，或将其纳入「选择运行」。"
                )
            prev_tail = resolve_prev_segment_output(
                plan, all_segments, seg.index, completed_outputs, node_id, workflow_name=workflow_name
            )
            # Hydrate prev into completed_* so phase-align trim can rewrite
            # in-memory exports + disk cache even on「分段导出」/ partial re-run
            # (resolve_prev may return a cache tensor without storing it).
            if prev_idx >= 0 and prev_tail is not None and prev_idx not in completed_outputs:
                completed_outputs[prev_idx] = prev_tail
            prev_seg = all_segments[prev_idx] if prev_idx >= 0 else None
            prev_av = completed_av_latents.get(prev_idx)
            if prev_av is None and prev_seg is not None:
                prev_av = load_segment_av_latent(
                    node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name
                )
                if prev_av is not None:
                    completed_av_latents[prev_idx] = prev_av
            prev_handoff = completed_av_handoff.get(prev_idx)
            if prev_handoff is None and prev_seg is not None:
                prev_handoff = load_segment_handoff_meta(
                    node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name
                )
                if prev_handoff is not None:
                    completed_av_handoff[prev_idx] = prev_handoff
            prev_audio = completed_audios.get(prev_idx)
            if prev_audio is None and prev_seg is not None:
                prev_audio = load_segment_audio(
                    node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name
                )
                if prev_audio is not None:
                    completed_audios[prev_idx] = prev_audio
            if prev_handoff:
                prev_end_frame = handoff_end_frame(
                    trim_frames=int(prev_handoff.get("trim_frames") or 0),
                    export_frames=int(prev_handoff.get("export_frames") or 0),
                )
                sample_f = int(prev_handoff.get("sample_frames") or 0)
                # Absolute latent tail is only safe when it equals the export end.
                if sample_f > 0 and prev_end_frame >= sample_f:
                    prev_end_frame = None
            else:
                # Pixel fallback: decoded export has no overshoot beyond the file.
                prev_end_frame = None

        ctx_w = int(plan.width)
        ctx_h = int(plan.height)
        if clip_frames is not None and clip_frames.shape[0] > 0:
            ctx_h, ctx_w = int(clip_frames.shape[1]), int(clip_frames.shape[2])
        # H3 patchify requires W/H multiples of 32 (VAE÷16 then 2×2).
        assert_minimax_canvas(ctx_w, ctx_h)

        report_director_progress(
            node_id, segment_index=progress_index, segment_total=seg_total,
            phase="prepare", phase_value=1, phase_max=1, **meta,
        )

        positive_prompt = seg.prompt

        if seg.task_key == "fl2v":
            from .fl2v_timeline import reinforce_fl2v_prompt

            has_start = any(getattr(r, "index", None) == 0 for r in (seg.refs or []))
            has_end = any(getattr(r, "index", None) == 1 for r in (seg.refs or []))
            if not has_start and not has_end and seg.refs:
                # Legacy packs without explicit indices: [start] or [start, end].
                has_start = True
                has_end = len(seg.refs) >= 2
            positive_prompt = reinforce_fl2v_prompt(
                positive_prompt,
                has_end_frame=has_end,
                has_start_frame=has_start,
            )
        elif seg.task_key == "r2v":
            ref_idxs = [int(getattr(r, "index", 0)) for r in (seg.refs or []) if r is not None]
            vid_idxs = [int(getattr(v, "index", 0)) for v in (getattr(seg, "ref_videos", None) or []) if v is not None]
            audio_idxs = [int(getattr(a, "index", 0)) for a in (seg.ref_audios or []) if a is not None]
            positive_prompt = reinforce_r2v_prompt(
                positive_prompt,
                ref_indices=ref_idxs,
                video_indices=vid_idxs,
                audio_indices=audio_idxs,
            )
        elif seg.task_key == "v2v":
            positive_prompt = reinforce_v2v_prompt(positive_prompt)
        elif seg.task_key == "rv2v":
            ref_idxs = [int(getattr(r, "index", 0)) for r in (seg.refs or []) if r is not None]
            audio_idxs = [int(getattr(a, "index", 0)) for a in (seg.ref_audios or []) if a is not None]
            positive_prompt = reinforce_rv2v_prompt(
                positive_prompt, ref_indices=ref_idxs, audio_indices=audio_idxs,
            )

        report_director_progress(
            node_id, segment_index=progress_index, segment_total=seg_total,
            phase="context_encode", phase_value=0, phase_max=1, **meta,
        )

        # Stock official inputs first (refs / source video / keyframes).
        # prev_tail is unused for r2v/v2v/rv2v here — MC pins after conditioning.
        first_frame, last_frame, ref_images, ref_videos, ref_audios, ref_video_audios = _build_minimax_inputs(
            plan, seg, clip_frames=clip_frames, ctx_w=ctx_w, ctx_h=ctx_h, prev_tail=None,
        )

        # i2v with an explicit new start image = fresh anchor (skip motion context).
        # fl2v keeps last_frame when continuity is on; first_frame yields to context head.
        # r2v/v2v/rv2v: always eligible for MC when continuity_active (refs stay).
        i2v_new_anchor = seg.task_key == "i2v" and first_frame is not None
        use_motion_context = (
            continuity_active
            and not i2v_new_anchor
            and (prev_av is not None or prev_tail is not None)
        )
        if skip_first_sample:
            # Cached first-pass latent already has its original pin; don't rebuild MC.
            use_motion_context = False
        # OFF → context_n=0 → sample_len == official segment length only.
        context_n = snap_context_frames(plan.continuity_overlap_frames) if use_motion_context else 0
        sample_len, _planned_trim = generation_frame_budget(num_frames, context_n)
        if use_motion_context:
            # Clear single-frame first lock so multi-frame context owns the head.
            first_frame = None

        if seg.task_key in {"r2v", "v2v", "rv2v"} and (
            ref_images or ref_videos or ref_audios or ref_video_audios
        ) and audio_vae is None:
            raise ValueError("r2v/v2v/rv2v / reference conditioning requires audio_vae input.")

        # Conditioning cache: try to load from cache first
        cached_conditioning = None
        if use_conditioning_cache and not skip_first_sample:
            cached_conditioning = load_conditioning_cache(
                node_id=node_id,
                segment_index=seg.index,
                prompt=positive_prompt,
                width=ctx_w,
                height=ctx_h,
                length=sample_len,
                task_key=seg.task_key,
                ref_image_size=resolve_ref_image_size(seg, plan),
                ref_images=ref_images,
                workflow_name=workflow_name,
            )

        if cached_conditioning is not None:
            # Cache hit: use cached conditioning, skip CLIP encoding
            positive = cached_conditioning["positive"]
            negative = cached_conditioning["negative"]
            latent = cached_conditioning["latent"]
            task_hint = f"{seg.task_key} (cached)"
            reports.append(
                f"Seg #{seg.index + 1}: conditioning CACHE HIT — skip CLIP encoding"
            )
        else:
            # Cache miss: run full conditioning pipeline
            positive, negative, latent, task_hint = run_minimax_conditioning(
                clip=clip,
                vae=vae,
                audio_vae=audio_vae,
                prompt=positive_prompt,
                width=ctx_w,
                height=ctx_h,
                length=sample_len,
                task_key=seg.task_key,
                first_frame=first_frame,
                last_frame=last_frame,
                ref_images=ref_images,
                ref_videos=ref_videos,
                ref_video_audios=ref_video_audios,
                ref_audios=ref_audios,
                ref_image_size=resolve_ref_image_size(seg, plan),
            )
            
            # Save to cache for next run
            if use_conditioning_cache:
                save_conditioning_cache(
                    node_id=node_id,
                    segment_index=seg.index,
                    positive=positive,
                    negative=negative,
                    latent=latent,
                    prompt=positive_prompt,
                    width=ctx_w,
                    height=ctx_h,
                    length=sample_len,
                    task_key=seg.task_key,
                    ref_image_size=resolve_ref_image_size(seg, plan),
                    ref_images=ref_images,
                    workflow_name=workflow_name,
                )
                reports.append(
                    f"Seg #{seg.index + 1}: conditioning SAVED to cache"
                )

        trim_frames = 0
        if use_motion_context:
            # Pin audio from previous AV latent whenever available (official MC path).
            # Do not gate on decode_audio — mute only skips final audio decode.
            pin_audio = (
                audio_mode != AUDIO_MODE_MUTE
                and (prev_av is not None or prev_audio is not None)
            )
            # ------------------------------------------------------------------
            # 对齐下段 (align-to-next): pin the next segment's opening into this
            # segment's tail. Cache-driven middle-out mode — only runs when that
            # neighbour already holds an AV latent, otherwise it silently no-ops.
            # ------------------------------------------------------------------
            tail_context_latent = None
            tail_context_length = 0
            if bool(getattr(seg, "continuity_to_next", False)):
                tail_n = _resolve_tail_context_length(
                    latent, seg, context_n=context_n
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
                        log.info(
                            "Seg #%d: 对齐下段 active — pinning next segment head %df.",
                            seg.index + 1,
                            tail_n,
                        )
            positive, trim_frames, prev_export_trim = apply_motion_context(
                positive,
                latent,
                vae=vae,
                context_length=context_n,
                context_latent=prev_av,
                context_frames=prev_tail,
                # Always pass export audio so a canvas-mismatch fallback
                # (Refine upscale) can still pin audio from the decoded tail.
                context_audio=prev_audio,
                audio_vae=audio_vae,
                continue_audio=pin_audio,
                # t2v/i2v/r2v/v2v/rv2v: context owns the head.
                # fl2v keeps last_frame, marked so origin-shift retiming can move it.
                keep_existing_keyframes=(seg.task_key == "fl2v"),
                context_end_frame=prev_end_frame,
                audio_context_length=DEFAULT_AUDIO_CONTEXT_FRAMES,
                tail_context_latent=tail_context_latent,
                tail_context_length=tail_context_length,
            )
            # Phase-align can pin a few frames before the previous export end.
            # Drop that orphaned tail so concat does not replay it at the seam.
            # With the 17-frame-grid export length this is normally 0 and the
            # block below is dead code; kept as a safety net for the edges where
            # the pin window can still fall short (pixel-pin fallback,
            # non-standard context lengths, hand-off from an older cache).
            trimmed_prev_export = 0
            if prev_export_trim > 0:
                fps = float(plan.frame_rate or 24)
                prev_chunk = completed_outputs.get(prev_idx)
                if prev_chunk is None and prev_idx >= 0:
                    # Last-resort hydrate (export_mode=segments skipped the prev loop).
                    prev_seg_lazy = next(
                        (s for s in all_segments if s.index == prev_idx), None
                    )
                    if prev_seg_lazy is not None:
                        prev_chunk = load_segment_cache(
                            node_id, prev_seg_lazy, plan, allow_stale=True, workflow_name=workflow_name
                        )
                        if prev_chunk is not None:
                            completed_outputs[prev_idx] = prev_chunk
                            if prev_idx not in completed_audios:
                                lazy_aud = load_segment_audio(
                                    node_id, prev_seg_lazy, plan, allow_stale=True, workflow_name=workflow_name
                                )
                                if lazy_aud is not None:
                                    completed_audios[prev_idx] = lazy_aud
                if prev_chunk is not None:
                    prev_chunk, prev_audio_trim = trim_export_tail(
                        prev_chunk,
                        completed_audios.get(prev_idx),
                        prev_export_trim,
                        fps=fps,
                    )
                    completed_outputs[prev_idx] = prev_chunk
                    if prev_audio_trim is not None:
                        completed_audios[prev_idx] = prev_audio_trim
                    # Lists may already hold the untrimmed tensor/audio from when
                    # the previous segment finished — patch by timeline index.
                    run_pos = progress_pos.get(prev_idx)
                    if run_pos is not None:
                        if run_pos < len(segment_outputs):
                            segment_outputs[run_pos] = prev_chunk
                        if (
                            prev_audio_trim is not None
                            and run_pos < len(segment_audios)
                        ):
                            segment_audios[run_pos] = prev_audio_trim
                    # output_chunks is dense (skipped slots omitted) — match by seg.index.
                    if plan.export_mode == "all":
                        for oi, oseg in enumerate(output_segments):
                            if getattr(oseg, "index", -1) == prev_idx:
                                output_chunks[oi] = prev_chunk
                                break
                    prev_pre = completed_pre_refine.get(prev_idx)
                    if prev_pre is not None:
                        if int(prev_pre.shape[0]) > prev_export_trim:
                            prev_pre, _ = trim_export_tail(
                                prev_pre, None, prev_export_trim, fps=fps
                            )
                        completed_pre_refine[prev_idx] = prev_pre
                        if run_pos is not None and run_pos < len(segment_pre_refine):
                            segment_pre_refine[run_pos] = prev_pre
                        if plan.export_mode == "all":
                            for oi, oseg in enumerate(output_segments):
                                if getattr(oseg, "index", -1) == prev_idx:
                                    if oi < len(output_pre_chunks):
                                        output_pre_chunks[oi] = prev_pre
                                    break
                    # Persist the trimmed export so partial re-runs reload the same
                    # A/V lengths — the head/tail window and the segment's clip.mp4
                    # are rewritten together, so the disk cache cannot end up
                    # holding the pre-trim body with the post-trim head/tail.
                    # replace_audio=False: do not unlink audio.pt when only video
                    # was hydrated.
                    prev_seg = next(
                        (s for s in all_segments if s.index == prev_idx), None
                    )
                    if prev_seg is not None:
                        prev_handoff = dict(completed_av_handoff.get(prev_idx) or {})
                        prev_handoff["export_frames"] = int(prev_chunk.shape[0])
                        prev_handoff["phase_align_trim"] = int(prev_export_trim)
                        completed_av_handoff[prev_idx] = prev_handoff
                        save_segment_cache(
                            node_id,
                            prev_seg,
                            plan,
                            prev_chunk,
                            av_latent=completed_av_latents.get(prev_idx),
                            handoff=prev_handoff,
                            audio=completed_audios.get(prev_idx),
                            replace_audio=False,
                            workflow_name=workflow_name,
                        )
                        # Rewrite incremental mp4 so mid-run files match trimmed length.
                        if hold_after_first:
                            pre_path = maybe_export_segment_mp4(
                                mp4_run_dir,
                                plan,
                                prev_seg,
                                prev_chunk,
                                completed_audios.get(prev_idx),
                                suffix="pre",
                            )
                            mp4_paths = [pre_path] if pre_path else []
                        else:
                            mp4_paths = maybe_export_segment_mp4s(
                                mp4_run_dir,
                                plan,
                                prev_seg,
                                prev_chunk,
                                completed_audios.get(prev_idx),
                                pre_frames=completed_pre_refine.get(prev_idx),
                            )
                        extra_passes = list(completed_refine_passes.get(prev_idx) or [])
                        rewritten_extra: list[tuple[str, torch.Tensor]] = []
                        for suffix, frames in extra_passes:
                            clipped = frames
                            if int(clipped.shape[0]) > prev_export_trim:
                                clipped, _ = trim_export_tail(
                                    clipped, None, prev_export_trim, fps=fps
                                )
                            rewritten_extra.append((suffix, clipped))
                            extra_path = maybe_export_segment_mp4(
                                mp4_run_dir,
                                plan,
                                prev_seg,
                                clipped,
                                completed_audios.get(prev_idx),
                                suffix=suffix,
                            )
                            if extra_path:
                                mp4_paths.append(extra_path)
                        if rewritten_extra:
                            completed_refine_passes[prev_idx] = rewritten_extra
                            last_alias = copy_segment_mp4_suffix(
                                mp4_run_dir,
                                plan,
                                prev_seg,
                                dest_suffix=f"p{len(rewritten_extra) + 1}",
                            )
                            if last_alias:
                                mp4_paths.append(last_alias)
                        for mp4_path in mp4_paths:
                            reports.append(
                                f"Segment {prev_idx + 1}: {mp4_export_kind(mp4_path)} "
                                f"updated after continuity trim → {mp4_path}"
                            )
                    trimmed_prev_export = int(prev_export_trim)
                    log.info(
                        "Director continuity: trimmed %df from seg #%d export "
                        "(phase-align pin gap)",
                        prev_export_trim,
                        prev_idx + 1,
                    )
                else:
                    log.warning(
                        "Director continuity: phase-align wanted to trim %df from "
                        "seg #%d but prev export was unavailable — seam may echo.",
                        prev_export_trim,
                        prev_idx + 1,
                    )
            task_hint = f"{task_hint} + motion context {trim_frames}f"
            reports.append(
                f"Seg #{seg.index + 1}: motion context ON — pin {trim_frames}f "
                f"from seg #{seg.index} "
                f"({'AV latent' if prev_av is not None else 'pixels'}"
                f"{', +audio' if pin_audio else ', video-only'}); "
                f"sample={sample_len}f → export {num_frames}f"
                + (
                    f"; trimmed prev export -{trimmed_prev_export}f (phase pin)"
                    if trimmed_prev_export
                    else ""
                )
            )

        report_director_progress(
            node_id, segment_index=progress_index, segment_total=seg_total,
            phase="context_encode", phase_value=1, phase_max=1, **meta,
        )

        if clear_vram_between_segments:
            cleanup_segment_vram(enabled=True, unload_models=seg_total > 1)

        def _report_sample_phase(phase: str, value: float) -> None:
            report_director_progress(
                node_id, segment_index=progress_index, segment_total=seg_total,
                phase=phase, phase_value=value, phase_max=1, **meta,
            )

        def _report_step_preview(step: int, total_steps: int, x0) -> None:
            # Live frame for the batch-card preview slot (「生成中…」 area).
            try:
                from .tae_preview import pil_to_jpeg_b64, x0_to_preview_pil

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

        if skip_first_sample:
            samples = pre_cache["av_latent"]
            cached_h = pre_cache.get("handoff") or {}
            trim_frames = int(cached_h.get("trim_frames") or 0)
            cached_sample = int(cached_h.get("sample_frames") or 0)
            if cached_sample > 0:
                sample_len = cached_sample
            reports.append(
                f"Segment {ui_idx + 1}/{timeline_seg_total}: 命中一采缓存 "
                f"(seed={int(getattr(plan, 'sample_seed', seed) or seed)})，跳过一采，开始二采"
            )
        else:
            samples = sample_single_stage(
                model=model,
                positive=positive,
                negative=negative,
                latent=latent,
                seed=seed,
                cfg=cfg,
                steps=steps,
                sampler_name=sampler,
                scheduler=scheduler,
                shift_video=shift_video,
                shift_audio=shift_audio,
                on_phase=_report_sample_phase,
                on_step_preview=_report_step_preview if live_tae_preview else None,
                preview_every=1,
            )

        first_pass_samples = samples
        first_pass_gpu = None
        pre_export = None
        run_refine = will_refine and not hold_after_first
        if will_refine:
            cached_frames = pre_cache.get("frames") if skip_first_sample else None
            if isinstance(cached_frames, torch.Tensor) and cached_frames.numel() > 0:
                pre_export = cached_frames.detach().cpu().float()
                if run_refine and isinstance(getattr(plan, "refine", None), dict) and refine_needs_canvas(plan.refine):
                    first_pass_gpu = pre_export
            else:
                try:
                    report_director_progress(
                        node_id, segment_index=progress_index, segment_total=seg_total,
                        phase="decode", phase_value=0, phase_max=1, **meta,
                    )
                    first_pass_gpu, _ = _decode_av_latent(
                        samples, vae, audio_vae, decode_audio=False,
                    )
                    pre_export = first_pass_gpu.detach().cpu().float()
                except Exception as exc:
                    log.warning(
                        "Segment %s first-pass decode for images_pre_refine failed (%s).",
                        ui_idx + 1,
                        exc,
                    )
                    first_pass_gpu = None
                    pre_export = None

        pack = getattr(plan, "refine", None)
        upscale_frames = (
            first_pass_gpu
            if run_refine and isinstance(pack, dict) and refine_needs_canvas(pack)
            else None
        )
        if first_pass_gpu is not None and upscale_frames is None:
            del first_pass_gpu
            first_pass_gpu = None
        export_len = (
            minimax_phase_aligned_export_frames(num_frames)
            if trim_frames > 0
            else int(num_frames)
        )
        if will_refine and not skip_first_sample:
            save_first_pass_cache(
                node_id,
                seg,
                plan,
                av_latent=first_pass_samples,
                frames=pre_export,
                handoff={
                    "trim_frames": int(trim_frames),
                    "export_frames": int(export_len),
                    "sample_frames": int(sample_len),
                    "official_mc_length": False,
                },
                workflow_name=workflow_name,
            )
        pass_clips: list[tuple[str, torch.Tensor]] = []

        def _export_refine_pass(pass_i: int, n_passes: int, latent: dict) -> None:
            if mp4_run_dir is None or int(pass_i) >= int(n_passes):
                return
            suffix = f"p{int(pass_i)}"
            try:
                report_director_progress(
                    node_id, segment_index=progress_index, segment_total=seg_total,
                    phase="decode", phase_value=0, phase_max=1, **meta,
                )
                decoded_p, audio_p = _decode_av_latent(
                    latent, vae, audio_vae, decode_audio=decode_audio,
                )
                decoded_p, audio_p = _trim_decoded_to_export(
                    decoded_p,
                    audio_p,
                    trim_frames=trim_frames,
                    export_len=export_len,
                    plan=plan,
                )
                frames_p = decoded_p.cpu().float()
                del decoded_p
                path = maybe_export_segment_mp4(
                    mp4_run_dir,
                    plan,
                    seg,
                    frames_p,
                    audio_p if isinstance(audio_p, dict) else None,
                    suffix=suffix,
                )
                pass_clips.append((suffix, frames_p))
                if path:
                    reports.append(
                        f"Segment {ui_idx + 1}/{timeline_seg_total}: "
                        f"{mp4_export_kind(path)} saved → {path}"
                    )
            except Exception as exc:
                log.warning(
                    "Segment %s refine pass %d mp4 export failed (%s).",
                    ui_idx + 1,
                    pass_i,
                    exc,
                )

        if run_refine:
            samples, refine_note = apply_segment_refine(
                plan,
                seg,
                samples=samples,
                model=model,
                vae=vae,
                audio_vae=audio_vae,
                positive=positive,
                negative=negative,
                seed=seed,
                cfg=cfg,
                first_steps=steps,
                sampler_name=sampler,
                scheduler=scheduler,
                shift_video=shift_video,
                shift_audio=shift_audio,
                on_phase=_report_sample_phase,
                on_step_preview=_report_step_preview if live_tae_preview else None,
                first_pass_images=upscale_frames,
                trim_frames=trim_frames,
                on_pass=_export_refine_pass if mp4_run_dir is not None else None,
            )
        elif hold_after_first:
            refine_note = (
                f"先确认一采（已缓存 seed={int(getattr(plan, 'sample_seed', seed) or seed)}，未二采；"
                "用同一 seed 再 Queue 将只跑二采）"
            )
        else:
            refine_note = ""
        samples = first_pass_samples if not run_refine else samples
        del upscale_frames
        if first_pass_gpu is not None:
            del first_pass_gpu
            first_pass_gpu = None

        report_director_progress(
            node_id, segment_index=progress_index, segment_total=seg_total,
            phase="decode", phase_value=0, phase_max=1, **meta,
        )
        decoded, audio_dict = _decode_av_latent(
            samples, vae, audio_vae, decode_audio=decode_audio,
        )
        # Motion context: the sample is longer (visible+ctx, 17k+5 aligned);
        # _trim_decoded_to_export drops the pinned prefix and crops to export_len.
        # Next segment pins at export end (trim+export), not the sample end — and
        # that export end must sit on the 17-frame VAE cycle grid, otherwise the
        # pin window stops short and the remainder is either trimmed (lost) or
        # replayed as a seam echo. See minimax_phase_aligned_export_frames.
        export_len = (
            minimax_phase_aligned_export_frames(num_frames)
            if trim_frames > 0
            else int(num_frames)
        )
        # ``decoded_full`` is the raw VAE decode (158 frames, motion-context
        # extended) and is only the input to the trim below. What gets persisted
        # is ``decoded``: the continuity prefix (the ~22 replayed frames of the
        # previous segment) is DROPPED, so a standalone segment download starts at
        # its own first frame instead of replaying ~1s of the previous segment —
        # that prefix replay is what made every non-first segment look ~1s short
        # ("4s instead of 5s"). The merge reads this same trimmed body and does not
        # re-trim it (see ``concat_chunks_lazy``), so total merge length is
        # unchanged: first segment 124f + each following 119f = 481f for 4x5s.
        decoded_full = decoded
        decoded, audio_dict = _trim_decoded_to_export(
            decoded_full,
            audio_dict,
            trim_frames=trim_frames,
            export_len=export_len,
            plan=plan,
        )
        report_director_progress(
            node_id, segment_index=progress_index, segment_total=seg_total,
            phase="decode", phase_value=1, phase_max=1, **meta,
        )

        chunk = decoded.cpu().float()
        if pre_export is not None:
            pre_export, _ = _trim_decoded_to_export(
                pre_export,
                None,
                trim_frames=trim_frames,
                export_len=export_len,
                plan=plan,
            )
            pre_chunk = pre_export.cpu().float()
        else:
            pre_chunk = chunk
        if hold_after_first and pre_chunk is chunk:
            pre_chunk = chunk.clone()
        handoff = {
            "trim_frames": int(trim_frames),
            "export_frames": int(chunk.shape[0]),
            "sample_frames": int(sample_len),
            # False ⇒ next pin must use context_end_frame = trim+export.
            "official_mc_length": False,
        }
        completed_av_latents[seg.index] = samples
        completed_av_handoff[seg.index] = handoff
        if isinstance(audio_dict, dict) and audio_dict.get("waveform") is not None:
            completed_audios[seg.index] = audio_dict
        # Persist the trimmed body (continuity prefix dropped), so a standalone
        # segment starts at its own first frame. The merge reads this same body and
        # does not re-trim it, keeping the merged length identical.
        # save_segment_cache writes the head/tail window *and* the segment's
        # clip.mp4 from this one tensor, so「分段导出」serves this exact body
        # without a re-decode and the two can never fall out of sync.
        save_segment_cache(
            node_id,
            seg,
            plan,
            decoded,
            av_latent=samples,
            handoff=handoff,
            audio=audio_dict if isinstance(audio_dict, dict) else None,
            workflow_name=workflow_name,
        )
        completed_outputs[seg.index] = chunk
        completed_pre_refine[seg.index] = pre_chunk
        completed_refine_passes[seg.index] = pass_clips

        #「分段导出」: flush mp4 as soon as this segment succeeds (crash-safe).
        # Confirmation hold has no final/second-pass clip yet: save only _pre.
        if hold_after_first:
            pre_path = maybe_export_segment_mp4(
                mp4_run_dir,
                plan,
                seg,
                chunk,
                audio_dict if isinstance(audio_dict, dict) else None,
                suffix="pre",
            )
            mp4_paths = [pre_path] if pre_path else []
        else:
            # Final clip = last refine pass; _pre = 一采; _pN = each refine round.
            mp4_paths = maybe_export_segment_mp4s(
                mp4_run_dir,
                plan,
                seg,
                chunk,
                audio_dict if isinstance(audio_dict, dict) else None,
                pre_frames=pre_chunk if run_refine else None,
            )
        n_refine = refine_passes_for(getattr(plan, "refine", None)) if run_refine else 1
        if isinstance(pack, dict) and (pack.get("mode") or "") == "latent_upscale":
            n_refine = 1
        if n_refine > 1:
            last_alias = copy_segment_mp4_suffix(
                mp4_run_dir, plan, seg, dest_suffix=f"p{n_refine}",
            )
            if last_alias:
                mp4_paths.append(last_alias)
        for mp4_path in mp4_paths:
            reports.append(
                f"Segment {ui_idx + 1}/{timeline_seg_total}: "
                f"{mp4_export_kind(mp4_path)} saved → {mp4_path}"
            )

        if seg.task_key in {"t2v", "i2v", "r2v", "fl2v", "v2v", "rv2v"} and decoded.shape[0] >= 1:
            try:
                frames_b64 = [
                    tensor_frame_to_jpeg_b64(decoded[i])
                    for i in range(int(decoded.shape[0]))
                ]
                h, w = int(decoded.shape[1]), int(decoded.shape[2])
                report_director_segment_preview(
                    node_id,
                    segment_index=ui_idx,
                    image_b64=frames_b64[0],
                    width=w,
                    height=h,
                    frames=frames_b64,
                    fps=float(plan.frame_rate or 24),
                )
            except Exception as exc:
                log.debug("Segment video preview skipped: %s", exc)

        if clear_vram_between_segments:
            cleanup_segment_vram(enabled=True)

        reports.append(
            f"Segment {ui_idx + 1}/{timeline_seg_total}: {task_hint} "
            f"({target_len} frames, seed={seed}"
            f"{', ' + refine_note if refine_note else ''})"
        )
        log.info(
            "MiniMax H3 Director segment %d/%d done (%d frames, task=%s)",
            ui_idx + 1, timeline_seg_total, target_len, seg.task_key,
        )
        return chunk, audio_dict, pre_chunk

    for seg in all_segments:
        if seg.index in run_indices:
            if clear_vram_between_segments and segment_outputs:
                cleanup_segment_vram(enabled=True)
            chunk, audio_dict, pre_chunk = _run_one_segment(
                seg, progress_index=progress_pos[seg.index]
            )
            segment_outputs.append(chunk)
            segment_pre_refine.append(pre_chunk)
            segment_audios.append(audio_dict or {})
            if plan.export_mode == "all" and not partial_run:
                output_chunks.append(chunk)
                output_pre_chunks.append(pre_chunk)
                output_segments.append(seg)
            continue

        if plan.export_mode != "all":
            continue
        # Unselected slots are still hydrated below (continuity needs their AV
        # latent / handoff), they just no longer join the exported clips.

        # Prefer exact cache; pipeline-stale disk render is ok. A different
        # source video is rejected so v2v/rv2v can passthrough the new clip.
        cached = load_segment_cache(node_id, seg, plan, workflow_name=workflow_name)
        used_stale = False
        if cached is None:
            cached = load_segment_cache(node_id, seg, plan, allow_stale=True, workflow_name=workflow_name)
            used_stale = cached is not None
        if cached is not None:
            cached = cached.float()
            completed_outputs[seg.index] = cached
            completed_pre_refine[seg.index] = cached
            cached_audio = load_segment_audio(
                node_id, seg, plan, allow_stale=used_stale, workflow_name=workflow_name
            )
            if cached_audio is not None:
                completed_audios[seg.index] = cached_audio
            # Continuity for later sampled segments may need AV latent / handoff.
            cached_av = load_segment_av_latent(
                node_id, seg, plan, allow_stale=used_stale, workflow_name=workflow_name
            )
            if cached_av is not None:
                completed_av_latents[seg.index] = cached_av
            cached_handoff = load_segment_handoff_meta(
                node_id, seg, plan, allow_stale=used_stale, workflow_name=workflow_name
            )
            if cached_handoff is not None:
                completed_av_handoff[seg.index] = cached_handoff
            audio_note = ", +audio" if cached_audio is not None else ", no audio cache"
            stale_note = ", stale fingerprint" if used_stale else ""
            reports.append(
                f"Segment {seg.index + 1}/{len(all_segments)}: loaded from cache "
                f"({cached.shape[0]} frames{audio_note}{stale_note})"
                + (" — continuity only, not exported" if partial_run else "")
            )
            if not partial_run:
                output_chunks.append(cached)
                output_pre_chunks.append(cached)
                output_segments.append(seg)
            continue

        # Not selected + no cache: v2v/rv2v may fill from source video; gen batch must not
        # splice gray placeholders. If neither works, skip the slot (do not fail the run).
        fill = segment_passthrough_chunk(plan, seg)
        if fill is None:
            skipped_no_cache.append(seg.index + 1)
            reports.append(
                f"Segment {seg.index + 1}/{len(all_segments)}: skipped — no cache "
                "(outside run selection; omitted from merge)"
            )
            continue
        completed_outputs[seg.index] = fill
        completed_pre_refine[seg.index] = fill
        passthrough_indices.append(seg.index)
        # Passthrough exists only in RAM (never written to the segment cache),
        # so the merge must be handed it explicitly instead of reading from disk.
        merge_overrides[int(seg.index)] = fill
        reports.append(
            f"Segment {seg.index + 1}/{len(all_segments)}: source passthrough "
            f"({fill.shape[0]} frames, not sampled — outside run selection)"
        )
        if not partial_run:
            output_chunks.append(fill)
            output_pre_chunks.append(fill)
            output_segments.append(seg)

    if passthrough_indices:
        reports.append(
            "Passthrough (not sampled) segment(s) "
            f"{[i + 1 for i in passthrough_indices]} — used for continuity only"
            + ("" if partial_run else "; unselected gaps filled from source for「全部导出」.")
            + "."
        )
    if skipped_no_cache:
        reports.append(
            "Skipped segment(s) with no cache "
            f"{skipped_no_cache} — omitted from the export "
            "(勾选重跑或先全跑可补上)."
        )

    if not output_chunks and not segment_outputs:
        raise ValueError("Director plan produced no segments.")

    report_director_finish(node_id, seg_total)
    export_chunks = output_chunks if output_chunks else segment_outputs
    export_pre_chunks = output_pre_chunks if output_pre_chunks else segment_pre_refine
    export_segments = (
        output_segments
        if output_chunks
        else [all_segments[i] for i in sorted(run_indices)]
    )
    # Prefer completed_outputs: motion-context may have trimmed a prev export
    # tail (phase-align pin gap) after that chunk was already appended here.
    for i, seg in enumerate(export_segments):
        patched = completed_outputs.get(seg.index)
        if patched is not None:
            export_chunks[i] = patched
        patched_pre = completed_pre_refine.get(seg.index)
        if patched_pre is not None and i < len(export_pre_chunks):
            export_pre_chunks[i] = patched_pre

    # Free intermediate dictionaries — they only served segment-to-segment
    # phase alignment and are no longer needed.  export_chunks / export_pre_chunks
    # already hold the authoritative references for the merge.
    completed_outputs.clear()
    completed_pre_refine.clear()
    completed_av_latents.clear()
    completed_av_handoff.clear()
    completed_refine_passes.clear()
    import gc as _gc
    _gc.collect()

    # Aligned to export_chunks only (skipped slots are already omitted).
    export_audios: list[dict[str, Any]] = []
    missing_audio: list[int] = []
    for seg in export_segments:
        aud = completed_audios.get(seg.index)
        if isinstance(aud, dict) and aud.get("waveform") is not None:
            export_audios.append(aud)
        else:
            export_audios.append({})
            missing_audio.append(seg.index + 1)
    if missing_audio and plan.export_mode == "all":
        reports.append(
            "Audio cache missing for segment(s) "
            f"{missing_audio} — those slots are silent in the merge. "
            "Re-run them once (or run all) to refresh audio cache."
        )

    if partial_run:
        # 「选择运行」: export only what actually ran — one clip per contiguous run,
        # stitched exactly like「连续导出」. Unselected slots were hydrated above for
        # continuity but never rejoin the output (no more full-timeline merge that
        # spliced cache/source back in).
        from .segment_cache import build_run_selection_clips

        run_clips, run_audios, run_counts, run_mp4s = build_run_selection_clips(
            node_id, plan, run_list, segment_outputs, segment_audios,
            all_segments=all_segments, mp4_run_dir=mp4_run_dir,
            workflow_name=workflow_name,
        )
        if not run_clips:
            raise ValueError("Director plan produced no segments.")
        # Route through the segments layout so the node emits one IMAGE per run,
        # matching「连续导出」, instead of a single stitched merge.
        plan.export_mode = "segments"
        segment_outputs = run_clips
        segment_audios = run_audios
        export_frame_counts = run_counts
        combined = run_clips[0] if len(run_clips) == 1 else run_clips[0]
        # Pre-refine companion: re-stitch the corresponding pre-refine runs.
        pre_source = segment_pre_refine if segment_pre_refine else list(segment_outputs)
        pre_runs, _, _, _ = build_run_selection_clips(
            node_id, plan, run_list, pre_source, None,
            all_segments=all_segments, workflow_name=workflow_name,
        )
        pre_combined = pre_runs[0] if pre_runs else combined
        if run_mp4s:
            reports.append(
                "选择运行导出: " + ", ".join(f"#{p.split('seg_')[-1]}" for p in run_mp4s)
            )
        else:
            reports.append(
                "选择运行导出: "
                + ", ".join(f"#{run[0] + 1}-{run[-1] + 1}" for run in continuous_export_runs(run_list))
            )
        # Advance the cache generation now that fresh renders exist (see below).
        if not held_for_confirmation:
            try:
                sync_segment_slots(node_id, plan, workflow_name=workflow_name, gc=True)
            except Exception as exc:  # pragma: no cover - GC is best-effort
                log.warning("Segment cache cleanup after run skipped (%s).", exc)
        return (
            combined,
            segment_outputs,
            segment_audios,
            "\n".join(reports),
            export_frame_counts,
            pre_combined,
            segment_pre_refine,
            held_for_confirmation,
        )
    # segment_outputs path (分段导出 / image batch): keep run-order audios.
    if plan.export_mode == "all" and output_chunks:
        segment_audios = export_audios
        export_frame_counts = [int(c.shape[0]) for c in export_chunks]
    else:
        segment_audios = [
            completed_audios.get(idx) or (segment_audios[pos] if pos < len(segment_audios) else {})
            for pos, idx in enumerate(run_list)
        ]
        export_frame_counts = [int(t.shape[0]) for t in segment_outputs]
    # Free completed_audios — export_audios / segment_audios now hold what we need.
    completed_audios.clear()
    # --- Main merge: lazy-load from disk (peak ≈ result + one chunk) ---
    # Segment cache reads fall back to a stale render, matching the pick order
    # used to build export_segments — otherwise a「选择运行」+「全部导出」merge
    # fails on the very segment it just accepted. Passthrough fills come from
    # merge_overrides because they were never written to disk.
    combined = concat_chunks_lazy(
        node_id, plan, export_segments, overrides=merge_overrides,
        workflow_name=workflow_name,
    )
    merge_overrides.clear()
    # Free all in-memory chunk lists — merge read from disk.
    export_chunks.clear()
    # --- Pre-refine merge: must use in-memory chunks (different data) ---
    pre_source = export_pre_chunks if export_pre_chunks else segment_pre_refine
    if not pre_source:
        pre_source = list(segment_outputs)
    pre_combined = concat_continuous_chunks(pre_source, export_segments, plan)
    export_pre_chunks.clear()
    # Now — and only now — advance the cache generation. The run has just
    # written fresh renders, so the file groups it superseded are expendable.
    # The pre-run sync above and every read-only HTTP probe deliberately leave
    # them alone, otherwise re-wording a prompt would delete the last render
    # before a replacement exists.
    if not held_for_confirmation:
        try:
            sync_segment_slots(node_id, plan, workflow_name=workflow_name, gc=True)
        except Exception as exc:  # pragma: no cover - GC is best-effort
            log.warning("Segment cache cleanup after run skipped (%s).", exc)
    return (
        combined,
        segment_outputs,
        segment_audios,
        "\n".join(reports),
        export_frame_counts,
        pre_combined,
        segment_pre_refine,
        held_for_confirmation,
    )
