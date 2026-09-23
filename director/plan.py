"""Building a Director plan from a timeline, plus the plan facade.

``build_director_plan`` turns the timeline JSON into a :class:`DirectorPlan` — one
:class:`SegmentPlan` per group, each carrying its prompt, task mode, frame budget,
seam settings and reference materials. ``plan_summary`` renders the human-readable
breakdown the node reports.

The rest of the timeline logic was split by concern; this module re-exports it so
existing callers keep working (see ``__all__``):

* :mod:`plan_refs`    — loading reference materials, flattening them to node kwargs
* :mod:`plan_ranges`  — export / second-sample windows, counting, clip trimming
* :mod:`plan_prompts` — r2v / rv2v / v2v prompt reinforcement
* :mod:`plan_types`   — the plan dataclasses and request parsers

New code should import from those modules directly.
"""

from __future__ import annotations

import json
import logging

import torch

from ..lib.constants import FALLBACK_LONG_EDGE
from ..lib.image_prep import assert_minimax_canvas, resolve_output_dimensions
from ..lib.task_prompts import get_task_prompt_spec, resolve_task_key
from ..lib.video_io import (
    load_timeline_segment,
    logical_frame_count,
    logical_frame_map,
    video_clips_from_timeline,
)
from .gen_timeline import build_gen_director_plan, is_gen_timeline
from .plan_prompts import (
    reinforce_r2v_prompt,
    reinforce_rv2v_prompt,
    reinforce_v2v_prompt,
)
from .plan_ranges import (
    _clip_segment_ranges,
    _resolve_export_total,
    _segment_ranges_from_timeline,
    _trim_timeline_for_export,
    count_all_timeline_segments,
    count_timeline_segments,
    parse_second_sample,
    parse_segment_export,
    prepare_segment_clip,
)
from .plan_refs import (
    _continuous_reference_enabled,
    _load_ref_audios,
    _load_ref_videos,
    _load_refs,
    _ref_video_has_file,
    _resolve_global_reference_video,
    load_reference_tensor,
    ref_audios_to_dict,
    ref_video_audios_to_dict,
    ref_videos_to_dict,
    reference_video_for_segment,
    refs_to_kwargs_for_context,
    segment_ref_audios_for_context,
    segment_refs_for_context,
)
from .plan_types import (
    DirectorPlan,
    SegmentPlan,
    parse_run_selection,
    resolve_export_mode,
    resolve_ref_image_size,
)

__all__ = [
    # Built here.
    "build_director_plan",
    "plan_summary",
    # Re-exported: reference materials.
    "_load_ref_audios",
    "_load_ref_videos",
    "_load_refs",
    "load_reference_tensor",
    "ref_audios_to_dict",
    "ref_videos_to_dict",
    "ref_video_audios_to_dict",
    "refs_to_kwargs_for_context",
    "segment_ref_audios_for_context",
    "segment_refs_for_context",
    "reference_video_for_segment",
    # Re-exported: ranges / counting.
    "count_all_timeline_segments",
    "count_timeline_segments",
    "parse_segment_export",
    "parse_second_sample",
    "prepare_segment_clip",
    # Re-exported: prompt reinforcement.
    "reinforce_r2v_prompt",
    "reinforce_rv2v_prompt",
    "reinforce_v2v_prompt",
]

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director")


def build_director_plan(
    timeline_data: str,
    *,
    global_task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
) -> DirectorPlan:
    timeline: dict = {}
    if timeline_data and timeline_data.strip():
        try:
            timeline = json.loads(timeline_data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid timeline_data JSON: {exc}") from exc

    global_block = timeline.get("global") or {}
    edit_mode = timeline.get("editMode") or timeline.get("edit_mode") or "global"
    if edit_mode not in ("global", "segment"):
        edit_mode = "global"

    task_type = global_block.get("taskType") or global_task_type or "v2v — 视频转视频(Video to Video)"
    prompt = global_block.get("prompt") or global_prompt or ""
    global_refs = _load_refs(global_block.get("refs") or [])
    global_ref_audios = _load_ref_audios(
        global_block.get("refAudios") or global_block.get("ref_audios") or []
    )
    global_ref_video = _resolve_global_reference_video(timeline)

    task_key_early = resolve_task_key(task_type)
    if task_key_early == "fl2v" or str(timeline.get("timelineMode") or "").lower() == "fl2v":
        from .fl2v_timeline import build_fl2v_director_plan

        return build_fl2v_director_plan(
            timeline,
            global_task_type=task_type,
            global_prompt=prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )
    if is_gen_timeline(timeline, task_key_early):
        return build_gen_director_plan(
            timeline,
            global_task_type=task_type,
            global_prompt=prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )

    frame_map = logical_frame_map(timeline)
    source_total = logical_frame_count(timeline) or int(timeline.get("totalFrames") or total_frames or 0)
    export_max = int(
        (timeline.get("output") or {}).get("maxExportFrames")
        or (timeline.get("output") or {}).get("max_export_frames")
        or 0
    )
    export_total = _resolve_export_total(timeline, source_total)

    load_timeline = _trim_timeline_for_export(timeline, export_total) if export_total < source_total else timeline

    clips = video_clips_from_timeline(load_timeline)
    if not clips and not (load_timeline.get("video") or {}).get("frames"):
        raise ValueError(
            "No source video in MiniMax H3 Director Opt. Upload a video inside the node timeline UI before running."
        )

    try:
        probe = load_timeline_segment(load_timeline, 0, 1)
        loaded_h = int(probe.shape[1])
        loaded_w = int(probe.shape[2])
    except Exception as exc:
        log.warning("Could not probe source video frame: %s", exc)
        video_meta = load_timeline.get("video") or {}
        loaded_w = int(video_meta.get("width") or width)
        loaded_h = int(video_meta.get("height") or height)

    source_video = torch.zeros(0, max(1, loaded_h), max(1, loaded_w), 3)
    video_meta = timeline.get("video") or {}
    meta_w = int(video_meta.get("width") or 0)
    meta_h = int(video_meta.get("height") or 0)

    output_block = timeline.get("output") or {}
    export_mode = resolve_export_mode(output_block)
    out_w, out_h, ref_max, output_mode = resolve_output_dimensions(
        loaded_w or meta_w or int(width),
        loaded_h or meta_h or int(height),
        mode=str(output_block.get("mode") or "long_edge"),
        long_edge=int(output_block.get("longEdge") or output_block.get("long_edge") or ref_max_size or FALLBACK_LONG_EDGE),
        fixed_width=int(output_block.get("width") or timeline.get("width") or width),
        fixed_height=int(output_block.get("height") or timeline.get("height") or height),
    )
    assert_minimax_canvas(out_w, out_h)

    total = int(load_timeline.get("totalFrames") or export_total or total_frames or 0)
    if total <= 0:
        total = source_total

    segment_ranges = _segment_ranges_from_timeline(timeline, source_total or total)
    segment_ranges = _clip_segment_ranges(segment_ranges, total)
    segments: list[SegmentPlan] = []
    continuous_ref = _continuous_reference_enabled(timeline, edit_mode, resolve_task_key(task_type))

    for idx, (start, end, seg_data) in enumerate(segment_ranges):
        if edit_mode == "global":
            seg_prompt = prompt
            seg_task = task_type
            seg_refs = list(global_refs)
            seg_ref_audios = list(global_ref_audios)
            seg_ref_video = dict(global_ref_video)
            use_global = True
        else:
            use_global = False
            seg_prompt = (seg_data.get("prompt") or "").strip() or prompt
            seg_task = seg_data.get("taskType") or seg_data.get("task_type") or task_type
            # Segment mode: only this segment's refs — never inherit global.refs / refAudios.
            seg_refs = _load_refs(seg_data.get("refs") or [])
            seg_ref_audios = _load_ref_audios(
                seg_data.get("refAudios") or seg_data.get("ref_audios") or []
            )
            seg_ref_video = dict(seg_data.get("referenceVideo") or seg_data.get("reference_video") or {})

        seg_task_key = resolve_task_key(seg_task)
        seg_refs = segment_refs_for_context(seg_task_key, seg_refs)
        seg_ref_audios = segment_ref_audios_for_context(seg_task_key, seg_ref_audios)
        ref_start = start if continuous_ref and seg_task_key == "ads2v" else 0

        segments.append(
            SegmentPlan(
                index=idx,
                start_frame=start,
                end_frame=end,
                prompt=seg_prompt,
                task_type=seg_task,
                task_key=seg_task_key,
                use_global=use_global,
                refs=seg_refs,
                ref_audios=seg_ref_audios,
                reference_video_meta=seg_ref_video,
                reference_video_start_frame=ref_start,
            )
        )

    for seg in segments:
        if seg.task_key != "ads2v":
            continue
        if _ref_video_has_file(seg.reference_video_meta):
            continue
        raise ValueError(
            f"ads2v (广告植入) segment #{seg.index + 1} requires a reference video. "
            "Upload the content-to-insert clip for this segment in the Director node UI."
        )

    from .segment_continuity import (
        resolve_continuity_settings,
        resolve_continuity_redraw,
        resolve_segment_continuity_from_prev,
    )

    continuity_enabled, continuity_overlap = resolve_continuity_settings(
        timeline, segment_count=len(segments)
    )
    continuity_redraw = resolve_continuity_redraw(timeline)
    for seg, (_start, _end, seg_data) in zip(segments, segment_ranges):
        seg.continuity_from_prev = resolve_segment_continuity_from_prev(
            seg_data if isinstance(seg_data, dict) else {},
            segment_index=seg.index,
        )
        seg.ref_image_size = resolve_ref_image_size(
            seg_data if isinstance(seg_data, dict) else {},
            load_timeline,
        )

    return DirectorPlan(
        frame_rate=float(timeline.get("frameRate") or frame_rate or 24),
        total_frames=total,
        width=out_w,
        height=out_h,
        ref_max_size=ref_max,
        output_mode=output_mode,
        source_width=int(meta_w or loaded_w),
        source_height=int(meta_h or loaded_h),
        global_task_type=task_type,
        global_task_key=resolve_task_key(task_type),
        global_prompt=prompt,
        global_refs=global_refs,
        segments=segments,
        source_video=source_video,
        edit_mode=edit_mode,
        raw=load_timeline,
        source_total_frames=source_total or total,
        export_max_frames=export_max,
        export_mode=export_mode,
        run_indices=parse_run_selection(timeline, len(segments)),
        continuity_enabled=continuity_enabled,
        continuity_overlap_frames=continuity_overlap,
        continuity_redraw=continuity_redraw,
        global_ref_audios=global_ref_audios,
    )


def plan_summary(plan: DirectorPlan) -> str:
    mode = str(plan.raw.get("timelineMode") or "")
    if mode in ("gen_blank", "gen_image", "prompt_batch", "image_batch", "fl2v"):
        if mode == "fl2v":
            mode_label = "首尾帧 (fl2v)"
        elif mode in ("prompt_batch", "image_batch"):
            mode_label = f"批量生成 ({plan.global_task_key})"
        else:
            mode_label = "空白画布" if mode == "gen_blank" else "图片生成"
        lines = [
            f"MiniMax H3 Director Opt [{mode_label}] ({plan.edit_mode}): "
            f"{plan.segment_count} segment(s), {plan.total_frames} frames @ {plan.frame_rate:.2f} fps",
            f"Output: {plan.width}×{plan.height} ({plan.output_mode})",
            f"Global task: {get_task_prompt_spec(plan.global_task_type).label}",
        ]
        if plan.continuity_enabled:
            pinned = [
                seg.index + 1
                for seg in plan.segments
                if seg.index > 0 and getattr(seg, "continuity_from_prev", True)
            ]
            skipped_pin = [
                seg.index + 1
                for seg in plan.segments
                if seg.index > 0 and not getattr(seg, "continuity_from_prev", True)
            ]
            lines.append(
                f"Segment continuity: ON (motion context {plan.continuity_overlap_frames}f)"
            )
            if pinned:
                lines.append("  Pin from prev: #" + ", #".join(str(i) for i in pinned))
            if skipped_pin:
                lines.append(
                    "  Hard cut (per-segment off): #"
                    + ", #".join(str(i) for i in skipped_pin)
                )
        for seg in plan.segments:
            pin_note = ""
            if plan.continuity_enabled and seg.index > 0:
                pin_note = (
                    " — pin←prev"
                    if getattr(seg, "continuity_from_prev", True)
                    else " — hard cut"
                )
            lines.append(
                f"  #{seg.index + 1} [{seg.start_frame}:{seg.end_frame}] "
                f"{seg.frame_count}f — {seg.task_key}{pin_note} — "
                f"{seg.prompt[:60]}{'…' if len(seg.prompt) > 60 else ''}"
            )
        return "\n".join(lines)

    mode_label = (
        f"视频编辑 ({plan.global_task_key})"
        if plan.global_task_key in {"v2v", "rv2v"}
        else "源视频时间轴"
    )
    lines = [
        f"MiniMax H3 Director Opt [{mode_label}] ({plan.edit_mode}): {plan.segment_count} segment(s), "
        f"{plan.total_frames} frames @ {plan.frame_rate:.2f} fps",
    ]
    if plan.export_max_frames > 0 and plan.source_total_frames > plan.total_frames:
        lines.append(
            f"Export cap: {plan.total_frames}/{plan.source_total_frames} frames "
            f"(max {plan.export_max_frames})"
        )
    export_label = "分段导出" if plan.export_mode == "segments" else "全部导出"
    lines.append(f"Export mode: {export_label}")
    if plan.continuity_enabled:
        pinned = [
            seg.index + 1
            for seg in plan.segments
            if seg.index > 0 and getattr(seg, "continuity_from_prev", True)
        ]
        skipped_pin = [
            seg.index + 1
            for seg in plan.segments
            if seg.index > 0 and not getattr(seg, "continuity_from_prev", True)
        ]
        lines.append(
            f"Segment continuity: ON (motion context {plan.continuity_overlap_frames}f "
            "→ pin previous tail + trim prefix; t2v/i2v/fl2v/r2v/v2v/rv2v)"
        )
        if pinned:
            lines.append(
                "  Pin from prev: #"
                + ", #".join(str(i) for i in pinned)
            )
        if skipped_pin:
            lines.append(
                "  Hard cut (per-segment off): #"
                + ", #".join(str(i) for i in skipped_pin)
            )
    elif plan.segment_count >= 2 and plan.global_task_key in {
        "t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v",
    }:
        lines.append(
            "Segment continuity: OFF — hard cuts between segments "
            "(enable「段间引导」in Director UI; recommend 22 frames)"
        )
    else:
        lines.append("Segment continuity: OFF (per-segment generation)")
    if plan.run_indices is not None:
        selected = sorted(plan.run_indices)
        skipped = [i + 1 for i in range(plan.segment_count) if i not in plan.run_indices]
        lines.append(
            f"Run selection: {len(selected)}/{plan.segment_count} segment(s) "
            f"(#{', #'.join(str(i + 1) for i in selected)}; skipped #{', #'.join(map(str, skipped)) or 'none'})"
        )
    lines.append(f"Global task: {get_task_prompt_spec(plan.global_task_type).label}")
    for seg in plan.segments:
        pin_note = ""
        if plan.continuity_enabled and seg.index > 0:
            pin_note = (
                " — pin←prev"
                if getattr(seg, "continuity_from_prev", True)
                else " — hard cut"
            )
        lines.append(
            f"  #{seg.index + 1} [{seg.start_frame}:{seg.end_frame}] "
            f"{seg.frame_count}f — {seg.task_key}{pin_note} — "
            f"{seg.prompt[:60]}{'…' if len(seg.prompt) > 60 else ''}"
        )
    return "\n".join(lines)
