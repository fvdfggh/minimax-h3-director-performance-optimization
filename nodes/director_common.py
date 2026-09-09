"""Shared helpers for the MiniMax H3 Director Opt timeline node."""

from __future__ import annotations

import json
import logging

import torch

from ..director.audio_export import (
    AUDIO_MODE_GENERATE,
    build_director_audio_outputs,
    resolve_audio_mode,
    source_audio_report_note,
)
from ..director.frame_align import pad_or_trim_frames
from ..director.gen_timeline import is_prompt_batch_timeline, is_video_batch_task_key
from ..director.plan import build_director_plan, count_all_timeline_segments, count_timeline_segments, plan_summary, _parse_second_sample
from ..director.progress import report_director_planning
from ..lib.image_prep import fit_canvas, fit_video_long_edge
from ..lib.video_io import load_timeline_segment
from ..lib.task_prompts import task_type_combo_options

log = logging.getLogger("ComfyUI-MiniMaxH3-Director")

# ---------------------------------------------------------------------------
# Fixed behaviour — these were node widgets and are no longer editable from the
# UI. Values match the previous widget defaults. Change them here to change the
# behaviour; restart ComfyUI afterwards.
# ---------------------------------------------------------------------------
#: Reuse CLIP/VAE conditioning cached on disk across runs (same prompt/canvas).
#: The on-disk key is a fingerprint of prompt/canvas/model, so a changed prompt
#: simply produces a new entry — a stale hit can only happen if you revert to a
#: previously-used prompt, in which case use the node's 清空缓存 button.
USE_CONDITIONING_CACHE = True
#: Unload models + empty the CUDA cache after each segment.
CLEAR_VRAM_BETWEEN_SEGMENTS = True
#: Decode the timeline source clip onto the separate ``source_images`` output.
EXPORT_SOURCE_IMAGES = False


def timeline_required_inputs() -> dict:
    """Timeline + prompt widgets shared by Director nodes."""
    combo_options, combo_meta = task_type_combo_options()
    return {
        "task_type": (combo_options, combo_meta),
        "global_prompt": (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "tooltip": "Synced from in-node UI (global mode).",
            },
        ),
        "bd_grp_sample": ("BDGROUP", {"default": "采样设置"}),
        "cfg": (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "CFG for KSampler."},
        ),
        "seed": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 0xFFFFFFFFFFFFFFFF,
                "control_after_generate": True,
                "tooltip": "Random seed for sampling.",
            },
        ),
        "frame_rate": (
            "FLOAT",
            {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01, "tooltip": "Timeline / output FPS (H3 trained at 24)."},
        ),
        "width": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
        "height": ("INT", {"default": 480, "min": 32, "max": 8192, "step": 32}),
        "ref_max_size": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
        "total_frames": (
            "INT",
            {
                "default": 124,
                "min": 5,
                "max": 100000,
                "tooltip": "Timeline total frames (fl2v = sum of shots). Per-shot generation still capped near 512.",
            },
        ),
        "timeline_data": (
            "STRING",
            {"default": "", "multiline": True, "tooltip": "Internal — video, segments, refs (populated by UI)."},
        ),
    }


def director_perf_inputs() -> dict:
    """Performance widgets shared by Director nodes.

    The behaviour switches that used to live here (``use_conditioning_cache``,
    ``batch_mode``, ``clear_vram_between_segments``, ``export_source_images``)
    are fixed constants now — see the top of this module.
    """
    return {
        "workflow_name": (
            "STRING",
            {
                "default": "",
                "hidden": True,
                "tooltip": (
                    "只读：前端自动填入当前工作流名称。文本缓存与 batch 中间缓存"
                    "按工作流名分目录存放，同名工作流的缓存互不干扰。"
                    "清除缓存请使用节点上的「清空缓存」按钮。"
                ),
            },
        ),
    }


def default_timeline_json(
    *,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
) -> str:
    return json.dumps(
        {
            "version": 4,
            "editMode": "global",
            "totalFrames": total_frames,
            "frameRate": frame_rate,
            "width": width,
            "height": height,
            "refMaxSize": ref_max_size,
            "output": {
                "mode": "fixed",
                "longEdge": ref_max_size,
                "width": width,
                "height": height,
                "maxExportFrames": 0,
                "exportMode": "all",
                "audioMode": "generate",
                "refImageSize": "match",
            },
            "videoClips": [],
            "video": {
                "fileName": "",
                "videoFile": "",
                "subfolder": "",
                "type": "input",
                "frames": [],
                "frameMap": [],
            },
            "global": {"taskType": task_type, "prompt": global_prompt, "refs": [], "referenceVideo": {}, "continuousReference": False},
            "segments": [
                {
                    "id": "s0",
                    "start": 0,
                    "length": total_frames,
                    "prompt": "",
                    "taskType": "",
                    "refs": [],
                    "referenceVideo": {},
                }
            ],
        },
        ensure_ascii=False,
    )


def prepare_director_plan(
    *,
    timeline_data: str,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
    unique_id: str | None,
    i2v_groups=None,
    r2v_groups=None,
):
    from ..director.external_groups import (
        build_plan_from_external_groups,
        validate_external_group_inputs,
    )

    if not timeline_data or not timeline_data.strip():
        timeline_data = default_timeline_json(
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )

    task_key, ext_groups, family = validate_external_group_inputs(
        task_type=task_type,
        i2v_groups=i2v_groups,
        r2v_groups=r2v_groups,
    )

    if ext_groups is not None and family is not None:
        report_director_planning(
            unique_id,
            len(ext_groups),
            timeline_segment_total=len(ext_groups),
        )
        plan = build_plan_from_external_groups(
            ext_groups,
            family=family,
            timeline_data=timeline_data,
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )
        _attach_segment_export(plan, timeline_data)
        _attach_second_sample(plan, timeline_data)
        log.info(
            "MiniMax H3 Director Opt: external %s groups × %d (task=%s) | %s",
            family,
            len(ext_groups),
            task_key,
            plan_summary(plan).replace("\n", " | "),
        )
        return plan

    report_director_planning(
        unique_id,
        count_timeline_segments(timeline_data),
        timeline_segment_total=count_all_timeline_segments(timeline_data),
    )

    plan = build_director_plan(
        timeline_data,
        global_task_type=task_type,
        global_prompt=global_prompt,
        total_frames=total_frames,
        frame_rate=frame_rate,
        width=width,
        height=height,
        ref_max_size=ref_max_size,
    )
    _attach_segment_export(plan, timeline_data)
    _attach_second_sample(plan, timeline_data)
    log.info(plan_summary(plan).replace("\n", " | "))
    return plan


def _attach_segment_export(plan, timeline_data: str) -> None:
    """Stamp the「分段导出」request onto a finished plan.

    Applied here rather than inside each timeline builder (gen / fl2v / external
    groups) so there is exactly one place that has both the parsed timeline and
    the final segment list. Never raises — a malformed block just means the
    feature stays off and the node runs normally.
    """
    from ..director.plan import _parse_segment_export

    try:
        timeline = json.loads(timeline_data) if timeline_data and timeline_data.strip() else {}
    except json.JSONDecodeError:
        return
    if not isinstance(timeline, dict):
        return
    try:
        plan.segment_export = _parse_segment_export(timeline, len(plan.segments))
        _seg_export = plan.segment_export
        log.info(
            "MiniMax H3 Director Opt 分段导出 parsed: enabled=%s mode=%s source=%s indices=%s nseg=%d",
            _seg_export.enabled if _seg_export else None,
            _seg_export.mode if _seg_export else None,
            _seg_export.normalized_source() if _seg_export else None,
            _seg_export.indices if _seg_export else None,
            len(plan.segments),
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("MiniMax H3 Director Opt: 分段导出 config ignored (%s)", exc)
        plan.segment_export = None


def _attach_second_sample(plan, timeline_data: str) -> None:
    """Stamp the「二次采样」request onto a finished plan (mirrors _attach_segment_export).

    Reads ``timeline.output.secondSample`` (or the top-level ``secondSample`` the
    picker writes) and stores a :class:`SegmentSecondSampleRequest` on the plan.
    Never raises — a malformed block just means the feature stays off and the node
    runs a normal first pass.
    """
    try:
        timeline = json.loads(timeline_data) if timeline_data and timeline_data.strip() else {}
    except json.JSONDecodeError:
        return
    if not isinstance(timeline, dict):
        return
    try:
        plan.second_sample = _parse_second_sample(timeline, len(plan.segments))
        _ss = plan.second_sample
        log.info(
            "MiniMax H3 Director Opt 二次采样 parsed: enabled=%s indices=%s nseg=%d",
            _ss.enabled if _ss else None,
            _ss.indices if _ss else None,
            len(plan.segments),
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("MiniMax H3 Director Opt: 二次采样 config ignored (%s)", exc)
        plan.second_sample = None


def _fit_source_clip_to_plan(plan, raw_clip: torch.Tensor) -> torch.Tensor:
    if plan.output_mode == "fixed":
        return fit_canvas(raw_clip, plan.width, plan.height)
    return fit_video_long_edge(raw_clip, plan.ref_max_size)


def build_source_images_output(
    plan,
    images_out: list[torch.Tensor],
    *,
    split_outputs: bool,
) -> list[torch.Tensor]:
    if split_outputs:
        chunks: list[torch.Tensor] = []
        # images_out is run-order (选择运行), not full timeline order.
        segs = plan.segments
        run_idx = getattr(plan, "run_indices", None)
        if run_idx is not None:
            segs = [
                plan.segments[i]
                for i in sorted(run_idx)
                if 0 <= i < len(plan.segments)
            ]
        for seg, generated in zip(segs, images_out):
            target_len = int(generated.shape[0])
            raw = load_timeline_segment(plan.raw, seg.start_frame, seg.end_frame)
            fitted = _fit_source_clip_to_plan(plan, raw)
            chunks.append(pad_or_trim_frames(fitted, target_len).cpu().float())
        return chunks

    target_len = int(images_out[0].shape[0]) if images_out else int(plan.total_frames or 0)
    raw = load_timeline_segment(plan.raw, 0, target_len)
    fitted = _fit_source_clip_to_plan(plan, raw)
    return [pad_or_trim_frames(fitted, target_len).cpu().float()]


def _empty_source_images_for(images_out: list[torch.Tensor]) -> list[torch.Tensor]:
    if not images_out:
        return [torch.full((1, 1, 1, 3), 0.5)]
    placeholders: list[torch.Tensor] = []
    for img in images_out:
        if isinstance(img, torch.Tensor) and img.ndim == 4:
            h, w, c = int(img.shape[1]), int(img.shape[2]), int(img.shape[3])
        else:
            h, w, c = 1, 1, 3
        placeholders.append(torch.full((1, h, w, c), 0.5))
    return placeholders


def _ensure_nonempty_image_batches(
    images_out: list[torch.Tensor],
    *,
    label: str,
    fallback: tuple[int, int, int] | None = None,
) -> list[torch.Tensor]:
    # A completely empty list is illegal as an IMAGE output — ComfyUI slices every
    # list input per batch item and an empty list makes the downstream node's
    # ``slice_dict`` index out of range. This happens on「分段导出」of a segment
    # that has only a latent (no decoded frames): nothing is exportable, so the
    # split layout yields zero clips. Emit a single neutral placeholder instead
    # of failing the whole graph.
    if not images_out:
        if fallback is None:
            fallback = (1, 1, 3)
        h, w, c = int(fallback[0]), int(fallback[1]), int(fallback[2])
        log.warning(
            "Director %s output is empty (no exportable segments); "
            "emitting a 1-frame neutral placeholder.",
            label,
        )
        return [torch.full((1, max(1, h), max(1, w), max(1, c)), 0.5)]
    fixed: list[torch.Tensor] = []
    for i, img in enumerate(images_out):
        if not isinstance(img, torch.Tensor) or img.ndim != 4:
            raise ValueError(f"Director {label}[{i}] is not a valid IMAGE tensor.")
        if int(img.shape[0]) <= 0:
            h, w, c = int(img.shape[1]), int(img.shape[2]), int(img.shape[3])
            log.warning("Director %s[%d] has 0 frames; emitting 1-frame placeholder.", label, i)
            fixed.append(torch.full((1, max(1, h), max(1, w), max(1, c)), 0.5))
        else:
            fixed.append(img)
    return fixed


def _layout_image_batches(
    plan,
    combined,
    segment_outputs,
    *,
    export_segments: bool,
    is_batch: bool,
    video_batch: bool,
    segment_frame_counts: list[int] | None = None,
) -> tuple[list[torch.Tensor], int]:
    if export_segments or (is_batch and not video_batch):
        images_out = segment_outputs
        frame_count = sum(int(s.shape[0]) for s in segment_outputs)
        # Batch mode drops the per-segment frames once the merge owns them; the
        # counts still describe the merged clip, so keep the report accurate.
        if not segment_outputs and segment_frame_counts:
            frame_count = sum(int(n) for n in segment_frame_counts)
        return images_out, frame_count
    combined = pad_or_trim_frames(combined, plan.total_frames).cpu().float()
    return [combined], int(combined.shape[0])


def finalize_director_outputs(
    plan,
    combined,
    segment_outputs,
    report,
    *,
    export_source_images: bool = False,
    segment_audios: list | None = None,
    segment_frame_counts: list[int] | None = None,
):
    is_batch = is_prompt_batch_timeline(plan.raw, plan.global_task_key)
    export_segments = plan.export_mode == "segments"
    video_batch = is_video_batch_task_key(plan.global_task_key)
    split_layout = export_segments or (is_batch and not video_batch)

    images_out, frame_count = _layout_image_batches(
        plan,
        combined,
        segment_outputs,
        export_segments=export_segments,
        is_batch=is_batch,
        video_batch=video_batch,
        segment_frame_counts=segment_frame_counts,
    )
    # 「分段导出」of latent-only segments yields zero clips on the split layout.
    # An empty images list would make the downstream node's slice_dict index out
    # of range AND would propagate an empty AUDIO list below. Make it non-empty
    # up front so every consumer (audio / source) sees a valid list.
    if not images_out:
        _fb_h = int(getattr(plan, "height", 0) or 0)
        _fb_w = int(getattr(plan, "width", 0) or 0)
        _fb = (_fb_h, _fb_w, 3) if _fb_h > 0 and _fb_w > 0 else None
        images_out = _ensure_nonempty_image_batches([], label="images", fallback=_fb)
        log.warning(
            "Director: no exportable segment frames; emitted a neutral placeholder "
            "for the images/audio outputs."
        )
    if export_segments and len(segment_outputs) > 1:
        report = (
            report
            + f"\n\nExport mode: segments — {len(segment_outputs)} clip(s) on images output."
        )
    if plan.run_indices is not None and split_layout:
        report = (
            report
            + f"\n\nPartial run: output contains {len(segment_outputs)} re-generated clip(s) only."
        )
    if not split_layout:
        if video_batch and is_batch and len(segment_outputs) > 1:
            report = report + f"\n\nExport mode: all — merged {frame_count} frame(s) on images output."
        if plan.run_indices is not None and video_batch:
            report = report + f"\n\nPartial run: re-generated {len(segment_outputs)} video group(s)."

    split_for_audio = split_layout
    audio_frame_end = frame_count if not split_for_audio else None
    audio_mode = resolve_audio_mode(plan)
    use_generated = audio_mode == AUDIO_MODE_GENERATE
    # Prefer caller-provided export lengths (post continuity trim); else match IMAGE batches.
    if segment_frame_counts is None and segment_audios and split_for_audio:
        segment_frame_counts = [int(s.shape[0]) for s in segment_outputs]
    audio_out, source_fallback = build_director_audio_outputs(
        plan,
        images_out,
        export_segments=split_for_audio,
        output_frame_end=audio_frame_end,
        segment_audios=segment_audios if use_generated else None,
        segment_frame_counts=segment_frame_counts if use_generated else None,
        audio_mode=audio_mode,
    )
    report = report + source_audio_report_note(
        plan,
        audio_out,
        export_segments=split_for_audio,
        output_frame_end=audio_frame_end,
        used_generated_audio=bool(use_generated and segment_audios),
        audio_mode=audio_mode,
        source_fallback=source_fallback,
    )

    # Merged layout: ``images_out`` is the merged clip and every length has been
    # captured above, so the per-segment frames are a second full copy of the
    # same video. Drop them before returning — nothing below reads them again.
    # (Split layout must NOT do this: there ``images_out`` *is* this list.)
    if not split_layout:
        del segment_outputs[:]

    split_source_outputs = export_segments or (is_batch and not video_batch)
    if export_source_images:
        try:
            source_images_out = build_source_images_output(
                plan,
                images_out,
                split_outputs=split_source_outputs,
            )
            source_frames = sum(int(batch.shape[0]) for batch in source_images_out)
            report = report + (
                f"\n\nSource images: decoded {source_frames} timeline frame(s) "
                f"on {len(source_images_out)} source_images batch(es)."
            )
        except Exception as exc:
            log.warning("Source images output failed: %s", exc)
            # Never disguise generated frames as the source comparison. A neutral
            # placeholder makes the failure visible while preserving the expensive run.
            source_images_out = _empty_source_images_for(images_out)
            report = report + (
                "\n\nSource images: FAILED — emitted neutral placeholder(s), not generated "
                f"frames. Check the timeline source path/decode ({type(exc).__name__}: {exc})."
            )
    else:
        source_images_out = _empty_source_images_for(images_out)

    fb_h = int(getattr(plan, "height", 0) or 0)
    fb_w = int(getattr(plan, "width", 0) or 0)
    fb = (fb_h, fb_w, 3) if fb_h > 0 and fb_w > 0 else None
    images_out = _ensure_nonempty_image_batches(images_out, label="images", fallback=fb)
    source_images_out = _ensure_nonempty_image_batches(source_images_out, label="source_images", fallback=fb)

    report = report + "\n\n有问题联系作者：AI搅拌手  QQ交流群：551482703"

    fps_out = float(plan.frame_rate or 24.0)
    return images_out, audio_out, fps_out, frame_count, source_images_out, report
