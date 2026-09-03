"""MiniMax H3 Director — timeline UI + official MiniMax H3 AV execution."""

from __future__ import annotations

import logging

import comfy.samplers

from ..director.executor_core import execute_director_plan_core
from .director_common import (
    finalize_director_outputs,
    prepare_director_plan,
    timeline_required_inputs,
    director_perf_inputs,
)

_CATEGORY = "MiniMaxH3"

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.nodes")

_DEFAULT_GLOBAL_PROMPT = "A cinematic scene with natural motion and synchronized ambience"


def director_timeline_required_inputs() -> dict:
    """Timeline widgets — defaults aligned with official MiniMax H3 workflow templates."""
    inputs = timeline_required_inputs()
    combo_options, combo_meta = inputs["task_type"]

    gp_meta = dict(inputs["global_prompt"][1])
    gp_meta["default"] = _DEFAULT_GLOBAL_PROMPT
    gp_meta["tooltip"] = (
        "User prompt — sent directly to MiniMaxH3ImageToVideo / ReferenceToVideo. "
        "r2v: <Picture 1>. v2v: source-timeline edit (<Video 1>). "
        "rv2v: source timeline + reference images (<Video 1> + <Picture N>)."
    )

    frames_meta = dict(inputs["total_frames"][1])
    frames_meta["default"] = 124
    frames_meta["tooltip"] = (
        "Frame count at 24 fps; snapped to MiniMax 17k+5 grid (124 ≈ 5s)."
    )

    return {
        **inputs,
        "task_type": (combo_options, combo_meta),
        "global_prompt": ("STRING", gp_meta),
        "total_frames": ("INT", frames_meta),
    }


class MiniMaxH3Director:
    """In-node timeline Director using ComfyUI official MiniMax H3 pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "MiniMax H3 UNET (UNETLoader)."},
                ),
                "video_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 video VAE (minimax_h3_video_vae)."},
                ),
                "audio_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 audio VAE (minimax_h3_audio_vae). Required for r2v / v2v / rv2v."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "CLIPLoader type=minimax (qwen3vl)."},
                ),
                **director_timeline_required_inputs(),
            },
            "optional": {
                "i2v_groups": (
                    "MMX_DIR_GROUP",
                    {
                        "tooltip": (
                            "External Image to Video group(s) (t2v / i2v / fl2v). "
                            "When connected, overrides UI cards for execution (external priority). "
                            "Connect Group (Image to Video).group, or Groups Combine."
                        ),
                    },
                ),
                "r2v_groups": (
                    "MMX_DIR_GROUP",
                    {
                        "tooltip": (
                            "External Reference to Video group(s). "
                            "When connected, overrides UI cards for execution (external priority). "
                            "Connect Group (Reference to Video).group, or Groups Combine."
                        ),
                    },
                ),
                "refine": (
                    "MMX_DIR_REFINE",
                    {
                        "tooltip": (
                            "Optional Refine node. When connected, each segment runs a second "
                            "sample pass (same-size refine, or upscale then sample). "
                            "Wire a MODEL into Refine.refine_model to use a different UNET for that pass; "
                            "unwired uses this Director model. "
                            "images is the refined result; images_pre_refine is the first pass. "
                            "Unconnected = single-pass (current behavior)."
                        ),
                    },
                ),
                "bd_grp_advanced": ("BDGROUP", {"default": "高级采样"}),
                "steps": (
                    "INT",
                    {
                        "default": 25,
                        "min": 1,
                        "max": 200,
                        "tooltip": "Sampling steps — official template: 25.",
                    },
                ),
                "sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": "res_multistep",
                        "tooltip": "Official template: KSamplerSelect res_multistep.",
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "default": "simple",
                        "tooltip": "Official template: BasicScheduler simple.",
                    },
                ),
                "shift_video": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "MiniMaxH3SigmaShift shift_video."},
                ),
                "shift_audio": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "MiniMaxH3SigmaShift shift_audio."},
                ),
                **director_perf_inputs(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **_kwargs):
        if input_types is not None:
            expected = {
                "model": "MODEL",
                "video_vae": "VAE",
                "audio_vae": "VAE",
                "clip": "CLIP",
            }
            for name, want in expected.items():
                got = input_types.get(name)
                if got is not None and got != want:
                    return f"{name}: expected {want}, linked node returns {got}."
        return True

    @classmethod
    def IS_CHANGED(cls, unique_id=None, workflow_name=None, **kwargs):
        # Do not return NaN: that would re-run every Director queue even when
        # confirm_first_pass is off. Linked Refine is None here, so fingerprint
        # the .pre cache files that only the confirmation hold writes.
        del kwargs
        from ..director.segment_cache import first_pass_cache_disk_signature

        return first_pass_cache_disk_signature(unique_id, workflow_name=workflow_name)

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "source_images", "report", "images_pre_refine")
    OUTPUT_IS_LIST = (True, True, False, False, True, False, True)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "MiniMax H3 Director: MiniMaxH3ImageToVideo / ReferenceToVideo conditioning, "
        "single-stage KSampler + MiniMaxH3SigmaShift, LTXVSeparateAVLatent decode. "
        "Supports t2v / i2v / fl2v / r2v / v2v / rv2v. "
        "Optional i2v_groups / r2v_groups accept multi-group packs from Director Group nodes "
        "(external priority over UI cards). Optional refine accepts MiniMax H3 Director Refine "
        "(second sample / upscale). images_pre_refine is the first-pass video before refine. "
        "Defaults: 0.4MP 16:9 (864×480), 5s / 124 frames @ 24 fps."
    )

    def execute(
        self,
        model,
        video_vae,
        audio_vae,
        clip,
        task_type,
        global_prompt,
        frame_rate,
        width,
        height,
        ref_max_size,
        total_frames,
        timeline_data,
        unique_id=None,
        i2v_groups=None,
        r2v_groups=None,
        refine=None,
        steps=25,
        sampler="res_multistep",
        scheduler="simple",
        cfg=1.0,
        seed=0,
        shift_video=12.0,
        shift_audio=3.0,
        clear_vram_between_segments=True,
        export_source_images=False,
        use_conditioning_cache=False,
        clear_conditioning_cache_on_run=False,
        batch_mode=False,
        workflow_name=None,
        **kwargs,
    ):
        del kwargs

        plan = prepare_director_plan(
            timeline_data=timeline_data,
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
            unique_id=unique_id,
            i2v_groups=i2v_groups,
            r2v_groups=r2v_groups,
            refine=refine,
        )

        if batch_mode:
            # Batch mode: three-phase execution
            from ..director.batch_executor import execute_director_batch
            combined, segment_outputs, segment_audios, report, export_frame_counts, pre_combined, pre_segments, held_for_confirmation = (
                execute_director_batch(
                    plan,
                    node_id=unique_id,
                    model=model,
                    vae=video_vae,
                    audio_vae=audio_vae,
                    clip=clip,
                    cfg=cfg,
                    seed=seed,
                    steps=steps,
                    sampler=sampler,
                    scheduler=scheduler,
                    shift_video=shift_video,
                    shift_audio=shift_audio,
                    use_conditioning_cache=use_conditioning_cache,
                    workflow_name=workflow_name,
                )
            )
        else:
            # Normal mode
            combined, segment_outputs, segment_audios, report, export_frame_counts, pre_combined, pre_segments, held_for_confirmation = (
                execute_director_plan_core(
                    plan,
                    node_id=unique_id,
                    workflow_name=workflow_name,
                    model=model,
                    vae=video_vae,
                    audio_vae=audio_vae,
                    clip=clip,
                    cfg=cfg,
                    seed=seed,
                    steps=steps,
                    sampler=sampler,
                    scheduler=scheduler,
                    shift_video=shift_video,
                    shift_audio=shift_audio,
                    clear_vram_between_segments=clear_vram_between_segments,
                    use_conditioning_cache=use_conditioning_cache,
                    clear_conditioning_cache_on_run=clear_conditioning_cache_on_run,
                )
            )

        result = finalize_director_outputs(
            plan,
            combined,
            segment_outputs,
            report,
            export_source_images=export_source_images,
            segment_audios=segment_audios,
            segment_frame_counts=export_frame_counts,
            pre_refine_combined=pre_combined,
            pre_refine_segments=pre_segments,
            block_final_images=held_for_confirmation,
        )

        # 「分段导出」不再另外写磁盘：节点 OUTPUT 已由 finalize_director_outputs
        # 通过 segment_outputs 直接输出（与「运行」一致），batch / normal 两条路径
        # 都在上面的 execute_* 中填充了 segment_outputs。这里若再调用
        # run_segment_export 会重复向 minimax_segment_export/<node_id>/ 落盘，与
        # 「不保存磁盘、节点直接输出」的需求冲突，故禁用。
        seg_export = getattr(plan, "segment_export", None)
        if seg_export is not None and seg_export.enabled and seg_export.indices:
            log.info(
                "分段导出 → 节点输出模式: %d 个片段 (mode=%s, 不写磁盘)",
                len(list(seg_export.indices)),
                seg_export.normalized_mode(),
            )

        return result
