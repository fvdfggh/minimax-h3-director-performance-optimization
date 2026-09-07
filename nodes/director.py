"""MiniMax H3 Director — timeline UI + official MiniMax H3 AV execution."""

from __future__ import annotations

import logging

import comfy.samplers

from ..director.batch_executor import execute_director_batch
from .director_common import (
    CLEAR_VRAM_BETWEEN_SEGMENTS,
    EXPORT_SOURCE_IMAGES,
    USE_CONDITIONING_CACHE,
    finalize_director_outputs,
    prepare_director_plan,
    timeline_required_inputs,
    director_perf_inputs,
)

_CATEGORY = "MiniMaxH3"

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.nodes")

_DEFAULT_GLOBAL_PROMPT = "A cinematic scene with natural motion and synchronized ambience"

# Which wired MODEL slot the run should sample with. ``model`` stays the
# required main UNET and is also the fallback when the picked slot is unwired.
RUN_MODEL_MAIN = "主模型 (model)"
RUN_MODEL_B = "备用模型 1 (model_b)"
RUN_MODEL_C = "备用模型 2 (model_c)"
RUN_MODEL_CHOICES = (RUN_MODEL_MAIN, RUN_MODEL_B, RUN_MODEL_C)


def resolve_run_model(run_model, *, model, model_b=None, model_c=None):
    """Pick the UNET for this run. Unwired / unknown slot falls back to ``model``.

    Returns ``(model, note)`` where ``note`` is a short label for the run report.
    """
    key = str(run_model or RUN_MODEL_MAIN).strip()
    if key == RUN_MODEL_B:
        picked, label = model_b, "model_b"
    elif key == RUN_MODEL_C:
        picked, label = model_c, "model_c"
    else:
        picked, label = model, "model"
    if picked is None:
        return model, f"{label} 未接线 → 已回退到主模型 model"
    return picked, label


def _sanitize_timeline(width, height, total_frames):
    """Clamp absurd timeline dimensions that a legacy widget-shift can produce.

    A workflow saved while ``run_model`` sat at the top of the widget list shifts
    every timeline value by one slot, so width/height/total_frames can become
    nonsense (e.g. a frame_rate value landing in width). That blows the token
    sequence up and makes Star7's Sol path overflow on SM75. Pull the obvious
    outliers back into a sane MiniMax H3 range so the run at least proceeds.
    """
    changed = []

    def _clamp_int(v, lo, hi, default, name):
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            return default, True
        if iv < lo or iv > hi:
            return default, True
        return iv, False

    width, c1 = _clamp_int(width, 64, 4096, 864, "width")
    if c1:
        changed.append("width")
    height, c2 = _clamp_int(height, 64, 4096, 480, "height")
    if c2:
        changed.append("height")
    total_frames, c3 = _clamp_int(total_frames, 1, 500, 124, "total_frames")
    if c3:
        changed.append("total_frames")
    if changed:
        log.warning(
            "MiniMax H3 Director: 检测到异常 timeline 参数（疑似旧版控件错位），"
            "已回退到安全默认值: %s。请重新保存工作流以固化正确参数。",
            ", ".join(changed),
        )
    return width, height, total_frames


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
                    {"tooltip": "MiniMax H3 UNET (UNETLoader)。主模型，也是备用口未接线时的兜底。"},
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
                "run_model": (
                    list(RUN_MODEL_CHOICES),
                    {
                        "default": RUN_MODEL_MAIN,
                        "tooltip": (
                            "本次运行用哪个 MODEL 口：主模型 model，或备用口 model_b / model_c。"
                            "选中的口没接线时自动回退到主模型 model。"
                        ),
                    },
                ),
                "model_b": (
                    "MODEL",
                    {
                        "tooltip": (
                            "备用 UNET 1。run_model 选「备用模型 1」时使用；"
                            "未接线则回退到主模型 model。"
                        ),
                    },
                ),
                "model_c": (
                    "MODEL",
                    {
                        "tooltip": (
                            "备用 UNET 2。run_model 选「备用模型 2」时使用；"
                            "未接线则回退到主模型 model。"
                        ),
                    },
                ),
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
                # ── 自定义采样（默认关闭，不开/不接时行为与之前完全一致）──────
                # 刻意放在 optional 末尾：新增 BOOLEAN 会占用 widgets_values 下标，
                # 插在已有控件（含隐藏的 workflow_name）之前会让旧工作流整体错位。
                "sigmas": (
                    "SIGMAS",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "自定义噪声调度：接 ComfyUI 自带 BasicScheduler 或 ManualSigmas。"
                            "仅当「使用 sigmas」开启时生效。"
                            "生效后步数 = len(sigmas) - 1，denoise 固定 1.0，"
                            "steps / scheduler 不再参与调度（采样器仍用高级采样里的 sampler）。"
                        ),
                    },
                ),
                "use_sigmas": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "是否使用 sigmas：开启后用 sigmas 口接进来的 SIGMAS 作为噪声调度。"
                            "生效时步数 = len(sigmas) - 1，denoise 固定 1.0，"
                            "steps / scheduler 不再参与调度（采样器仍用高级采样里的 sampler）。"
                            "未接线或解析失败会自动回退到默认采样。"
                        ),
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **_kwargs):
        if input_types is not None:
            expected = {
                "model": "MODEL",
                "model_b": "MODEL",
                "model_c": "MODEL",
                "video_vae": "VAE",
                "audio_vae": "VAE",
                "clip": "CLIP",
            }
            for name, want in expected.items():
                got = input_types.get(name)
                if got is not None and got != want:
                    return f"{name}: expected {want}, linked node returns {got}."
        return True

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "source_images", "report")
    OUTPUT_IS_LIST = (True, True, False, False, True, False)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "MiniMax H3 Director: MiniMaxH3ImageToVideo / ReferenceToVideo conditioning, "
        "single-stage KSampler + MiniMaxH3SigmaShift, LTXVSeparateAVLatent decode. "
        "Supports t2v / i2v / fl2v / r2v / v2v / rv2v. "
        "Optional i2v_groups / r2v_groups accept multi-group packs from Director Group nodes "
        "(external priority over UI cards). "
        "run_model picks which wired MODEL slot samples: model (main, required) or the "
        "optional model_b / model_c; an unwired pick falls back to model. "
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
        model_b=None,
        model_c=None,
        run_model=RUN_MODEL_MAIN,
        steps=25,
        sampler="res_multistep",
        scheduler="simple",
        cfg=1.0,
        seed=0,
        shift_video=12.0,
        shift_audio=3.0,
        workflow_name=None,
        sigmas=None,
        use_sigmas=False,
        **kwargs,
    ):
        del kwargs  # dropped widgets (batch_mode / use_conditioning_cache / ...) land here

        width, height, total_frames = _sanitize_timeline(width, height, total_frames)

        # 自定义 SIGMAS 调度：只有开关打开且接线正常时才生效，
        # 否则完全走原来的 steps + sampler + scheduler 路径。
        sigma_override = None
        if use_sigmas:
            if sigmas is None:
                log.warning(
                    "MiniMax H3 Director: 已开启「使用 sigmas」但 sigmas 口未接线，"
                    "本次回退到默认采样。"
                )
            else:
                from ..director.core_sampling import normalize_sigmas

                sigma_override = normalize_sigmas(sigmas)
                if sigma_override is not None:
                    log.info(
                        "MiniMax H3 Director: 使用自定义 SIGMAS 调度 —— %d 个 sigma = %d 步"
                        "（steps / scheduler 不再参与调度）。",
                        int(sigma_override.numel()),
                        int(sigma_override.numel()) - 1,
                    )

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
        )

        model, model_note = resolve_run_model(
            run_model, model=model, model_b=model_b, model_c=model_c
        )
        log.info("MiniMax H3 Director: 运行模型 %s", model_note)

        combined, segment_outputs, segment_audios, report, export_frame_counts = (
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
                sigmas=sigma_override,
                use_conditioning_cache=USE_CONDITIONING_CACHE,
                clear_vram_between_segments=CLEAR_VRAM_BETWEEN_SEGMENTS,
                workflow_name=workflow_name,
            )
        )

        if model_note:
            report = f"{report}\n\n运行模型: {model_note}"

        result = finalize_director_outputs(
            plan,
            combined,
            segment_outputs,
            report,
            export_source_images=EXPORT_SOURCE_IMAGES,
            segment_audios=segment_audios,
            segment_frame_counts=export_frame_counts,
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
