"""MiniMax H3 Director — timeline UI + official MiniMax H3 AV execution."""

from __future__ import annotations

import logging

import comfy.samplers

from ..director.batch_executor import execute_director_batch
from ..director.second_sampling import (
    DEFAULT_SECOND_SIGMA_SAMPLER,
    DEFAULT_SECOND_SIGMAS,
    SECOND_SEED_FIXED,
)
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
                # ── 连接帧加噪（仅 r2v/v2v/rv2v 生效）──────────────────────────
                # 复用 H3-Context-Noise 锥形加噪语义：对「上一段给下一段做参考的 n 帧」
                # 连接帧（被 H3 Motion Context 钉死的参考帧 latent）施加锥形高斯噪声，
                # 而非 100% 冻结。默认关闭（行为与之前完全一致）。开启后使用固定默认锥：
                # alpha=0.2 / alpha_end=0.04 / ramp=2，噪声复用采样的全局种子。
                "conn_noise": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "连接帧加噪：仅对 r2v/v2v/rv2v 的参考连接帧生效。"
                            "关闭 = 连接帧完全冻结（行为与之前一致）；"
                            "开启 = 通过 tapered noise_mask 让整段采样的【全局噪声】按固定锥权重流入这些帧"
                            "（高噪区 α=0.45 覆盖除最后 3 帧外的全部连接帧，随后 3 帧线性衰减到接缝处 α=0.10），"
                            "噪声直接复用采样的全局种子（一采用 seed、二采用固定的二采种子）。"
                            "二采自动用二采的全局噪声重复此过程。"
                        ),
                    },
                ),
                # ── 二级采样（二采）──────────────────────────────────────────
                # 刻意放在 optional 最末尾：新增 widget 会占用 widgets_values 下标，
                # 插在既有控件之前会让旧工作流整体错位。
                "bd_grp_second": ("BDGROUP", {"default": "二级采样"}),
                "upscale_model": (
                    "LATENT_UPSCALE_MODEL",
                    {
                        "tooltip": (
                            "二级采样专用放大模型：接 MiniMax H3 Latent Upscaler (3D) [Model] 节点"
                            "（nodes/latent_upscaler_3d.py）。运行时先把缓存 latent 放大再二次采样；"
                            "二采硬性要求连接此模型，未连接则二采直接报错。"
                        ),
                    },
                ),
                "second_sigmas": (
                    "SIGMAS",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "二级采样（二采）专用噪声调度：接 BasicScheduler 或 ManualSigmas。"
                            "与「使用 sigmas」相互独立——这是二采自己的调度。"
                            f"未接线时自动使用默认海螺二采调度 {list(DEFAULT_SECOND_SIGMAS)}"
                            f"（{DEFAULT_SECOND_SIGMA_SAMPLER} {len(DEFAULT_SECOND_SIGMAS) - 1} 步）。"
                        ),
                    },
                ),
                "second_run_model": (
                    list(RUN_MODEL_CHOICES),
                    {
                        "default": RUN_MODEL_MAIN,
                        "tooltip": (
                            "二级采样用哪个 MODEL 口：主模型 model，或备用口 model_b / model_c；"
                            "选中的口未接线时自动回退到主模型 model。"
                        ),
                    },
                ),
                "second_denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "二级采样（二采）去噪强度 denoise（0.0~1.0，默认 1.0）。"
                            "二采硬性走自定义 SIGMAS，故 denoise 通过缩放整条噪声调度生效："
                            "首 sigma 变为 denoise×sigma[0]，步数不变、起始噪声更小，"
                            "从而保留更多原 latent。1.0 = 完全重采样；越接近 0 越接近原图。"
                        ),
                    },
                ),
                "second_seed": (
                    "INT",
                    {
                        "default": SECOND_SEED_FIXED,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": (
                            "二级采样（二采）固定种子：用户自选的固定值，二采用自己的全局噪声场"
                            "（与一首采样的 seed 相互独立，不会联动定制）。"
                            "默认使用内置的固定值。"
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
        # ── 连接帧加噪（仅 r2v/v2v/rv2v 生效）── BOOLEAN 开关，默认关闭 ──
        conn_noise=False,
        # ── 二级采样（二采）—— 参数名与 INPUT_TYPES 末尾一致 ──
        upscale_model=None,
        second_run_model=RUN_MODEL_MAIN,
        second_sigmas=None,
        second_denoise=1.0,
        second_seed=SECOND_SEED_FIXED,
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

        # 二次采样 (second pass): 当时间线携带 secondSample 一次性触发时，跳过一采，
        # 直接对已有缓存段做 放大 + 重采样 + 连续合并出片，不与「运行」混用。
        _second_req = getattr(plan, "second_sample", None)
        if _second_req is not None and getattr(_second_req, "enabled", False) and getattr(
            _second_req, "indices", None
        ):
            import torch
            from ..director.second_sampling import run_second_sampling

            _second_model, _second_note = resolve_run_model(
                second_run_model, model=model, model_b=model_b, model_c=model_c
            )
            log.info("MiniMax H3 Director: 二次采样使用模型 %s", _second_note)

            # 二采硬性要求放大模型：未连接直接报错，不再静默回退到原分辨率。
            if upscale_model is None:
                _report = (
                    "二次采样失败：必须连接放大模型 (upscale_model)。\n"
                    "请在节点上接入 MiniMax H3 Latent Upscaler (3D) [Model] 节点"
                    "（nodes/latent_upscaler_3d.py），再触发二采。"
                )
                return (
                    [torch.zeros(1, 1, 1, 3)],
                    [{"waveform": torch.zeros(1, 1, 1), "sample_rate": int(plan.frame_rate or 24)}],
                    float(plan.frame_rate or 24),
                    0,
                    [torch.zeros(1, 1, 1, 3)],
                    _report,
                )

            # 二采 sigmas：未接线 → 默认海螺二采调度（euler 3 步，与已删除的
            # MiniMaxH3DirectorRefine 默认一致）；自行接线则沿用高级采样里的 sampler。
            _second_sigmas_eff = (
                second_sigmas if second_sigmas is not None else DEFAULT_SECOND_SIGMAS
            )
            _second_sampler_eff = (
                DEFAULT_SECOND_SIGMA_SAMPLER if second_sigmas is None else sampler
            )
            _sampled = run_second_sampling(
                node_id=unique_id,
                workflow_name=workflow_name,
                plan=plan,
                all_segments=plan.segments,
                selected_indices=list(_second_req.indices),
                model=_second_model,
                vae=video_vae,
                audio_vae=audio_vae,
                upscale_model=upscale_model,
                second_cfg=cfg,
                second_steps=steps,
                second_sampler=_second_sampler_eff,
                second_scheduler=scheduler,
                second_shift_video=shift_video,
                second_shift_audio=shift_audio,
                second_sigmas=_second_sigmas_eff,
                second_denoise=second_denoise,
                second_seed=second_seed,
                audio_mode="movie",
                decode_audio=True,
                conn_noise=conn_noise,
            )
            if _sampled.get("error"):
                _report = f"二次采样失败：{_sampled['error']}"
                return (
                    [torch.zeros(1, 1, 1, 3)],
                    [{"waveform": torch.zeros(1, 1, 1), "sample_rate": int(plan.frame_rate or 24)}],
                    float(plan.frame_rate or 24),
                    0,
                    [torch.zeros(1, 1, 1, 3)],
                    _report,
                )
            # 出片在 run_second_sampling 内部按「相邻段」分批完成（避免所有段的
            # 解码帧同时驻留内存），这里只取汇总结果。
            _export = _sampled.get("export") or {}

            def _fmt(items: list[dict]) -> list[str]:
                out = []
                for it in items or []:
                    try:
                        label = f"#{int(it.get('index', -1)) + 1}"
                    except (TypeError, ValueError):
                        label = "?"
                    out.append(f"  {label} [{it.get('stage', '?')}] {it.get('reason', '')}")
                return out

            _done = sorted({int(i) for i in _second_req.indices})
            _ok = _sampled.get("results") or {}
            _files = _export.get("files", []) if isinstance(_export, dict) else []
            _lines = [
                f"二次采样完成：勾选 {len(_done)} 段，成功 {len(_ok)} 段 → {len(_files)} 个文件",
                f"噪声调度：{_sampled.get('schedule', '未知')}",
            ]
            _failures = _fmt(_sampled.get("failures") or [])
            if _failures:
                _lines.append(f"失败 {len(_failures)} 段（这些段没有产出）：")
                _lines.extend(_failures)
            _warns = _fmt(_sampled.get("warnings") or [])
            if _warns:
                _lines.append("告警：")
                _lines.extend(_warns)
            _lines.append("输出文件：" if _files else "（无输出文件）")
            _lines.extend(f"  {p}" for p in _files)
            _report = "\n".join(_lines)
            # 与一采统一解码：用真实二采帧走 finalize_director_outputs 输出 IMAGE/AUDIO，
            # 不再回退到 1x1 占位帧（否则下游 VideoCombine 会以 libx264 无法开启崩溃）。
            _second = _sampled.get("second_chunks") or []
            _second_aud = _sampled.get("second_audios") or []
            _second_order = _sampled.get("second_order") or []
            _pairs = sorted(
                zip(_second_order, _second, _second_aud),
                key=lambda t: int(t[0]),
            )
            _second = [p[1] for p in _pairs]
            _second_aud = [p[2] for p in _pairs]
            if _second:
                _combined = torch.cat(_second, dim=0)
                _frame_counts = [int(t.shape[0]) for t in _second]
                result = finalize_director_outputs(
                    plan,
                    _combined,
                    list(_second),
                    _report,
                    export_source_images=EXPORT_SOURCE_IMAGES,
                    segment_audios=list(_second_aud),
                    segment_frame_counts=_frame_counts,
                )
            else:
                result = finalize_director_outputs(
                    plan,
                    None,
                    [],
                    _report,
                    export_source_images=EXPORT_SOURCE_IMAGES,
                )
            return result

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
                conn_noise=conn_noise,
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
