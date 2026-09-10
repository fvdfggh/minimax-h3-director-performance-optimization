"""Second-sample (二采) executor for MiniMax H3 Director Opt.

Pipeline, per selected segment:

1. load the first-pass AV latent (``seg_*`` cache group);
2. upscale the *video* latent via the connected ``LATENT_UPSCALE_MODEL`` object
   (``upscale_model(latent)``), re-merge AV — this happens up-front for every
   selected segment so the upscaler is loaded once and then parked in RAM;
3. load the first-pass text encoding straight from its recorded hash key
   (``conditioning_cache`` — the second pass cannot recompute the Qwen prefill);
4. seam: :func:`h3_motion_context.apply_motion_context` with the **upscaled**
   previous segment as context (a first-pass latent would be the wrong size and
   the seam would be refused);
5. :func:`core_sampling.sample_single_stage` with the second-pass seed and the
   **mandatory** ``sigmas`` schedule;
6. decode -> frames + audio, trim to the export body;
7. cache under the ``seg2_*`` group (:func:`segment_cache.save_second_pass_cache`).

As soon as a run of timeline-adjacent segments is complete,
:func:`export_second_pass` stitches it (timeline neighbours only) into an mp4 and
the run's frames are dropped — exactly like the「连续导出」layout, but streamed
instead of buffered, so a large selection never holds every decoded clip in RAM
at once.

Design notes
------------
* **Devices are never dictated here.** No ``map_location``, no ``.to(device)``:
  the sampler, the VAE and the upscaler each move tensors where they need them.
  Pinning a device by hand is what previously forced a CPU latent into a CUDA
  UNET (and, conversely, would pin a GPU latent on a CPU-only box).
* **VRAM**: the upscaler is asked to stay resident for the whole upscale phase
  (``force_unload=False``) and is then explicitly parked back in RAM before any
  sampling starts, so sampling only ever has the UNET + VAE in VRAM.
* **System RAM**: segments are upscaled in batches bounded by
  ``memory_guard_bytes``, and each completed run of adjacent segments is stitched
  + written immediately instead of every decoded clip being kept until the end
  (one run's pixels in RAM, not the whole selection's).
* **Nothing fails silently.** A segment that cannot be seamed, sampled or cached
  is reported with a stage + reason instead of being quietly skipped — a skipped
  segment used to leave the context prefix inside the exported clip.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any, Callable

import torch

from . import segment_slots
from .conditioning_cache import (
    load_conditioning_by_key, load_segment_second_params,
    retime_conditioning_for_frames,
)
from .core_sampling import normalize_sigmas, sample_single_stage
from .frame_align import minimax_align_frame_count
from .h3_motion_context import (
    apply_motion_context,
    pixel_frames_for_latent_t,
    snap_context_frames,
    handoff_end_frame,
    video_from_latent,
    resolve_tail_context_length,
    DEFAULT_AUDIO_CONTEXT_FRAMES,
)
from .segment_continuity import is_continuity_active
from .segment_cache import (
    _expected_export_frames,
    load_segment_av_latent,
    load_segment_cache,
    load_segment_audio,
    load_segment_handoff_meta,
    load_next_segment_av_latent,
    run_segment_export,
    save_second_pass_cache,
    slot_content_hash,
    sync_segment_slots,
    sync_second_segment_slots,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.second_sampling")

#: (seg_index, done, total) progress callback.
ProgressCb = Callable[[int, int, int], None]

# 二采默认噪声调度：取自已删除的 MiniMaxH3DirectorOptRefine 模块
# （原 director/refine_pack.py: HAILUO_REFINE_SIGMAS + DEFAULT_REFINE_SIGMA_SAMPLER）。
# 海螺参考生视频二采：ManualSigmas 4 个数 = euler 3 步。
DEFAULT_SECOND_SIGMAS = (0.85, 0.7250, 0.4219, 0.0)
DEFAULT_SECOND_SIGMA_SAMPLER = "euler"
#: 二采专属固定种子：不暴露给用户调整，始终用同一个全局噪声场。
#: 取一个与一首采样种子（UI 的 seed 默认 0）不同的固定值，确保二采有自己独立的
#: 全局噪声场，不会与一首采样完全重合。
SECOND_SEED_FIXED = 20240

#: Batch budget for the upscaled latents held in system RAM before a batch is
#: flushed through sampling. Purely a RAM guard — VRAM is handled by parking the
#: upscaler before sampling, not by shrinking this.
DEFAULT_MEMORY_GUARD_BYTES = 2 * 1024**3


# ---------------------------------------------------------------------------
# AV latent split / join (mirrors the original refine helpers)
# ---------------------------------------------------------------------------

def _split_av(samples: dict):
    """Split an AV latent dict into ``(video_latent, audio_latent)``."""
    from comfy_extras.nodes_lt import LTXVSeparateAVLatent

    sep = LTXVSeparateAVLatent.execute(samples)
    if hasattr(sep, "args"):
        sep = sep.args
    return sep[0], sep[1]


def _join_av(video_latent, audio_latent, template: dict) -> dict:
    """Re-merge a video latent and an audio latent back into an AV latent dict."""
    out = dict(template)
    out.pop("noise_mask", None)
    try:
        from comfy_extras.nodes_lt import LTXVConcatAVLatent

        joined = LTXVConcatAVLatent.execute(video_latent, audio_latent)
        if hasattr(joined, "args"):
            joined = joined.args
        packed = joined[0]
        if isinstance(packed, dict) and "samples" in packed:
            return packed
        out["samples"] = packed
        return out
    except Exception:
        pass
    try:
        import comfy.nested_tensor

        v = video_latent.get("samples") if isinstance(video_latent, dict) else video_latent
        a = audio_latent.get("samples") if isinstance(audio_latent, dict) else audio_latent
        out["samples"] = comfy.nested_tensor.NestedTensor((v, a)) if a is not None else v
        return out
    except Exception:
        v = video_latent.get("samples") if isinstance(video_latent, dict) else video_latent
        a = audio_latent.get("samples") if isinstance(audio_latent, dict) else audio_latent
        out["samples"] = (v, a) if a is not None else v
        return out


def _accept_upscaled(up) -> dict | None:
    """Normalise whatever the upscale model returned into a latent dict."""
    if isinstance(up, dict) and "samples" in up:
        return up
    if isinstance(up, torch.Tensor):
        return {"samples": up}
    if isinstance(up, (list, tuple)) and up and isinstance(up[0], dict) and "samples" in up[0]:
        return up[0]
    if isinstance(up, dict):
        return up
    return None


def _tensor_bytes(value: Any) -> int:
    """Approximate resident bytes of an AV latent (video + audio)."""
    total = 0
    samples = value.get("samples") if isinstance(value, dict) else value
    if isinstance(samples, torch.Tensor):
        total += int(samples.numel()) * int(samples.element_size())
    elif isinstance(samples, (tuple, list)):
        for item in samples:
            if isinstance(item, torch.Tensor):
                total += int(item.numel()) * int(item.element_size())
    return total


def _wanted_context_frames(params: dict, plan) -> int:
    """How many context frames the *cached* conditioning was actually built for.

    ``context_n`` is what the FIRST pass really pinned — 0 when it pinned
    nothing, e.g. because the previous segment had not been sampled in that run.
    The cached text encoding only carries room for that many head frames, so the
    second pass has to reuse the exact same number: re-deriving it from the
    widget would pin a prefix the conditioning has no capacity for (either a
    refused seam or a clip whose head is unpinned noise).
    """
    raw = params.get("context_n")
    if raw is None:
        # Map written before ``context_n`` was recorded: fall back to the plan.
        return int(snap_context_frames(plan.continuity_overlap_frames))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _export_frame_budget(plan, seg, decoded_n: int, context_n: int) -> int:
    """Export length of a second-pass segment, on the FIRST pass's terms.

    ``seg.frame_count`` alone is not the export length: the first pass exports
    the segment's own aligned length (see :func:`_expected_export_frames`), and
    the replayed head is trimmed separately rather than snapping the export
    short. Reusing that helper is what keeps a second-pass clip the same length
    as its first-pass counterpart instead of a few frames longer/shorter.
    """
    _planned_trim, planned = _expected_export_frames(plan, seg, fallback_n=decoded_n)
    if int(context_n or 0) <= 0:
        # No replayed head was applied, so the body is the plain aligned length.
        frame_count = int(getattr(seg, "frame_count", 0) or 0)
        if frame_count > 0:
            try:
                return int(minimax_align_frame_count(max(5, frame_count)))
            except Exception:  # pragma: no cover - defensive
                return int(planned or decoded_n)
    return int(planned or 0) or int(decoded_n)


# ---------------------------------------------------------------------------
# Core executor
# ---------------------------------------------------------------------------

def run_second_sampling(
    *,
    node_id: str | None,
    workflow_name: str | None,
    plan,
    all_segments,
    selected_indices: list[int],
    model,
    vae,
    audio_vae,
    upscale_model,
    second_cfg: float,
    second_steps: int,
    second_sampler: str,
    second_scheduler: str,
    second_shift_video: float,
    second_shift_audio: float,
    second_sigmas,
    second_denoise: float = 1.0,
    second_seed: int = SECOND_SEED_FIXED,
    audio_mode: str = "movie",
    decode_audio: bool = True,
    memory_guard_bytes: int = DEFAULT_MEMORY_GUARD_BYTES,
    out_dir: str | None = None,
    on_progress: ProgressCb | None = None,
    # Connected-frame (r2v/v2v/rv2v) taper noise — applied to the pinned
    # reference frames via a tapered noise_mask so the second pass re-uses its
    # own global noise. Off by default (matches first pass default).
    conn_noise: bool = False,
) -> dict[str, Any]:
    """Run the second pass over ``selected_indices`` and cache each result.

    Returns ``results`` (per-segment ok/frame count), a ``failures`` list of
    ``{"index", "stage", "reason"}`` — a stage, not a bare message, so the report
    can say *why* a segment produced nothing — and the ``export`` summary
    (``files`` / ``skipped`` / ``export_dir``).

    与一采 batch 流程一致，分两阶段：
      Phase 2（采样）：遍历选中段，只跑 UNet 采样生成 latent，暂不解码；段间连续性
                       用本 run 已采样 latent（与必要时磁盘缓存）拼接。
      Phase 3（解码）：采样全部完成后，卸载 UNet、常驻 VAE，把收集到的 latent 在
                       最后统一顺序解码、裁剪、落盘缓存，再做连续导出。
    VAE 解码放到最后一起做，避免每采样一段就解一次码、解码与采样交错占用 VRAM。
    """
    if not node_id:
        return {"error": "node_id required"}

    # Imported lazily: both live in batch_executor which pulls in comfy nodes.
    from .batch_executor import _decode_av_latent, _trim_decoded_to_export

    ordered = sorted({int(i) for i in selected_indices})
    seg_by_index = {int(getattr(s, "index", -1)): s for s in (all_segments or [])}
    total = len(ordered)
    done = 0

    results: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    # 两阶段状态：Phase 2 先把采样 latent 与段元数据收在这里，Phase 3 一起解码。
    sampled_latents: dict[int, dict] = {}
    seg_meta: dict[int, dict] = {}
    pending_order: list[int] = []
    decoded_order: list[int] = []
    export_files: list[str] = []
    export_skipped: list[dict[str, Any]] = []
    export_dir_out = ""
    # 与一采统一的节点输出：Phase 3 解码后保留每段真实解码帧/音频，
    # 供 nodes/director.py 走 finalize_director_outputs 输出 IMAGE/AUDIO，
    # 避免回退到 1x1 占位帧导致下游 VideoCombine 因 libx264 无法开启而崩溃。
    second_frames: list[torch.Tensor] = []
    second_audio: list[dict] = []
    second_idx: list[int] = []

    def _finalize() -> None:
        """Phase 3: decode every sampled latent together, then export.

        Mirrors 一采 batch 的 Phase 3 — UNet 卸载、VAE 常驻，把 Phase 2 收齐的
        latent 在最后统一顺序解码、裁剪、落盘缓存（seg2_*_clip.mp4 + frames_ht），
        再委托 run_segment_export 读取缓存做连续导出。解码前的连续性已靠本 run 的
        latent / handoff 完成，无需提前解码。
        """
        nonlocal export_dir_out
        if not pending_order:
            return
        # Unload UNet to free VRAM for the VAE (UNet ~20GB + VAE ~5GB > 22GB GPU),
        # exactly like 一采 Phase 3.
        try:
            from .vram_cleanup import cleanup_segment_vram
            cleanup_segment_vram(enabled=True, unload_models=True)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("二采: UNet 卸载失败 (%s)", exc)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for idx in sorted(pending_order):
            _decode_one(idx)
        # 导出读取刚写入的 seg2_* 缓存（与一采「连续导出」source=二采 一致）。
        if decoded_order:
            try:
                written = export_second_pass(
                    node_id=node_id,
                    workflow_name=workflow_name,
                    plan=plan,
                    all_segments=all_segments,
                    selected_indices=list(decoded_order),
                    sampled={},
                    out_dir=out_dir,
                )
            except Exception as exc:  # pragma: no cover - defensive
                export_skipped.append({"run": list(decoded_order), "reason": str(exc)})
                log.warning("二采导出失败 (%s)", exc)
            else:
                export_files.extend(written.get("files") or [])
                export_skipped.extend(written.get("skipped") or [])
                export_dir_out = str(written.get("export_dir") or export_dir_out)

    def _fail(idx: int, stage: str, reason: str) -> None:
        failures.append({"index": int(idx), "stage": stage, "reason": str(reason)})
        log.warning("二采: 段 #%d %s 失败 (%s)", int(idx) + 1, stage, reason)

    def _advance(idx: int) -> None:
        nonlocal done
        done += 1
        if on_progress:
            on_progress(int(idx), done, total)

    # --- slot maps ----------------------------------------------------------
    # Both passes must be in sync before anything is read or written. The first
    # pass map is normally only synced by the status route, so a timeline edit
    # followed by an immediate run would otherwise read another segment's latent.
    for name, call in (
        ("一采", lambda: sync_segment_slots(node_id, plan, workflow_name=workflow_name)),
        ("二采", lambda: sync_second_segment_slots(node_id, plan, workflow_name=workflow_name)),
    ):
        try:
            call()
        except Exception as exc:
            # Not recoverable: without a stem every cache write below silently
            # no-ops, which is exactly the "report says 0 files" failure mode.
            return {
                "error": f"二采 {name} slot map 同步失败: {exc}",
                "failures": [{"index": -1, "stage": "sync", "reason": str(exc)}],
            }

    if second_sigmas is None:
        # 二采硬性要求 sigmas：未接线时回退到默认海螺二采调度
        # （原 MiniMaxH3DirectorOptRefine / refine_pack.py 的 HAILUO_REFINE_SIGMAS）。
        second_sigmas = DEFAULT_SECOND_SIGMAS
        log.info(
            "二采: 未提供 SIGMAS，使用默认海螺二采调度 %s（%s %d 步）。",
            list(DEFAULT_SECOND_SIGMAS),
            DEFAULT_SECOND_SIGMA_SAMPLER,
            len(DEFAULT_SECOND_SIGMAS) - 1,
        )
    schedule_note = "连线 SIGMAS" if second_sigmas is not DEFAULT_SECOND_SIGMAS else (
        f"默认海螺二采调度 {list(DEFAULT_SECOND_SIGMAS)}"
    )
    sigma_tensor = normalize_sigmas(second_sigmas)
    if sigma_tensor is None:
        # normalize_sigmas 仅在 <2 个值时返回 None；理论上不会到这（默认值 4 个且以 0 收尾）。
        log.warning("二采: SIGMAS 解析失败，强制使用默认海螺二采调度。")
        sigma_tensor = normalize_sigmas(DEFAULT_SECOND_SIGMAS)
        schedule_note = f"默认海螺二采调度 {list(DEFAULT_SECOND_SIGMAS)}（连线值解析失败）"

    # denoise：二采硬性走自定义 SIGMAS，而 ComfyUI 在提供 sigmas 时会忽略
    # sample() 的 denoise 参数（schedule 由 sigmas 决定）。要让「denoise」生效，
    # 只能把整条调度按 denoise 缩放——首 sigma 变成 denoise*sigma[0]，步数不变，
    # 起始噪声更小，从而保留更多原 latent（标准 img2img denoise 语义）。
    # denoise>=1 时还原为 1.0（完全重采样），不做缩放。
    try:
        _denoise = float(second_denoise)
    except (TypeError, ValueError):
        _denoise = 1.0
    _denoise = max(0.0, min(1.0, _denoise))
    if _denoise < 1.0 and sigma_tensor is not None:
        sigma_tensor = sigma_tensor * _denoise
        schedule_note = f"{schedule_note}（denoise={_denoise:.3f}）"

    # --- phase 1: upscale every selected latent -----------------------------
    # ``force_unload=False`` keeps the upscaler resident for the whole batch, so
    # it is loaded once instead of once per segment. It is parked back in RAM
    # before the first sample so only the UNET + VAE share VRAM while sampling.
    def _upscale_video(video_latent):
        if upscale_model is None:
            return video_latent
        try:
            return upscale_model(video_latent, force_unload=False)
        except TypeError:
            # A third-party upscale model without that knob: fall back to its
            # own default behaviour (it unloads itself after every call).
            return upscale_model(video_latent)

    def _park_upscaler() -> None:
        unload = getattr(upscale_model, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception as exc:
                log.debug("二采: 放大模型卸载失败 (%s)", exc)

    # idx -> upscaled AV latent, flushed in timeline order once the batch is full
    batch: dict[int, dict] = {}
    batch_bytes = 0

    def _upscale_one(idx: int) -> dict | None:
        nonlocal batch_bytes
        seg = seg_by_index.get(int(idx))
        if seg is None:
            _fail(idx, "plan", f"时间线中找不到段 #{int(idx) + 1}")
            return None
        try:
            av = load_segment_av_latent(
                node_id, seg, plan, allow_stale=True, workflow_name=workflow_name,
                variant=segment_slots.VARIANT_FIRST,
            )
        except Exception as exc:
            _fail(idx, "latent", f"一采 latent 读取失败: {exc}")
            return None
        if not isinstance(av, dict) or "samples" not in av:
            _fail(idx, "latent", "无一首采 latent 缓存")
            return None
        try:
            video_latent, audio_latent = _split_av(av)
            up = _upscale_video(video_latent)
            upscaled = _accept_upscaled(up)
            if upscaled is None:
                _fail(idx, "upscale", "放大模型未返回 latent")
                return None
            new_av = _join_av(upscaled, audio_latent, av)
        except Exception as exc:
            _fail(idx, "upscale", str(exc))
            return None
        batch_bytes += _tensor_bytes(new_av)
        return new_av

    # --- phase 2: seam + sample + decode + cache ----------------------------
    def _prev_video_tail(prev_seg) -> "torch.Tensor | None":
        """上一段**二采**的解码视频尾帧,作为 pixel-path 接缝锚点。

        **只读取二采产物(seg2_* clip),绝不用一采尾帧(seg_* clip)** —— 二采引用
        上段时,只能以"上一段被精修后的视频"为准,不能用一采尾帧污染二采的连续性。
        没有二采视频时返回 None,由调用方回退到 latent 连续性。
        """
        try:
            frames = load_segment_cache(
                node_id, prev_seg, plan, allow_stale=True,
                workflow_name=workflow_name, variant=segment_slots.VARIANT_SECOND,
            )
        except Exception:
            frames = None
        if frames is not None and int(getattr(frames, "shape", (0,))[0]) > 0:
            return frames
        return None

    def _context_for(idx: int):
        """``(context_latent, context_frames)`` for this segment's seam.

        强制「第二种方式」:**二采引用上段时只认二采自己的产物**,绝不以任何形式
        引用一采——既不用一采视频尾帧,也**不用一采 latent**。

        - 上一段有二采视频(seg2_* clip)→ 返回 ``(None, seg2_*尾帧)``,强制
          apply_motion_context 走 pixel 路径,把上一段二采视频尾帧拼到下一段头部。
        - 上一段尚无二采视频 → 回退到**二采自己的 latent**:同 run 内已采样的
          upscaled latent,或磁盘上的 seg2_* latent。二者都没有就 ``(None, None)``
          (不 pin,硬切),绝不退化到一采 latent。
        - 注:帧数 context_n 仍可从一采缓存的 conditioning 里读,那只是"对齐用的
          帧数",不是对一采内容的引用,所以没问题。
        """
        prev_idx = int(idx) - 1
        prev_seg = seg_by_index.get(prev_idx)
        if prev_seg is None:
            return None, None

        # 1) 优先:上一段二采视频尾帧 → 强制 pixel 路径。
        video_tail = _prev_video_tail(prev_seg)
        if video_tail is not None:
            return None, video_tail

        # 2) 上一段尚无二采视频:只用「二采自己的 latent」,绝不引用一采。
        #    同 run 内上一段已采样 → 用 sampled_latents[prev],即上一段二采采样后
        #    的 latent(解码即上一段二采视频),这才是二采应当参考的内容。
        #    注意:不要用 batch[prev]——那是 _upscale_one 从 VARIANT_FIRST 加载再放大的
        #    一采 upscaled latent,属于一采参考,已被禁止。
        fresh = sampled_latents.get(prev_idx)
        if fresh is not None:
            return fresh, None
        # 磁盘上的二采 latent(seg2_*,二采产物)。
        try:
            cached = load_segment_av_latent(
                node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name,
                variant=segment_slots.VARIANT_SECOND,
            )
        except Exception:
            cached = None
        if cached is not None:
            return cached, None
        # 没有任何二采产物可用 → 不 pin(硬切),绝不退化到一采 latent。
        return None, None

    def _sample_one(idx: int, new_av: dict) -> None:
        seg = seg_by_index.get(int(idx))
        if seg is None:
            _fail(idx, "plan", f"时间线中找不到段 #{int(idx) + 1}")
            return

        # text encoding, loaded by the hash key the first pass recorded.
        # Keyed by the segment's CONTENT hash, not its position: after a timeline
        # reorder the segment now living at this index is a different one, and
        # position-keyed params would hand it a neighbour's prompt silently.
        try:
            params = load_segment_second_params(
                node_id, workflow_name, int(idx),
                slot_key=slot_content_hash(seg, plan),
            ) or {}
        except Exception as exc:
            _fail(idx, "text", f"二采参数读取失败: {exc}")
            return
        text_key = str(params.get("text_key") or "")
        if not text_key:
            _fail(idx, "text", "该段无 text_key（需先跑一次一采）")
            return
        try:
            cond = load_conditioning_by_key(node_id, workflow_name, text_key)
        except Exception as exc:
            _fail(idx, "text", f"文本编码读取失败: {exc}")
            return
        if cond is None or cond.get("positive") is None:
            _fail(idx, "text", f"文本编码缓存缺失 (text_key={text_key})")
            return
        positive, negative = cond["positive"], cond.get("negative")

        # The text key ignores the sample length, so the cached anchors may have
        # been timed against the first pass's duration — re-map them onto this
        # pass's latent before anything reads them.
        try:
            _want = pixel_frames_for_latent_t(int(video_from_latent(new_av).shape[2]))
        except Exception:  # pragma: no cover - defensive
            _want = 0
        positive = retime_conditioning_for_frames(
            positive,
            int((cond.get("metadata") or {}).get("frame_count") or 0),
            _want,
        )

        # 与一采一致：被参照段导出 17k（相位由参照帧承载），无参照段导出 17k+5。
        _, expected_export = _expected_export_frames(plan, seg, fallback_n=0)
        expected_export = int(expected_export)

        # seam — same gating as the first pass, or the head of the sample is
        # left unpinned while still being exported.
        positive_seam = positive
        trim_frames = 0
        context_n = 0
        if int(idx) > 0 and is_continuity_active(plan, seg):
            wanted = _wanted_context_frames(params, plan)
            if wanted <= 0:
                # The first pass pinned nothing for this segment, so its cached
                # text encoding has no spare head frames. Pinning here would
                # either be refused or leave an unpinned head in the clip.
                warnings.append({
                    "index": int(idx),
                    "stage": "seam",
                    "reason": "一采未使用运动上下文（文本编码无上下文余量），本段不继承运动",
                })
            else:
                context_latent, context_frames = _context_for(idx)
                if context_latent is None and context_frames is None:
                    warnings.append({
                        "index": int(idx),
                        "stage": "seam",
                        "reason": "上一段无可用上下文，本段不继承运动",
                    })
                else:
                    context_n = snap_context_frames(wanted)
                    # --- 与一采统一的结尾 / 连续性上下文 ---
                    # 1) 上一段音频连续性（一采: context_audio=prev_audio）
                    prev_idx = int(idx) - 1
                    prev_seg = seg_by_index.get(prev_idx)
                    prev_audio = None
                    if prev_seg is not None:
                        prev_audio = load_segment_audio(
                            node_id, prev_seg, plan,
                            allow_stale=True, workflow_name=workflow_name,
                            variant=segment_slots.VARIANT_SECOND,
                        )
                        if prev_audio is None:
                            prev_audio = load_segment_audio(
                                node_id, prev_seg, plan,
                                allow_stale=True, workflow_name=workflow_name,
                            )
                    # 2) 上一段 handoff 结尾边界（一采: context_end_frame=prev_end_frame）
                    prev_end_frame = None
                    if prev_seg is not None:
                        # 优先用本 run 采样阶段已算好的 handoff（与一采 Phase 2 的
                        # completed_av_handoff 一致）；否则回退磁盘缓存。
                        prev_handoff = (
                            seg_meta.get(int(prev_idx), {}).get("handoff")
                            or load_segment_handoff_meta(
                                node_id, prev_seg, plan,
                                allow_stale=True, workflow_name=workflow_name,
                                variant=segment_slots.VARIANT_SECOND,
                            )
                            or load_segment_handoff_meta(
                                node_id, prev_seg, plan,
                                allow_stale=True, workflow_name=workflow_name,
                            )
                        )
                        if prev_handoff:
                            prev_end_frame = handoff_end_frame(
                                trim_frames=int(prev_handoff.get("trim_frames") or 0),
                                export_frames=int(prev_handoff.get("export_frames") or 0),
                            )
                            sample_f = int(prev_handoff.get("sample_frames") or 0)
                            if sample_f > 0 and prev_end_frame >= sample_f:
                                prev_end_frame = None
                    # 3) 对齐下段：把下一段开头 pin 进本段结尾（一采: tail_context_*）
                    tail_context_latent = None
                    tail_context_length = 0
                    tail_context_offset = 0
                    if bool(getattr(seg, "continuity_to_next", False)):
                        tail_n = resolve_tail_context_length(
                            new_av, expected_export, context_n=context_n
                        )
                        if tail_n > 0:
                            next_latent = load_next_segment_av_latent(
                                node_id, seg.index, workflow_name=workflow_name
                            )
                            if next_latent is not None:
                                tail_context_latent = next_latent
                                tail_context_length = tail_n
                                # 用已裁切的下一段开头（跳过其自身头裁），与一采一致。
                                _next_seg = seg_by_index.get(int(seg.index) + 1)
                                if _next_seg is not None:
                                    _next_ho = load_segment_handoff_meta(
                                        node_id, _next_seg, plan,
                                        allow_stale=True, workflow_name=workflow_name,
                                    )
                                    if _next_ho:
                                        tail_context_offset = int(_next_ho.get("trim_frames", 0) or 0)
                    try:
                        positive_seam, trim_frames, _ = apply_motion_context(
                            positive,
                            new_av,
                            vae=vae,
                            context_length=int(context_n),
                            context_latent=context_latent,
                            context_frames=context_frames,
                            context_audio=prev_audio,
                            audio_vae=audio_vae,
                            continue_audio=(
                                str(audio_mode).lower() != "mute"
                                and (context_latent is not None or prev_audio is not None)
                            ),
                            keep_existing_keyframes=(str(getattr(seg, "task_key", "")) == "fl2v"),
                            context_end_frame=prev_end_frame,
                            audio_context_length=DEFAULT_AUDIO_CONTEXT_FRAMES,
                            tail_context_latent=tail_context_latent,
                            tail_context_length=tail_context_length,
                            tail_context_offset=tail_context_offset,
                            conn_noise=conn_noise,
                            seed=int(second_seed),
                        )
                    except Exception as exc:
                        # Do NOT swallow this and continue: without the pin the
                        # sample's head is unguided noise that survives the later
                        # trim and lands in the export.
                        _fail(idx, "seam", str(exc))
                        return

        try:
            sampled = sample_single_stage(
                model=model,
                positive=positive_seam,
                negative=negative,
                latent=new_av,
                seed=int(second_seed),
                cfg=float(second_cfg),
                steps=int(second_steps),
                sampler_name=second_sampler,
                scheduler=second_scheduler,
                shift_video=float(second_shift_video),
                shift_audio=float(second_shift_audio),
                denoise=1.0,
                sigmas=sigma_tensor,
                apply_shift=True,
            )
        except Exception as exc:
            _fail(idx, "sample", str(exc))
            return

        # --- Phase 2 收尾：仅保存采样结果，VAE 解码推迟到 Phase 3 统一进行 ---
        # 与一采 batch 一致：先收齐所有 latent，再在末尾一起解码。handoff 的
        # export_frames 用节点参数推导（与一采 Phase 2 的 completed_av_handoff
        # 一致），用于下一段采样时的相位对齐；实际裁剪长度在 Phase 3 解码后定。
        _, expected_export = _expected_export_frames(plan, seg, fallback_n=0)
        seg_meta[int(idx)] = {
            "trim_frames": int(trim_frames or 0),
            "context_n": int(context_n or 0),
            "handoff": {
                "trim_frames": int(trim_frames or 0),
                "export_frames": int(expected_export),
                "sample_frames": int(expected_export),
            },
        }
        sampled_latents[int(idx)] = sampled
        pending_order.append(int(idx))

    def _decode_one(idx: int) -> None:
        """Phase 3: decode one sampled latent, trim, cache, keep for node output."""
        seg = seg_by_index.get(int(idx))
        if seg is None:
            _fail(idx, "plan", f"时间线中找不到段 #{int(idx) + 1}")
            return
        meta = seg_meta.get(int(idx))
        sampled = sampled_latents.get(int(idx))
        if meta is None or sampled is None:
            _fail(idx, "decode", "采样记录缺失")
            return
        try:
            images, audio = _decode_av_latent(sampled, vae, audio_vae, decode_audio=decode_audio)
        except Exception as exc:
            _fail(idx, "decode", str(exc))
            return
        if images is None or int(images.shape[0]) <= 0:
            _fail(idx, "decode", "解码结果为空")
            return
        decoded_n = int(images.shape[0])  # 裁剪前的真实采样帧数
        try:
            # Same frame budget the first pass exports (the aligned length, with
            # the replayed head trimmed separately), NOT the raw widget
            # frame_count — otherwise the second-pass clip ends up a few frames
            # longer or shorter than the first-pass one.
            export_len = _export_frame_budget(
                plan, seg, int(images.shape[0]), int(meta["context_n"])
            )
            images, audio = _trim_decoded_to_export(
                images,
                audio,
                trim_frames=int(meta["trim_frames"] or 0),
                export_len=export_len,
                plan=plan,
            )
        except Exception as exc:
            warnings.append({"index": int(idx), "stage": "trim", "reason": str(exc)})

        handoff = dict(meta["handoff"])
        handoff["export_frames"] = int(images.shape[0])
        handoff["sample_frames"] = int(decoded_n)
        meta_extra = {
            "source_segment": int(idx),
            "upscaled": upscale_model is not None,
            "upscale_model": type(upscale_model).__name__ if upscale_model is not None else None,
            "second_seed": int(second_seed),
            "second_cfg": float(second_cfg),
            "second_steps": int(second_steps),
            "second_sampler": second_sampler,
            "second_scheduler": second_scheduler,
            "second_denoise": float(_denoise),
            "second_sigmas": None if sigma_tensor is None else sigma_tensor.tolist(),
            "schedule_source": schedule_note,
            "context_frames": int(meta["context_n"]),
            "has_audio": bool(isinstance(audio, dict) and audio.get("waveform") is not None),
        }
        try:
            if not save_second_pass_cache(
                node_id,
                seg,
                plan,
                images,
                av_latent=sampled,
                audio=audio,
                handoff=handoff,
                meta_extra=meta_extra,
                workflow_name=workflow_name,
            ):
                warnings.append({"index": int(idx), "stage": "cache", "reason": "seg2 槽位缺失"})
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append({"index": int(idx), "stage": "cache", "reason": str(exc)})

        results[int(idx)] = {"ok": True, "frames": int(images.shape[0])}
        # 真实解码帧/音频保留给节点输出（与一采统一解码）；导出读取 seg2_* 缓存。
        second_frames.append(images)
        second_audio.append(audio if isinstance(audio, dict) else {})
        second_idx.append(int(idx))
        decoded_order.append(int(idx))

    def _flush() -> None:
        # Phase 2: sample every pending latent (no decode here — VAE decode is
        # deferred to Phase 3, run once at the very end via _finalize).
        for j in sorted(batch):
            try:
                _sample_one(j, batch[j])
            finally:
                _advance(j)
        batch.clear()

    try:
        for idx in ordered:
            new_av = None
            try:
                new_av = _upscale_one(idx)
            finally:
                if new_av is None:
                    _advance(idx)
            if new_av is None:
                continue
            batch[int(idx)] = new_av
            # RAM guard: sample what we have before holding any more. The
            # upscaler is parked first so it never shares VRAM with the UNET.
            if batch_bytes >= int(memory_guard_bytes or 0) and len(batch) > 1:
                _park_upscaler()
                _flush()
                batch_bytes = 0
                gc.collect()

        _park_upscaler()
        _flush()
        _finalize()
    finally:
        # Park the upscaler, then clear the allocator cache once for the whole
        # run. Main models stay loaded: unloading them here would force every
        # following run to搬运 them back from RAM for no benefit.
        _park_upscaler()
        try:
            from .vram_cleanup import cleanup_segment_vram

            cleanup_segment_vram(enabled=True, unload_models=False)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("二采: 结束清理失败 (%s)", exc)
        gc.collect()

    # Segments that produced nothing still belong in the report: the incremental
    # export above only ever sees the runs that succeeded, so their "skipped"
    # entries are added here.
    missing = [int(i) for i in ordered if int(i) not in results]
    for idx in missing:
        export_skipped.append({"run": [idx], "reason": "该段二采未成功"})

    # 与一采统一的节点输出：按时间线序号排序的真实解码帧 / 音频。
    _order_sorted = sorted(range(len(second_idx)), key=lambda k: second_idx[k])
    second_chunks = [second_frames[k] for k in _order_sorted]
    second_audios = [second_audio[k] for k in _order_sorted]
    second_order = [second_idx[k] for k in _order_sorted]

    return {
        "results": results,
        "failures": failures,
        "warnings": warnings,
        "schedule": schedule_note,
        "second_chunks": second_chunks,
        "second_audios": second_audios,
        "second_order": second_order,
        "export": {
            "files": export_files,
            "skipped": export_skipped,
            "export_dir": export_dir_out,
        },
    }


# ---------------------------------------------------------------------------
# Continuous-merge export (adjacent runs only)
# ---------------------------------------------------------------------------

def export_second_pass(
    *,
    node_id: str | None,
    workflow_name: str | None,
    plan,
    all_segments,
    selected_indices: list[int],
    sampled: dict[str, Any],
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Stitch second-pass results into mp4s (timeline-adjacent runs only).

    与一采「连续导出 / 选择运行」统一：直接复用 :func:`run_segment_export`
    （``mode="continuous"``、``variant="2nd"``）。二采在采样阶段已通过
    :func:`save_second_pass_cache` 把 ``seg2_*_clip.mp4`` + ``frames_ht``（首尾帧
    接缝窗口）+ latent + audio + meta 落盘，产物与一采导出完全一致，因此这里应当和
    一采对同一份缓存做同样的拼接——从缓存视频读取、用 ``frames_ht`` 的首尾帧还原
    接缝窗口做平滑——而不是把刚解码、还在内存里的帧直接喂进去。

    ``out_dir`` 沿用二采专属导出目录（``minimax_second_pass_export``），保持与
    之前一致的落盘位置；若不传则由 :func:`run_segment_export` 自行决定目录。
    """
    if not selected_indices:
        return {"files": [], "skipped": [], "export_dir": out_dir or ""}
    res = run_segment_export(
        node_id,
        plan,
        selected_indices,
        mode="continuous",
        workflow_name=workflow_name,
        variant=segment_slots.VARIANT_SECOND,
        out_dir=out_dir,
    )
    return {
        "files": res.get("files") or [],
        "skipped": res.get("skipped") or [],
        "export_dir": res.get("export_dir") or (out_dir or ""),
    }
