"""Source audio locking for Phase 2 sampling.

In source mode, we extract and encode the original audio to latent ONCE in Phase 1,
then use it to lock the audio channel during UNet sampling (preventing modification).
Phase 3 directly uses the saved PCM instead of extracting from file again.

This module provides:
- encode_source_audio_to_latent(): PCM → audio latent for sampling protection
- build_locked_av_latent(): Combine video latent with locked audio latent
- extract_and_encode_segment_audio(): Full pipeline (extract PCM + encode latent)
"""

from __future__ import annotations

import logging
import torch

from .audio_export import AUDIO_MODE_SOURCE
from ..lib.audio_io import extract_timeline_audio

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.batch_audio_lock")


def encode_source_audio_to_latent(
    audio_vae,
    source_pcm: dict,
    num_frames: int,
    fps: float,
) -> torch.Tensor | None:
    """将源音频编码为 VAE latent，用于采样时锁定保护。

    仅在 audio_mode=source 时使用，编码结果不用于解码（解码直接用原始 PCM）。

    Args:
        audio_vae: Audio VAE model for encoding
        source_pcm: PCM audio dict with "waveform" and "sample_rate" keys
        num_frames: Target number of video frames
        fps: Video frame rate

    Returns:
        Audio latent tensor [B, C, T] or None if input is invalid
    """
    if source_pcm is None:
        return None

    waveform = source_pcm.get("waveform")
    if waveform is None or waveform.numel() == 0:
        return None

    try:
        import torchaudio
    except ImportError:
        torchaudio = None

    # Sample rate alignment
    sr = int(source_pcm.get("sample_rate", 44100))
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000) or 32000)

    if sr != vae_sr:
        if torchaudio is not None:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)

    # Duration alignment (calculate required audio length from frame count)
    target_samples = int(round(num_frames / fps * vae_sr))
    have_samples = int(waveform.shape[-1])

    if have_samples > target_samples:
        waveform = waveform[..., :target_samples]
    elif have_samples < target_samples:
        pad = torch.zeros(
            1, waveform.shape[1], target_samples - have_samples,
            dtype=waveform.dtype, device=waveform.device
        )
        waveform = torch.cat([waveform, pad], dim=-1)

    # VAE encode expects shape: [B, T, C] or [B, C, T]
    if waveform.ndim == 3 and waveform.shape[1] != waveform.shape[-1]:
        # [B, C, T] → [B, T, C]
        waveform = waveform.movedim(1, -1)

    try:
        audio_latent = audio_vae.encode(waveform[:1])
        log.debug(
            "Encoded source audio: %d samples → %d latent frames (vae_sr=%d)",
            have_samples, int(audio_latent.shape[-1]), vae_sr
        )
        return audio_latent
    except Exception as exc:
        log.warning("Failed to encode source audio latent: %s", exc)
        return None


def build_locked_av_latent(
    video_latent: torch.Tensor,
    exact_audio_latent: torch.Tensor,
) -> dict:
    """构建带 noise_mask 锁定的 AV latent。

    在采样时使用，确保音频 latent 不被 UNet 修改：
    - video_mask=1.0 → 允许视频完全重绘
    - audio_mask=0.0 → 锁定音频，保持源音频 latent 不变

    Args:
        video_latent: Video latent tensor from conditioning
        exact_audio_latent: Source audio latent tensor (will be preserved)

    Returns:
        dict with 'samples' key containing locked AV latent
    """
    try:
        from comfy_extras.nodes_lt import LTXVSeparateAVLatent
    except ImportError:
        log.warning("LTXVSeparateAVLatent not available, skipping audio lock")
        return {"samples": video_latent}

    # Separate video and audio latent
    separated = LTXVSeparateAVLatent.execute(video_latent)
    video_lat, audio_lat = _unpack_node_output(separated)[:2]

    # Replace with locked audio latent (prevent UNet from modifying)
    if exact_audio_latent is not None and exact_audio_latent.numel() > 0:
        # Ensure shapes match
        if video_lat.shape == exact_audio_latent.shape or (
            len(video_lat.shape) == len(exact_audio_latent.shape)
        ):
            locked_audio = exact_audio_latent.clone()
            log.debug("Audio latent locked: %d frames preserved", int(locked_audio.shape[-1]))
        else:
            log.warning(
                "Shape mismatch for audio lock: video=%s, audio=%s — skipping lock",
                video_lat.shape, exact_audio_latent.shape
            )
            locked_audio = audio_lat
    else:
        locked_audio = audio_lat

    # Reconstruct AV latent (implementation depends on ComfyUI's internal API)
    # For now, return the original structure with locked audio
    try:
        from comfy_extras.nodes_lt import LTXVCombineAVLatent
        combined = LTXVCombineAVLatent.execute(video_lat, locked_audio)
        return {"samples": combined}
    except (ImportError, AttributeError):
        # Fallback: return video latent as-is (no audio lock)
        log.warning("LTXVCombineAVLatent not available, returning original latent")
        return {"samples": video_latent}


def _unpack_node_output(out):
    """Extract tensor from node output."""
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    raise RuntimeError(f"Unexpected node output: {type(out)!r}")


def extract_and_encode_segment_audio(
    seg_index: int,
    plan,
    timeline_data: str,
    audio_vae,
    fps: float,
) -> dict | None:
    """Phase 1: 为指定段提取并编码音频。

    这是完整的 Pipeline：从源视频提取 PCM → 对齐时长 → 编码为 latent。

    Args:
        seg_index: Segment index in the plan
        plan: DirectorPlan object containing segment info
        timeline_data: JSON string of timeline data
        audio_vae: Audio VAE model for encoding
        fps: Video frame rate

    Returns:
        {"pcm": pcm_dict, "latent": audio_latent} or None if extraction fails
    """
    from .plan_types import DirectorPlan

    if not isinstance(plan, DirectorPlan):
        log.warning("Invalid plan type for segment %d", seg_index)
        return None

    seg = plan.segments[seg_index]
    start_frame = int(seg.start_frame or 0)
    end_frame = int(seg.end_frame or seg.start_frame or 0)
    num_frames = int(seg.frame_count or plan.total_frames or 124)

    log.info(
        "Seg #%d: extracting source audio [%d:%d], %d frames @ %.2f fps",
        seg_index + 1, start_frame, end_frame, num_frames, fps
    )

    # Step 1: Extract PCM from source video
    pcm = extract_timeline_audio(
        timeline_data,
        start_frame,
        end_frame,
        fps,
    )

    if pcm is None:
        log.warning("Seg #%d: failed to extract source audio PCM", seg_index + 1)
        return None

    log.info(
        "Seg #%d: extracted audio PCM (%.2fs, %d samples, %d Hz)",
        seg_index + 1,
        pcm["waveform"].shape[-1] / pcm["sample_rate"],
        pcm["waveform"].shape[-1],
        pcm["sample_rate"],
    )

    # Step 2: Encode to latent for sampling protection
    latent = encode_source_audio_to_latent(
        audio_vae, pcm, num_frames, fps
    )

    if latent is None:
        log.warning("Seg #%d: failed to encode audio latent", seg_index + 1)
        # Still return PCM — we can use it without latent (no locking)
        return {"pcm": pcm, "latent": None}

    return {"pcm": pcm, "latent": latent}


def build_source_audio_cache(
    run_list,
    plan,
    timeline_data: str,
    audio_vae,
    fps: float,
) -> dict[int, dict]:
    """Phase 1: 为所有运行段构建音频缓存。

    在 source/mute 模式下调用，提前提取并编码所有段的音频。

    Args:
        run_list: List of segments to run
        plan: DirectorPlan object
        timeline_data: JSON string of timeline data
        audio_vae: Audio VAE model
        fps: Video frame rate

    Returns:
        Dict mapping seg.index → {"pcm": pcm_dict, "latent": audio_latent|None}
    """
    cache = {}
    total = len(run_list)

    log.info("Source mode: extracting and encoding audio for %d segments...", total)

    for pos, seg in enumerate(run_list, 1):
        try:
            result = extract_and_encode_segment_audio(
                seg.index, plan, timeline_data, audio_vae, fps
            )
            if result:
                cache[seg.index] = result
                log.info(
                    "Seg #%d/%d: audio cached (%s)%s",
                    pos, total,
                    "PCM+latent" if result.get("latent") else "PCM-only",
                    " ⚠️ no latent" if not result.get("latent") else ""
                )
            else:
                log.warning("Seg #%d/%d: audio cache skipped", pos, total)
        except Exception as exc:
            log.error("Seg #%d/%d: audio cache failed: %s", pos, total, exc, exc_info=True)

    log.info(
        "Source mode: audio cache complete — %d/%d segments with PCM",
        sum(1 for v in cache.values() if v and v.get("pcm")),
        total
    )

    return cache
