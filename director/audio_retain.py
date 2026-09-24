"""「保留音频」— pin an extracted clip into sampling, then hand it back verbatim.

What it does
------------

The「音频」tab can mark one extracted entry per timeline card as *retained*
(``timeline.segments[i].retainAudioId``). For such a segment the run:

1. **Phase 2** VAE-encodes that PCM into the AV latent's **audio stream** and
   zeroes that stream's ``noise_mask``, so the UNet sees it — and therefore
   conditions the video on it — but can never re-draw it;
2. **Phase 3** skips the audio VAE decode entirely and muxes *the same PCM*,
   so what you hear is bit-for-bit the clip you picked, never a round-trip
   through the audio VAE.

Why encode instead of reusing the stored audio latent
-----------------------------------------------------

``audio_extract`` also saves an audio latent (``audio_<id>_latent.pt``), but its
T was sized for whatever the segment exported *last time*. The latent a run
needs is ``T = round(sample_len / fps * AUDIO_HZ)``, and ``sample_len`` differs
from the export length whenever「段间引导」is on (the head pin adds frames that
are sampled but not exported). Re-encoding from the PCM sidesteps that mismatch
completely — it is one cheap ``audio_vae.encode`` and it always lands on the
current grid.

Note the audio stream is independent of the canvas: its shape is
``[1, C, 2, T]`` with T driven only by the frame count, never by width/height.
So a retained clip keeps working when the card moves to a different canvas,
which the video pin cannot do (it drops to the pixel path on a canvas change).

Head pin offset
---------------

A segment is sampled longer than it is exported: 「段间引导」pins the previous
segment's tail at the head and that prefix is cut away after decoding. The
retained clip therefore has to start where the *export* starts, not at t=0 —
otherwise the muxed track runs ``trim_frames / fps`` seconds ahead of the audio
the UNet actually listened to. Callers pass that offset as ``head_seconds`` and
the clip is written from the matching audio tick onwards; the head region keeps
whatever the continuity pass pinned there.

Masking
-------

``noise_mask`` is a NestedTensor with the same two streams as ``samples``
(precedent: :func:`h3_latent_continue.apply_latent_continue`). Video keeps
whatever the continuity pass left there; audio goes to ``0`` (fully locked).

Keeping ``PREFIX_STEPS_KEY``
----------------------------

When「段间锥形重绘」ran first it leaves ``PREFIX_STEPS_KEY`` on the latent, so
:func:`sample_single_stage` installs the continue remask on top of our mask.
That is safe and must **not** be undone:

* ``comfy.samplers.KSampler.sample`` packs the two mask streams into one flat
  ``[B, 1, video_flat + audio_flat]`` tensor, and the remask (``_PrefixRemask``)
  only rewrites the leading ``video_flat`` slice — the audio half is untouched;
* the hard lock is enforced one level up by ``KSamplerX0Inpaint``, which mixes
  ``x`` and the denoised output against that packed mask. The remask's
  ``apply_model`` hook does hand the transformer a video-only 5-D mask, but that
  one never participates in the ``latent_image`` mixing that pins the audio.

So a retained clip stays locked even under 锥形重绘, and the per-sigma seam
refinement keeps working.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.audio_retain")

#: Per-segment timeline field holding the retained entry id ("" = not retained).
RETAIN_KEY = "retainAudioId"
#: Fallback when the audio VAE does not advertise its rate.
DEFAULT_VAE_SR = 32000


# --------------------------------------------------------------------------
# Reading the switch
# --------------------------------------------------------------------------

def build_retain_index(plan: Any) -> dict[int, str]:
    """``{plan segment index: entry id}`` for every card with「保留音频」ticked.

    Matched through ``seg.timeline_index`` rather than the plan index, because
    under「选择运行」the plan index is the compact run order while the switch
    lives on the timeline card.
    """
    raw = getattr(plan, "raw", None) or {}
    cards = raw.get("segments")
    if not isinstance(cards, list) or not cards:
        return {}
    out: dict[int, str] = {}
    for seg in (getattr(plan, "segments", None) or []):
        ui = int(getattr(seg, "timeline_index", getattr(seg, "index", -1)))
        if not 0 <= ui < len(cards):
            continue
        card = cards[ui]
        if not isinstance(card, dict):
            continue
        entry = str(card.get(RETAIN_KEY) or "").strip()
        if entry:
            out[int(seg.index)] = entry
    return out


def load_retain_pcm(
    node_id: str | None,
    workflow_name: str | None,
    entry_id: str,
) -> dict[str, Any] | None:
    """Load a retained entry's WAV as an AUDIO dict. ``None`` on any failure."""
    if not entry_id:
        return None
    try:
        from ..lib.audio_io import load_reference_audio
        from .audio_extract import resolve_audio_file

        path = resolve_audio_file(node_id, workflow_name, entry_id)
        if path is None:
            return None
        audio = load_reference_audio(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("保留音频: 条目 %s 读取失败 (%s)。", entry_id, exc)
        return None
    if not isinstance(audio, dict):
        return None
    wave = audio.get("waveform")
    if not isinstance(wave, torch.Tensor) or int(wave.numel()) <= 0:
        return None
    return {
        "waveform": wave.detach().cpu().float().contiguous(),
        "sample_rate": int(audio.get("sample_rate") or 0) or DEFAULT_VAE_SR,
    }


def build_retain_audio_cache(
    node_id: str | None,
    plan: Any,
    *,
    workflow_name: str | None = None,
    run_list: list[Any] | None = None,
) -> dict[int, dict[str, Any]]:
    """Phase 1: pull every retained clip's PCM into memory.

    Returns ``{seg.index: {"pcm": audio_dict, "entry_id": str}}``. A segment
    whose clip is missing (deleted from disk, entry removed) is simply absent —
    it then runs as a normal generate segment instead of failing the run.
    """
    index = build_retain_index(plan)
    if not index:
        return {}
    cache: dict[int, dict[str, Any]] = {}
    for seg in (run_list if run_list is not None else (getattr(plan, "segments", None) or [])):
        entry_id = index.get(int(getattr(seg, "index", -1)))
        if not entry_id:
            continue
        pcm = load_retain_pcm(node_id, workflow_name, entry_id)
        if pcm is None:
            log.warning(
                "保留音频 #%d: 条目 %s 不可用，本次按正常生成处理。",
                int(getattr(seg, "index", 0)) + 1, entry_id,
            )
            continue
        cache[int(seg.index)] = {"pcm": pcm, "entry_id": entry_id}
    return cache


# --------------------------------------------------------------------------
# Writing it into the AV latent
# --------------------------------------------------------------------------

def _streams(samples: Any) -> list[torch.Tensor]:
    from .h3_motion_context import _streams_from_latent

    return list(_streams_from_latent({"samples": samples}))


def _nested(video: torch.Tensor, audio: torch.Tensor, template: Any = None):
    from .h3_latent_continue import _nested as _join

    return _join(video, audio, template)


def _fit_waveform(wave: torch.Tensor, sr: int, vae_sr: int, want: int) -> torch.Tensor:
    """Resample to the VAE rate, then head-trim / zero-pad to exactly ``want``.

    Returns ``[B, C, want]`` — the channel axis must survive, because the audio
    VAE encodes ``wave[:1].movedim(1, -1)`` (i.e. ``[B, T, C]``).
    """
    if sr != vae_sr and sr > 0:
        try:
            import torchaudio

            wave = torchaudio.functional.resample(wave, sr, vae_sr)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("保留音频: %dHz → %dHz 重采样失败 (%s)，按原采样率送入。", sr, vae_sr, exc)

    if wave.ndim == 1:
        wave = wave.view(1, 1, -1)
    elif wave.ndim == 2:                      # [C, T] → [1, C, T]
        wave = wave.unsqueeze(0)
    elif wave.ndim != 3:                      # anything odd → mono
        wave = wave.reshape(1, 1, -1)

    have = int(wave.shape[-1])
    if have >= want:
        return wave[..., :want].contiguous()
    pad = wave.new_zeros((int(wave.shape[0]), int(wave.shape[1]), want - have))
    return torch.cat([wave, pad], dim=-1).contiguous()


def _audio_hz() -> float:
    """Audio ticks per second (``h3_motion_context.AUDIO_HZ``)."""
    try:
        from .h3_motion_context import AUDIO_HZ

        hz = float(AUDIO_HZ)
    except Exception:  # pragma: no cover - defensive
        hz = 40.0
    return hz if hz > 0 else 40.0


def _latent_audio_seconds(audio: torch.Tensor) -> float:
    """How many seconds the latent's own audio stream spans.

    Used when the caller cannot say (the second pass has no ``sample_len`` of its
    own — it inherits whatever the upscaler produced). The audio grid is
    ``AUDIO_HZ`` ticks per second and, unlike the video stream, does not depend
    on the canvas at all.
    """
    return float(int(audio.shape[-1])) / _audio_hz()


def apply_retain_audio(
    latent: dict[str, Any],
    audio_vae: Any,
    pcm: dict[str, Any],
    *,
    sample_len: int | None = None,
    fps: float = 24.0,
    seconds: float | None = None,
    head_seconds: float = 0.0,
) -> dict[str, Any]:
    """Encode ``pcm`` into ``latent``'s audio stream and lock that stream.

    ``head_seconds`` is the length of the head pin 段间引导 prepends and the
    export cuts away again; the clip starts after it so the exported frames line
    up with the audio the UNet conditioned on.

    Raises on anything unexpected so the caller can fall back to a normal
    generate segment rather than sampling garbage.
    """
    if not isinstance(latent, dict) or latent.get("samples") is None:
        raise ValueError("保留音频: latent 为空。")
    if audio_vae is None:
        raise ValueError("保留音频: 未连接音频 VAE (audio_vae)。")

    wave = pcm.get("waveform") if isinstance(pcm, dict) else None
    if not isinstance(wave, torch.Tensor) or int(wave.numel()) <= 0:
        raise ValueError("保留音频: PCM 为空。")

    streams = _streams(latent["samples"])
    if len(streams) < 2:
        raise ValueError("保留音频: 目标 latent 没有音频流。")
    video = streams[0]
    audio = streams[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)

    sr = int(pcm.get("sample_rate") or 0) or DEFAULT_VAE_SR
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", DEFAULT_VAE_SR) or DEFAULT_VAE_SR)
    if seconds is None and sample_len:
        seconds = float(sample_len) / float(fps or 24.0)
    if seconds is None or seconds <= 0:
        seconds = _latent_audio_seconds(audio)

    # Where the export starts. The clip is written from here on; the ticks before
    # it keep the continuity audio (and are cut away after decoding anyway).
    total_t = int(audio.shape[-1])
    head = float(head_seconds or 0.0)
    head_ticks = int(round(head * _audio_hz())) if head > 0 else 0
    head_ticks = max(0, min(head_ticks, max(0, total_t - 1)))
    span_seconds = max(0.0, float(seconds) - head)
    want = max(1, int(round(span_seconds * vae_sr)))
    fitted = _fit_waveform(wave, sr, vae_sr, want)

    z = audio_vae.encode(fitted[:1].movedim(1, -1))
    if not torch.is_tensor(z):
        raise ValueError("保留音频: 音频 VAE 编码未返回张量。")

    # The audio stream is [B, C, 2, T]; some VAE builds return the 2 packed into
    # the channel axis as [B, C*2, T]. Unpack to the stream's layout — but only
    # when the channel counts actually say so, or a plain [B, C, T] result would
    # be silently split in half.
    if (
        z.ndim == 3
        and audio.ndim == 4
        and int(z.shape[1]) == int(audio.shape[1]) * int(audio.shape[2])
    ):
        try:
            z = z.reshape(int(z.shape[0]), int(audio.shape[1]), int(audio.shape[2]), int(z.shape[-1]))
        except Exception:  # pragma: no cover - defensive
            pass

    patched_audio = audio.clone()
    room = total_t - head_ticks
    t = min(room, int(z.shape[-1]))
    if t < 1:
        raise ValueError("保留音频: 编码结果长度为 0。")
    if head_ticks == 0 and tuple(z.shape) == tuple(audio.shape):
        patched_audio = z.to(device=audio.device, dtype=audio.dtype)
    else:
        patched_audio[..., head_ticks : head_ticks + t] = z[..., :t].to(
            device=audio.device, dtype=audio.dtype
        )
        if head_ticks + t < total_t:
            # The clip is shorter than this segment: keep the tail silent rather
            # than letting the (never-decoded) leftover stream leak in.
            patched_audio[..., head_ticks + t :] = 0.0

    # --- mask: video fully sampled, audio fully locked -------------------
    existing_video_mask = None
    existing = latent.get("noise_mask")
    if existing is not None:
        try:
            parts = _streams(existing) if not torch.is_tensor(existing) else [existing]
        except Exception:
            parts = [existing] if torch.is_tensor(existing) else []
        if parts and torch.is_tensor(parts[0]) and int(parts[0].ndim) >= 3:
            existing_video_mask = parts[0]
    if existing_video_mask is None:
        existing_video_mask = torch.ones(
            (1, 1, int(video.shape[2]), int(video.shape[3]), int(video.shape[4])),
            device=patched_audio.device,
            dtype=torch.float32,
        )
    audio_mask = torch.zeros(
        (1, 1, 1, int(audio.shape[-1])),
        device=patched_audio.device,
        dtype=torch.float32,
    )

    out = dict(latent)
    template = latent.get("samples")
    out["samples"] = _nested(video, patched_audio, template)
    out["noise_mask"] = _nested(existing_video_mask, audio_mask, template)
    # PREFIX_STEPS_KEY / CONTINUE_SEAM_KEY are deliberately left alone: the
    # continue remask only rewrites the video half of the packed mask, so the
    # audio lock survives it (see the module docstring).
    return out
