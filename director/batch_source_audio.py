"""Source-mode audio: extract each segment's soundtrack once, up front.

In ``audio_mode = source`` the exported soundtrack comes from the source video,
not from the AV latent. The batch pipeline used to extract it in Phase 3, from
file, once per segment. This module extracts the PCM for every running segment
during Phase 1, so the source file is read a single time and Phase 3 (plus the
per-segment mp4 mux) reuses the in-memory tensor.

Scope note: only PCM is cached. The audio *latent* is not encoded here — the
sampler's audio conditioning is built from the reference audios
(``seg.ref_audios`` / reference-video soundtracks) by
``batch_prepare.encode_audio_vae_batch``, which remains unchanged.

Frame-length alignment happens in exactly one place: Phase 3
(:func:`batch_phases._decode_export_one_segment`) trims/pads the cached PCM to
the exported frame count and anchors it at ``seg.start_frame``.
"""

from __future__ import annotations

import logging

import torch

from ..lib.audio_io import extract_timeline_audio, frames_to_audio_samples

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.batch_source_audio")


def align_pcm_to_frames(pcm: dict, frame_count: int, fps: float) -> dict:
    """Trim/pad source PCM to exactly ``frame_count`` video frames.

    Cuts the tail (never the head) so the PCM stays anchored at the segment's
    timeline start — the continuity head pin lives *before* that point on the
    timeline, so it is not part of this segment's source audio.
    """
    sr = int(pcm.get("sample_rate") or 44100)
    wave = pcm["waveform"]
    target = frames_to_audio_samples(max(0, int(frame_count)), fps, sr)
    have = int(wave.shape[-1])
    if target <= 0:
        return {"waveform": wave[..., :0], "sample_rate": sr}
    if have == target:
        return {"waveform": wave, "sample_rate": sr}
    if have > target:
        return {"waveform": wave[..., :target].contiguous(), "sample_rate": sr}
    pad = torch.zeros(
        1, int(wave.shape[1]), target - have, dtype=wave.dtype, device=wave.device
    )
    return {"waveform": torch.cat([wave, pad], dim=-1), "sample_rate": sr}


def extract_segment_source_pcm(seg, plan, fps: float) -> dict | None:
    """Extract the source PCM for one segment's logical frame window.

    ``plan.raw`` is the parsed timeline — the same dict
    :mod:`audio_export` hands to ``extract_timeline_audio`` — so the
    frame → source-file mapping matches the rest of the pipeline exactly.
    """
    timeline = getattr(plan, "raw", None) or {}
    start_frame = int(getattr(seg, "start_frame", 0) or 0)
    end_frame = int(getattr(seg, "end_frame", start_frame) or start_frame)

    pcm = extract_timeline_audio(timeline, start_frame, end_frame, fps)
    if pcm is None:
        log.warning(
            "Seg #%d: no source audio for frames [%d:%d]",
            int(seg.index) + 1, start_frame, end_frame,
        )
        return None

    log.info(
        "Seg #%d: source audio PCM cached (%.2fs, %d samples, %d Hz)",
        int(seg.index) + 1,
        pcm["waveform"].shape[-1] / max(1, int(pcm["sample_rate"])),
        pcm["waveform"].shape[-1],
        pcm["sample_rate"],
    )
    return pcm


def build_source_audio_cache(run_list, plan, fps: float) -> dict[int, dict]:
    """Phase 1: extract every running segment's source PCM into memory.

    Returns ``{seg.index: {"pcm": pcm_dict}}``. Segments whose audio cannot be
    extracted are simply absent; Phase 3 then falls back to the regular decode
    path (empty audio → silent), never to a fabricated track.
    """
    cache: dict[int, dict] = {}
    total = len(run_list)
    log.info("Source audio: extracting PCM for %d segment(s) up front...", total)

    for pos, seg in enumerate(run_list, 1):
        try:
            pcm = extract_segment_source_pcm(seg, plan, fps)
        except Exception as exc:
            log.error(
                "Seg #%d/%d: source audio extract failed: %s",
                pos, total, exc, exc_info=True,
            )
            continue
        if pcm is not None:
            cache[seg.index] = {"pcm": pcm}

    log.info("Source audio: %d/%d segment(s) cached for reuse", len(cache), total)
    return cache
