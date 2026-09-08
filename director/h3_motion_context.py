"""Director-owned MiniMax H3 segment motion/audio continuation helpers.

Pins the previous segment's tail into the next segment as never-denoised
conditioning, and optionally pins the next segment's opening into this one's
tail. A pin makes the model *replay* the neighbouring segment at this sample's
edge, so a segment is sampled as a replayed head, its own body, and an optional
replayed tail — and only the body is exported. Inspired by the community Motion
Context approach; original Apache-2.0 code for this Director.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .frame_align import minimax_align_frame_count
from .h3_context_patches import (
    CTX_AUDIO_END_KEY,
    CTX_FRAME_KEY,
    ensure_layout_patch,
    ensure_payload_patch,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.h3_motion_context")

FPS = 24.0
AUDIO_HZ = 40.0
FRAME_RESCALE = 5.0 / 3.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

# Pixel windows that map to a whole number of VAE latent steps from cycle 0.
CONTEXT_FRAME_CHOICES = (5, 22, 39, 56)
DEFAULT_CONTEXT_FRAMES = 22
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)
#: Tail-pin length used for「对齐下段」(align-to-next).
#:
#: Deliberately *not* the widget's context length. The tail pin is replayed at
#: the end of the sample and then dropped, so a longer one only buys extra
#: sampling — the smallest step is enough to forge the join from both sides.
#: The head pin is the opposite: it is what carries the continuity you actually
#: see, so keep choosing it on merit from ``CONTEXT_FRAME_CHOICES``.
TAIL_CONTEXT_FRAMES = 5

CONTINUITY_TASK_KEYS = frozenset({"t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"})
# v8: v7 + export audio cache + fps in fingerprint + trim hydrate on partial re-run.
# v9: v8 + export length snapped down onto the 17-frame VAE cycle grid, so the
#     next segment's pin window ends exactly on the last exported frame and
#     gap_after_pin is always 0 (no trimmed frames, no seam echo). Cached
#     exports from v8 and earlier hold 17k+5 frames and must not be reused.
# v10: v9's export snap reverted — a segment exports its own aligned length
#      (17k+5) again instead of 17k, and the tail pin gets its room reserved by
#      ``generation_frame_budget`` rather than taken out of the free zone.
#      v9/v8 exports are 5 frames short and v8-v9 tail pins ate the ending;
#      reusing any of them would bring back the clipped speech.
# Single source of truth — imported by segment_cache.segment_cache_fingerprint.
CONTINUITY_PIPELINE_ID = "minimax_h3_motion_context_v11"
# Example workflow tested value (NikoDemon80): audio_context_length=24 with video=22.
DEFAULT_AUDIO_CONTEXT_FRAMES = 24


def snap_context_frames(raw: int | float | None) -> int:
    """Snap UI/plan overlap to a supported context window (official baseline 22)."""
    try:
        n = int(raw or DEFAULT_CONTEXT_FRAMES)
    except (TypeError, ValueError):
        n = DEFAULT_CONTEXT_FRAMES
    chosen = min(CONTEXT_FRAME_CHOICES, key=lambda g: (abs(g - n), -g))
    return int(chosen)


def snap_tail_context_frames(raw: int | float | None, max_frames: int) -> int:
    """Largest legal tail-pin length that fits the sample's spare tail capacity.

    The tail pin lives in the frames the sample already wastes beyond the export
    (``sample - head_pin - export``). Snapping with ``snap_context_frames``
    would round 17 back up to 22 and push the pin into the exported region, so
    the tail is clamped to whole latent steps at or under ``max_frames``
    instead — the UI overlap is honoured whenever the grid allows it.
    """
    try:
        want = int(raw or 0)
    except (TypeError, ValueError):
        return 0
    try:
        cap = int(max_frames or 0)
    except (TypeError, ValueError):
        return 0
    if want <= 0 or cap <= 0:
        return 0
    limit = min(want, cap)
    best = 0
    for k in range(1, 65):
        frames = pixel_frames_for_latent_t(k)
        if frames > limit:
            break
        best = frames
    return int(best)


def resolve_tail_context_length(
    latent: dict, export_frame_count: int, *, context_n: int = 0
) -> int:
    """Frames of the next segment to pin into this segment's tail.

    The pin replays the next segment's opening at the end of this sample, so it
    must live in frames the export does not use — whatever the sample carries
    past ``head_pin + export``. :func:`generation_frame_budget` reserves that
    room when the segment opts into「对齐下段」; this only measures what actually
    fits, because the latent grid rarely lines up exactly.

    Returns 0 when there is no spare tail capacity, which disables「对齐下段」
    rather than shortening the export.
    """
    try:
        samples = latent["samples"]
        latent_t = int(samples.shape[2])
        sample_frames = pixel_frames_for_latent_t(latent_t)
        export_frames = minimax_align_frame_count(int(export_frame_count))
        head_pin = snap_context_frames(context_n)
        spare = sample_frames - export_frames - head_pin
        if spare <= 0:
            return 0
        return int(snap_tail_context_frames(TAIL_CONTEXT_FRAMES, spare))
    except Exception:  # never block a run on an optional alignment
        return 0


def recommended_context_frames(task_key: str | None = None) -> int:
    """Official Motion Context baseline (22) for all continuity tasks."""
    del task_key
    return DEFAULT_CONTEXT_FRAMES


def pixel_frames_for_latent_t(latent_t: int) -> int:
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def steps_for_frames(n: int) -> int | None:
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def step_offsets(latent_t: int) -> list[int]:
    out, acc = [], 0
    for k in range(int(latent_t)):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def step_offsets_from(start_step: int, latent_t: int) -> list[int]:
    """Pixel offsets of ``latent_t`` steps starting at ``start_step``.

    ``step_offsets`` assumes a 0-phase start (step 0, where the first token
    spans 1 frame). A tail pin sits at the end of the sample and can begin on
    any step of the 5-cycle, so each step's span depends on the phase
    ``start_step % 5``. Anchoring a pin at the wrong phase misplaces it by up
    to 3 frames inside the VAE cycle and shows up as a torn seam.
    """
    start = int(start_step)
    base = pixel_frames_for_latent_t(start)
    out, acc = [], 0
    for k in range(int(latent_t)):
        out.append(base + acc)
        acc += FRAME_PER_TOKEN[(start + k) % 5]
    return out


def _streams_from_latent(latent: dict) -> list[torch.Tensor]:
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            f"Director continuity: expected MiniMax H3 AV NestedTensor, got {type(samples)!r}"
        )
    if not parts:
        raise ValueError("Director continuity: AV latent has no streams")
    return parts


def video_from_latent(latent: dict) -> torch.Tensor:
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            f"Director continuity: expected video latent [B,C,T,H,W], got {tuple(video.shape)}"
        )
    return video


def _resize_frames(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    import comfy.utils

    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return samples.movedim(1, -1)


def _phase_aligned_tail_start(
    total_steps: int, n_steps: int, end_frame: int | None
) -> tuple[int, int, int]:
    """Pick a 5-cycle-aligned step start whose pixel window ends at/before ``end_frame``.

    Returns ``(start_step, pin_end_px, gap_after_pin)``.

    When ``end_frame`` is None, use the absolute latent end (official Motion Context).
    Director passes the *exported* end so align() overshoot beyond the visible
    segment is never pinned into the next clip. ``gap_after_pin`` is how many
    exported frames sit *after* the pin window — those must be dropped from the
    previous export before concat, or the next clip's opening will echo them.
    """
    if n_steps > total_steps:
        raise ValueError(
            f"Director continuity: need {n_steps} latent steps, context has {total_steps}."
        )
    if end_frame is None:
        start = total_steps - n_steps
        if start % 5 != 0:
            raise RuntimeError(
                f"Director continuity: tail start cycle {start % 5} != 0; refusing shifted join."
            )
        pin_end = pixel_frames_for_latent_t(total_steps)
        return start, pin_end, 0

    end_limit = int(end_frame)
    best_start = None
    best_end_px = -1
    for start in range(0, total_steps - n_steps + 1, 5):
        start_px = pixel_frames_for_latent_t(start)
        end_px = start_px + pixel_frames_for_latent_t(n_steps)
        if end_px <= end_limit and end_px >= best_end_px:
            best_start = start
            best_end_px = end_px
    if best_start is None:
        raise RuntimeError(
            f"Director continuity: no phase-aligned {n_steps}-step window ending "
            f"at or before frame {end_limit}."
        )
    gap = max(0, end_limit - best_end_px)
    if gap > 0:
        log.info(
            "Director continuity: pin window ends %df before export end "
            "(phase align; export_end=%d, pin_end=%d) — prev export tail will be trimmed",
            gap,
            end_limit,
            best_end_px,
        )
    return best_start, best_end_px, gap


def _video_tail_blocks(
    latent: dict,
    n: int,
    *,
    end_frame: int | None = None,
) -> tuple[list[torch.Tensor], list[int], int, int, int]:
    """Return ``(blocks, offsets, covered, pin_end_px, gap_after_pin)``."""
    video = video_from_latent(latent)
    total = int(video.shape[2])
    steps = steps_for_frames(n)
    if steps is None:
        raise ValueError(
            f"Director continuity: {n} frames is not a whole number of latent steps "
            f"(use {', '.join(str(x) for x in CONTEXT_FRAME_CHOICES)})."
        )
    start, pin_end_px, gap = _phase_aligned_tail_start(total, steps, end_frame)
    covered = pixel_frames_for_latent_t(steps)
    if covered != n:
        raise RuntimeError(
            f"Director continuity: {steps} steps cover {covered} frames, expected {n}."
        )
    blocks = [video[:1, :, start + k : start + k + 1].clone() for k in range(steps)]
    return blocks, step_offsets(steps), covered, pin_end_px, gap


def _video_head_blocks(
    latent: dict,
    n: int,
    *,
    start_px: int = 0,
) -> tuple[list[torch.Tensor], list[int], int]:
    """Return ``(blocks, offsets, covered)`` for ``n`` frames from ``start_px``.

    Mirror of ``_video_tail_blocks``: used to pin the *next* segment's opening
    into the current segment's tail so the join is forged from both sides.
    ``start_px`` is the offset into the *next* sample — the next segment's own
    head pin is trimmed before it is referenced, so we read its trimmed opening
    (``start_px`` = the next segment's ``trim_frames``), not its raw sample head.
    Offsets are relative to the source clip; the caller re-maps them onto the
    target sample timeline with ``step_offsets_from``.
    """
    video = video_from_latent(latent)
    total = int(video.shape[2])
    steps = steps_for_frames(n)
    if steps is None:
        raise ValueError(
            f"Director continuity: {n} frames is not a whole number of latent steps "
            f"(use {', '.join(str(x) for x in CONTEXT_FRAME_CHOICES)})."
        )
    start_step = steps_for_frames(start_px)
    if start_step is None:
        raise ValueError(
            f"Director continuity: tail reference offset {start_px} frames is not a "
            f"whole number of latent steps."
        )
    if start_step + steps > total:
        raise ValueError(
            f"Director continuity: tail reference needs {start_step + steps} latent "
            f"steps, next segment has {total}."
        )
    covered = pixel_frames_for_latent_t(steps)
    if covered != n:
        raise RuntimeError(
            f"Director continuity: {steps} steps cover {covered} frames, expected {n}."
        )
    blocks = [video[:1, :, (start_step + k) : (start_step + k + 1)].clone() for k in range(steps)]
    return blocks, step_offsets(steps), covered


def _audio_tail_from_latent(
    latent: dict,
    a_frames: int,
    *,
    end_frame: int | None = None,
) -> tuple[torch.Tensor, int, float]:
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError("Director continuity: context latent has no audio stream.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError(
            f"Director continuity: expected audio latent [B,C,2,T], got {tuple(audio.shape)}"
        )
    total_t = int(audio.shape[-1])
    frames = pixel_frames_for_latent_t(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    if not (0.0 <= overhang < 1.0):
        log.warning(
            "Director continuity: unexpected audio grid (%d steps / %d frames); "
            "assuming no overhang.",
            total_t,
            frames,
        )
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        log.warning(
            "Director continuity: asked for %d audio steps, latent has %d; pinning all.",
            rt,
            total_t,
        )
        rt = total_t
    if rt < 1:
        raise ValueError("Director continuity: empty audio window")
    if end_frame is None:
        audio_end = total_t
    else:
        # Match the video pin window end (export end), not the sample overshoot.
        audio_end = int(round(float(end_frame) / float(FPS) * AUDIO_HZ))
        audio_end = max(rt, min(total_t, audio_end))
    audio_start = audio_end - rt
    if audio_start < 0:
        audio_start = 0
        rt = audio_end
    return audio[:1, ..., audio_start:audio_end].clone(), rt, float(overhang)


def _usable_context_audio(audio: dict | None) -> dict | None:
    """Return ``audio`` only when it has a non-empty waveform (resample-safe)."""
    if not isinstance(audio, dict):
        return None
    waveform = audio.get("waveform")
    if not isinstance(waveform, torch.Tensor) or waveform.numel() <= 0:
        return None
    if waveform.ndim < 1 or int(waveform.shape[-1]) <= 0:
        return None
    return audio


def _encode_tail_audio(audio_vae, audio: dict, seconds: float) -> tuple[torch.Tensor, int]:
    try:
        import torchaudio
    except ImportError:
        torchaudio = None
    usable = _usable_context_audio(audio)
    if usable is None:
        raise ValueError("Director continuity: empty context audio waveform")
    waveform = usable["waveform"]
    sr = int(usable.get("sample_rate") or getattr(audio_vae, "audio_sample_rate", 32000) or 32000)
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError(
                f"Director continuity: context audio is {sr} Hz, VAE wants {vae_sr} Hz, "
                "and torchaudio is unavailable."
            )
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    want = int(round(seconds * vae_sr))
    have = int(waveform.shape[-1])
    if have >= want:
        waveform = waveform[..., have - want :]
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, int(z.shape[-1])


def _existing_keyframes(positive) -> list[dict]:
    """Best-effort read of keyframes already on conditioning (e.g. fl2v last frame)."""
    try:
        if not positive:
            return []
        meta = positive[0][1] if isinstance(positive[0], (list, tuple)) else None
        if not isinstance(meta, dict):
            return []
        kfs = meta.get("minimax_keyframes") or []
        return [dict(kf) for kf in kfs if isinstance(kf, dict)]
    except Exception:
        return []


def apply_motion_context(
    positive,
    latent: dict,
    *,
    vae,
    context_length: int,
    context_latent: dict | None = None,
    context_frames: torch.Tensor | None = None,
    context_audio: dict | None = None,
    audio_vae=None,
    continue_audio: bool = True,
    keep_existing_keyframes: bool = True,
    context_end_frame: int | None = None,
    audio_context_length: int | None = None,
    tail_context_latent: dict | None = None,
    tail_context_length: int | None = None,
    tail_context_offset: int = 0,
) -> tuple[Any, int, int]:
    """Inject previous-segment motion (and optional audio) into conditioning.

    ``tail_context_latent`` / ``tail_context_length`` pin the *next* segment's
    opening into this segment's tail (align-to-next), mirroring the head pin.
    Both pins coexist: the head is forged by the previous segment, the tail by
    the next one, and only the middle is denoised.

    Returns ``(positive, trim_frames, prev_export_trim_tail)``.

    ``trim_frames`` is how many leading frames to drop after decode: the pin
    makes the model reproduce the previous segment's tail — picture *and* sound
    — at the start of this one, so that prefix is a repeat rather than content.
    It is the *measured* ``span``, not the requested context length: the VAE
    grid can shorten the pin, and trimming more than was actually pinned would
    eat into this segment.
    ``prev_export_trim_tail`` is how many frames to drop from the *previous*
    segment's export before concat (phase-align pin often ends a few frames
    before the export end; leaving them causes a visible ~5f echo at the seam).

    ``context_end_frame``: exclusive pixel index on the previous *sample*
    timeline to pin up to. With official sample semantics (no post-trim crop)
    this is usually ``None`` (absolute latent end). Kept for legacy overshoot
    caches.

    ``audio_context_length``: official example uses 24 with video context 22.
    """
    import node_helpers

    ensure_layout_patch()
    # r2v/v2v/rv2v already carry minimax_refs; stock payload overwrites keyframe
    # video latents unless coexistence merge is installed.
    ensure_payload_patch()
    context_length = snap_context_frames(context_length)
    if audio_context_length is None:
        audio_ctx = DEFAULT_AUDIO_CONTEXT_FRAMES
    else:
        try:
            audio_ctx = max(0, int(audio_context_length))
        except (TypeError, ValueError):
            audio_ctx = DEFAULT_AUDIO_CONTEXT_FRAMES

    video = video_from_latent(latent)
    width = int(video.shape[4]) * 16
    height = int(video.shape[3]) * 16
    frame_count = pixel_frames_for_latent_t(int(video.shape[2]))

    pin_audio_latent = context_latent
    if context_latent is not None:
        src = video_from_latent(context_latent)
        src_w, src_h = int(src.shape[4]) * 16, int(src.shape[3]) * 16
        if src_w == width and src_h == height:
            available = pixel_frames_for_latent_t(int(src.shape[2]))
            if context_end_frame is not None:
                available = min(available, max(0, int(context_end_frame)))
            video_src = "latent"
        elif context_frames is not None and int(context_frames.shape[0]) >= 1:
            # A canvas mismatch (larger stored AV latent vs this segment's canvas)
            # is not a crash: pin from the decoded export instead, which is what
            # concat actually uses.
            log.warning(
                "Director continuity: context latent is %dx%d but this segment "
                "is %dx%d — pin from decoded frames.",
                src_w,
                src_h,
                width,
                height,
            )
            context_end_frame = None
            pin_audio_latent = None
            available = int(context_frames.shape[0])
            video_src = "pixels"
        else:
            raise ValueError(
                f"Director continuity: context latent is {src_w}x{src_h} but this "
                f"segment is {width}x{height}. Regenerate the previous segment at "
                "this resolution, or keep a decoded export for pixel pin."
            )
    else:
        if context_frames is None or int(context_frames.shape[0]) < 1:
            raise ValueError(
                "Director continuity: need previous segment latent or decoded frames."
            )
        available = int(context_frames.shape[0])
        video_src = "pixels"

    n = min(int(context_length), available)
    if n < 1:
        raise ValueError("Director continuity: no frames available to pin")
    run = next(g for g in VIDEO_RUN_GRID if g <= n)
    if run != n:
        log.warning(
            "Director continuity: %d frames off VAE grid; pinning last %d.", n, run
        )
        n = run
    if n >= frame_count:
        raise ValueError(
            f"Director continuity: cannot pin {n} frames into a {frame_count}-frame clip."
        )

    pin_end_px: int | None = None
    prev_export_trim_tail = 0
    if video_src == "latent":
        blocks, offsets, covered, pin_end_px, prev_export_trim_tail = _video_tail_blocks(
            context_latent, n, end_frame=context_end_frame
        )
        span = covered
    else:
        # Decoded frames are already the export — absolute tail is correct.
        tail = _resize_frames(context_frames[available - n :], width, height)
        enc = vae.encode(tail)
        if getattr(enc, "ndim", 0) != 5:
            raise ValueError(
                f"Director continuity: VAE encode returned shape "
                f"{tuple(getattr(enc, 'shape', ()))}, expected [B,C,T,H,W]."
            )
        steps = int(enc.shape[2])
        offsets = step_offsets(steps)
        covered = pixel_frames_for_latent_t(steps)
        if covered != n:
            raise RuntimeError(
                f"Director continuity: {n} frames encoded to {steps} steps covering "
                f"{covered}; VAE grid mismatch."
            )
        blocks = [enc[:, :, k : k + 1] for k in range(steps)]
        span = covered
        pin_end_px = available
        prev_export_trim_tail = 0

    ctx_keyframes = [
        {
            "resolved_frame_index": 0,
            CTX_FRAME_KEY: int(p),
            "latent": blk,
        }
        for p, blk in zip(offsets, blocks)
    ]

    # --- align-to-next: pin the next segment's opening into this tail ---
    # Both pins are "cond" rows flagged never-denoised, and both make the model
    # replay the neighbouring segment at this sample's edge. The head replay is
    # trimmed after decode; the tail replay lives past the export.
    tail_span = 0
    if tail_context_latent is not None and int(tail_context_length or 0) > 0:
        try:
            sample_steps = int(video.shape[2])
            # Caller pre-snaps to the legal value that fits the sample's spare
            # tail capacity (``snap_tail_context_frames``). Snapping here with
            # ``snap_context_frames`` would round 17 back up to 22 and push the
            # pin into the exported region.
            tail_n = int(tail_context_length or 0)
            head_steps = steps_for_frames(span)
            tail_steps = steps_for_frames(tail_n)
            if head_steps is None or tail_steps is None:
                log.warning(
                    "Director continuity: tail pin %df is not a whole number of "
                    "latent steps; skipping align-to-next.",
                    tail_n,
                )
            elif head_steps + tail_steps >= sample_steps:
                log.warning(
                    "Director continuity: head pin %d + tail pin %d steps do not "
                    "fit in a %d-step sample; skipping align-to-next.",
                    head_steps,
                    tail_steps,
                    sample_steps,
                )
            else:
                tail_steps_val = int(tail_steps)
                t_blocks, _t_src_offsets, t_covered = _video_head_blocks(
                    tail_context_latent, tail_n, start_px=tail_context_offset
                )
                t_video = video_from_latent(tail_context_latent)
                if (
                    int(t_video.shape[3]) != int(video.shape[3])
                    or int(t_video.shape[4]) != int(video.shape[4])
                ):
                    log.warning(
                        "Director continuity: next segment latent is %dx%d but this "
                        "segment is %dx%d; skipping align-to-next.",
                        int(t_video.shape[4]) * 16,
                        int(t_video.shape[3]) * 16,
                        width,
                        height,
                    )
                else:
                    tail_start = sample_steps - tail_steps_val
                    t_offsets = step_offsets_from(tail_start, tail_steps_val)
                    ctx_keyframes.extend(
                        {
                            "resolved_frame_index": 0,
                            CTX_FRAME_KEY: int(p),
                            "latent": blk,
                        }
                        for p, blk in zip(t_offsets, t_blocks)
                    )
                    tail_span = int(t_covered)
                    log.info(
                        "Director continuity: pinned next-segment head %df "
                        "(%d steps at sample step %d-%d) into this tail.",
                        tail_span,
                        tail_steps_val,
                        tail_start,
                        sample_steps - 1,
                    )
        except Exception as exc:  # never break a run for an optional alignment
            log.warning("Director continuity: align-to-next unavailable (%s).", exc)

    merged = list(ctx_keyframes)
    if keep_existing_keyframes:
        for kf in _existing_keyframes(positive):
            # Drop stock first-frame at 0 — replaced by context head.
            if int(kf.get("resolved_frame_index", -1)) == 0 and CTX_FRAME_KEY not in kf:
                continue
            # Avoid duplicating director context markers.
            if CTX_FRAME_KEY in kf:
                continue
            # fl2v keeps the stock last_frame keyframe. Mark it with its own
            # resolved_frame_index (same pixel-index space as CTX_FRAME_KEY) so
            # _rewrite_keyframe_times can re-time it when references shift the
            # target origin, instead of skipping it and tripping the guard.
            rfi = int(kf.get("resolved_frame_index", -1))
            if rfi >= 0:
                kf = dict(kf)
                kf[CTX_FRAME_KEY] = rfi
            merged.append(kf)

    values: dict[str, Any] = {
        "minimax_keyframes": merged,
        "minimax_frame_count": frame_count,
    }
    # No ``minimax_tail_trim_frames``: the tail replay lives past the export
    # (``generation_frame_budget`` reserves the room), so the export simply stops
    # before it — there is nothing to trim from a region that was never written.
    # The span is logged below instead.
    out = node_helpers.conditioning_set_values(positive, values)

    context_audio = _usable_context_audio(context_audio)
    if continue_audio and pin_audio_latent is None and context_audio is None:
        log.warning(
            "Director continuity: previous export audio is empty; pinning video only."
        )
    if continue_audio and (pin_audio_latent is not None or context_audio is not None):
        # Official: audio window independent; 0 follows video span. Example WF uses 24.
        a_frames = int(audio_ctx) if audio_ctx > 0 else int(span)
        # Align audio pin end with the video pin window (not export overshoot).
        audio_end_limit = pin_end_px if pin_end_px is not None else context_end_frame
        if pin_audio_latent is not None:
            audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(
                pin_audio_latent, a_frames, end_frame=audio_end_limit
            )
        else:
            if audio_vae is None:
                raise ValueError(
                    "Director continuity: context_audio requires audio_vae "
                    "(or pass previous AV latent)."
                )
            audio_latent, ref_audio_t = _encode_tail_audio(
                audio_vae, context_audio, a_frames / float(FPS)
            )
            overhang = 0.0
        end_frame = float(span) + float(overhang) / FRAME_RESCALE
        end_coord = round(FRAME_RESCALE * end_frame)
        end_frame = end_coord / FRAME_RESCALE
        audio_ref = {
            "kind": "audio",
            "ref_audio_t": ref_audio_t,
            "audio_latent": audio_latent,
            CTX_AUDIO_END_KEY: end_frame,
        }
        out = node_helpers.conditioning_set_values(
            out, {"minimax_refs": [audio_ref]}, append=True
        )
        log.info(
            "Director continuity: pinned %d video frames (%s) + %d audio steps"
            "%s%s",
            span,
            video_src,
            ref_audio_t,
            f" (context_end={context_end_frame})" if context_end_frame is not None else "",
            f", trim_prev_export={prev_export_trim_tail}f" if prev_export_trim_tail else "",
        )
    else:
        log.info(
            "Director continuity: pinned %d video frames (%s), audio off%s%s",
            span,
            video_src,
            f" (context_end={context_end_frame})" if context_end_frame is not None else "",
            f", trim_prev_export={prev_export_trim_tail}f" if prev_export_trim_tail else "",
        )

    return out, int(span), int(prev_export_trim_tail)


def trim_context_prefix(
    images: torch.Tensor,
    audio: dict | None,
    trim_frames: int,
    *,
    fps: float = FPS,
    match_tail: bool = True,
) -> tuple[torch.Tensor, dict | None]:
    """Remove pinned head from decoded images/audio; optionally match audio duration."""
    trim = max(0, int(trim_frames))
    if trim > 0:
        if int(images.shape[0]) <= trim:
            raise ValueError(
                f"Director continuity: cannot trim {trim} frames from "
                f"{int(images.shape[0])}-frame decode."
            )
        images = images[trim:]
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return images, audio
    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate") or 32000)
    drop = int(round((trim / float(fps)) * sr)) if trim > 0 else 0
    if drop > 0 and int(waveform.shape[-1]) > drop:
        waveform = waveform[..., drop:]
    if match_tail:
        want = int(round((int(images.shape[0]) / float(fps)) * sr))
        if int(waveform.shape[-1]) > want:
            waveform = waveform[..., :want]
    return images, {"waveform": waveform, "sample_rate": sr}


def trim_export_tail(
    images: torch.Tensor,
    audio: dict | None,
    trim_frames: int,
    *,
    fps: float = FPS,
) -> tuple[torch.Tensor, dict | None]:
    """Drop trailing frames from a previous export so it ends at the pin window."""
    trim = max(0, int(trim_frames))
    if trim <= 0:
        return images, audio
    keep = int(images.shape[0]) - trim
    if keep < 1:
        raise ValueError(
            f"Director continuity: cannot drop {trim} tail frames from "
            f"{int(images.shape[0])}-frame export."
        )
    images = images[:keep]
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return images, audio
    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate") or 32000)
    want = int(round((keep / float(fps)) * sr))
    if int(waveform.shape[-1]) > want:
        waveform = waveform[..., :want]
    return images, {"waveform": waveform, "sample_rate": sr}


def generation_frame_budget(
    visible_frames: int,
    context_frames: int = 0,
    role: str = "none",
    tail_room: int = 0,
) -> tuple[int, int, int, int]:
    """Return ``(sample_length, trim_front, trim_back, export_length)``.

    Director continuity contract: a segment is sampled as up to three zones and
    only the middle is exported. Reference frames (head/tail pins) are *frozen*
    conditioning rows (``img_update=False``) that are never denoised; the export
    is always the clean middle. A referenced segment therefore drops the ``+5``
    VAE phase and exports a clean ``17k`` instead of ``17k+5`` — the phase is
    carried by the reference frames, so nothing is clipped.

    Roles (see ``segment_continuity_role``):
      * ``none``  standalone. sample = export = 17k+5.
      * ``prev``  head pinned from the previous segment's tail (17n+5) at the
                  front. sample = 17(k+n)+5, trim_front = 17n+5, export = 17k.
      * ``next``  tail pinned from the next segment's opening (5 frames).
                  sample = 17k+5, trim_back = 5, export = 17k.  [Phase 2/3]
      * ``both``  head + tail pinned. sample = 17(k+n+1)+5,
                  trim_front = 17n+5, trim_back = 17, export = 17k. [Phase 2/3]

    ``next``/``both`` carry the +5 VAE phase on the frozen tail reference too,
    so their export is also 17k. The two-pass ``run_batch`` sampler guarantees
    the next segment's latent exists on disk before these sample (trim_back is
    realised by the export-length cap, not a separate decode pass).
    """
    from .frame_align import minimax_align_frame_count

    visible = minimax_align_frame_count(max(5, int(visible_frames)))
    k = (visible - 5) // 17
    if role == "prev":
        n = max(0, (int(context_frames) - 5) // 17) if context_frames else 0
        sample = 17 * (k + n) + 5
        return sample, 17 * n + 5, 0, visible - 5
    if role == "next":
        # No head pin; the tail pin (next opening, TAIL_CONTEXT_FRAMES) lives past
        # the export. sample = 17k+5, trim_back = TAIL_CONTEXT_FRAMES, export = 17k.
        return visible, 0, TAIL_CONTEXT_FRAMES, visible - 5
    if role == "both":
        # Head pin (prev tail) + tail pin (next opening, 5 + 12 = 17 frames).
        # sample = 17(k+n+1)+5, trim_front = 17n+5, trim_back = 17, export = 17k.
        n = max(0, (int(context_frames) - 5) // 17) if context_frames else 0
        sample = 17 * (k + n + 1) + 5
        return sample, 17 * n + 5, 17, visible - 5
    # none
    return visible, 0, 0, visible


def handoff_end_frame(*, trim_frames: int, export_frames: int) -> int:
    """Sample-timeline pixel index where the exported segment ends (exclusive)."""
    return max(0, int(trim_frames)) + max(0, int(export_frames))
