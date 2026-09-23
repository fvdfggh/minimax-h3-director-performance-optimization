"""Per-segment cache codecs: pixel frames, head/tail windows and audio payloads.

Extracted from :mod:`segment_cache`, which had grown past 3000 lines. Everything
here only touches tensors, ``cache_layout`` paths and the atomic-write helpers,
so it can be read and changed without the slot-map / export machinery around it.

The uint8 ↔ float32 boundary is the subtle part: the cache stores 8-bit pixels to
keep the segment files small, while the seam pipeline blends in float [0,1]. Use
:func:`_frames_as` for every cross-dtype assignment — a plain ``.to(dtype)`` is
wrong in one direction and destroys the image in the other.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import torch

from ..lib.fs import (
    atomic_publish as _atomic_publish,
    safe_unlink as _safe_unlink,
    write_via_temp as _write_via_temp,
)
from . import cache_layout

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache.codecs")


def _audio_payload_to_cpu(audio: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize export AUDIO dict for disk cache (waveform on CPU)."""
    if not isinstance(audio, dict):
        return None
    wave = audio.get("waveform")
    if not isinstance(wave, torch.Tensor) or wave.numel() <= 0:
        return None
    sr = int(audio.get("sample_rate") or 0) or 32000
    return {
        "waveform": wave.detach().cpu().contiguous(),
        "sample_rate": sr,
    }


def _frames_to_disk(tensor: torch.Tensor) -> torch.Tensor:
    """Store pixel frames as uint8 [0,255]. Export is 8-bit anyway; float32 is 4× larger.

    Returns a tensor that **owns** its storage. ``contiguous()`` hands a
    contiguous slice back unchanged, and a trimmed export *is* a slice of the
    full VAE decode — saving that view persists every frame of the decode.
    """
    x = tensor.detach().cpu()
    if x.dtype == torch.uint8:
        return x.clone()
    return x.float().clamp(0, 1).mul(255).round().clamp(0, 255).to(torch.uint8)


def _frames_from_disk(loaded: Any) -> torch.Tensor | None:
    """Restore uint8 cache to float32 [0,1]; pass through legacy float caches."""
    if not isinstance(loaded, torch.Tensor):
        return None
    if loaded.dtype == torch.uint8:
        return loaded.float().div(255.0)
    return loaded.float()


# Head/tail retained per segment. Mirrors segment_continuity._seam_window() so the
# seam pipeline has real pixels on each side of a join without persisting the
# *whole* segment tensor (which is what made the cache balloon to GBs).
HEADTAIL_N = cache_layout.FRAMES_HT_N

#: ``frames_ht`` payload marker. Older caches stored a bare
#: ``[head_N, tail_N, H, W, C]`` tensor with no record of how long the render it
#: was cut from is, so a reader could not tell a current clip from a stale one
#: and spliced the two into a single "segment" made of two different renders.
HEADTAIL_VERSION = 2


def _frames_to_float(x: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] / legacy float cache → float32 [0,1]."""
    return x.float().div(255.0) if x.dtype == torch.uint8 else x.float()


def _frames_as(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert frames to ``dtype`` across the uint8 ↔ float32 boundary.

    The single entry point every cross-dtype assignment must route through. The
    two domains use *different ranges* for the same pixels — uint8 is [0,255],
    float32 is [0,1] — so a plain ``.to(dtype)`` is wrong in one direction and
    catastrophic in the other: casting float32 [0,1] straight to uint8
    truncates every value to 0 or 1, turning a seam window nearly black. Every
    existing caller used ``.to(target.dtype)`` and was therefore only correct
    while everything happened to be float32.

    Keeps ``torch.float32`` semantics for the seam pipeline, whose weighted
    blends need the headroom, and uint8 semantics for storage/transport.
    """
    x = x.detach()
    if dtype == torch.uint8:
        if x.dtype == torch.uint8:
            return x
        return x.float().clamp(0, 1).mul(255).round().clamp(0, 255).to(torch.uint8)
    if x.dtype == torch.uint8:
        return x.float().div(255.0)
    return x.to(dtype)


def _frames_to_headtail(tensor: torch.Tensor) -> dict[str, Any]:
    """Head/tail window of a render, as a self-describing payload.

    A segment's video on disk is ``clip.mp4``; this window keeps the first/last
    ``HEADTAIL_N`` frames the seam pipeline needs, as uint8 [0,255].
    ``total`` records the render's frame count so a reader can verify the window
    and the clip came from the same take before stitching them together.

    ``tail`` is empty when the whole segment already fits inside the window
    (``total <= 2 * HEADTAIL_N``) — the payload then holds every frame.
    """
    x = tensor.detach().cpu()
    if x.dtype != torch.uint8:
        x = x.float().clamp(0, 1).mul(255).round().clamp(0, 255).to(torch.uint8)
    x = x.contiguous()
    total = int(x.shape[0])
    if total <= 2 * HEADTAIL_N:
        # Cloned for the same reason as below: ``x`` (and an empty ``x[:0]``)
        # still reference the caller's storage, which for a trimmed export is
        # the whole VAE decode — saving the view would persist all of it.
        return {
            "version": HEADTAIL_VERSION,
            "total": total,
            "head": x.clone(),
            "tail": x[:0].clone(),
        }
    # ``.clone()``, never ``.contiguous()``: a contiguous slice is returned as-is
    # (no copy), so it keeps pointing at the *whole* segment's storage and
    # ``torch.save`` then persists every frame instead of just the window.
    return {
        "version": HEADTAIL_VERSION,
        "total": total,
        "head": x[:HEADTAIL_N].clone(),
        "tail": x[total - HEADTAIL_N:].clone(),
    }




def _load_headtail(ht_path: Path) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """``(head, tail, total)`` from a legacy ``frames_ht.pt``; older layouts too.

    Only reached when no ``frames_ht.mp4`` was written (see
    :func:`_load_headtail_paths`) — i.e. caches from before the window became a
    clip, or runs where the encoder was unavailable and the writer fell back to
    the tensor.

    ``tail`` ``None`` means the payload predates :data:`HEADTAIL_VERSION` (a bare
    ``[head_N, tail_N, ...]`` tensor) and the caller must split it by shape.
    ``total <= 0`` means the render's frame count is unknown.
    """
    try:
        payload = torch.load(ht_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        log.debug("Head/tail load failed for %s: %s", ht_path.name, exc)
        return None, None, 0
    if isinstance(payload, dict):
        head = payload.get("head")
        tail = payload.get("tail")
        if not torch.is_tensor(head) or head.ndim != 4:
            return None, None, 0
        if not torch.is_tensor(tail) or tail.ndim != 4:
            tail = None
        return head, tail, int(payload.get("total") or 0)
    if torch.is_tensor(payload) and payload.ndim == 4:
        return payload, None, 0
    return None, None, 0


# --- video-backed head/tail -------------------------------------------------
#
# The window is stored as ``frames_ht.mp4`` + a JSON sidecar instead of
# ``frames_ht.pt``. Same pixels the seam pipeline always consumed, but the
# tensor form is uncompressed uint8 — 32 frames at 720p is ~88 MB, which made
# the seam window the single largest file of a segment, larger than the whole
# rendered clip it annotates. Readers are unchanged: both forms come back as
# ``(head, tail, total)`` float/uint8 tensors and feed :func:`_splice_headtail`.


def _save_headtail(paths: dict[str, Path], tensor: torch.Tensor, *, fps: float) -> bool:
    """Persist the seam window as ``frames_ht.mp4`` + sidecar. Never raises.

    ``False`` means nothing was written and the caller must fall back to the
    lossless tensor form — a missing ffmpeg or an unsupported pix_fmt must not
    silently leave a segment with no seam window at all.

    The window is written as one clip: head lanes followed by tail lanes. They
    are not temporally adjacent, which costs a little compression, but keeps the
    file to a single decode on read.
    """
    payload = _frames_to_headtail(tensor)
    head = payload.get("head")
    tail = payload.get("tail")
    if not torch.is_tensor(head) or int(head.shape[0]) <= 0:
        return False
    total = int(payload.get("total") or 0)
    n_tail = int(tail.shape[0]) if torch.is_tensor(tail) else 0
    # uint8 → float [0,1]; the encoder converts straight back, so the round trip
    # is one quantisation, not two.
    window = head if n_tail <= 0 else torch.cat([head, tail], dim=0)
    mp4_path = paths["frames_ht_mp4"]
    meta_path = paths["frames_ht_meta"]
    try:
        from ..lib.video_export import write_frames_to_mp4

        tmp = mp4_path.with_name(f".{mp4_path.name}.{uuid.uuid4().hex}.tmp.mp4")
        try:
            write_frames_to_mp4(
                tmp,
                window.float().div(255.0),
                fps=float(fps or 24.0) or 24.0,
                crf=cache_layout.HEADTAIL_CRF,
                pix_fmt=cache_layout.HEADTAIL_PIX_FMT,
                preset=cache_layout.HEADTAIL_PRESET,
            )
            _atomic_publish(tmp, mp4_path)
        finally:
            _safe_unlink(tmp)
        # Written *after* the clip: a sidecar without its video would make the
        # reader report a window that is not there.
        side = {
            "version": HEADTAIL_VERSION,
            "total": total,
            "head_n": int(head.shape[0]),
            "tail_n": n_tail,
            "height": int(head.shape[1]),
            "width": int(head.shape[2]),
        }
        _write_via_temp(
            meta_path,
            lambda p: p.write_text(json.dumps(side, sort_keys=True), encoding="utf-8"),
        )
        # A .pt from an older run is now redundant *and* would shadow nothing,
        # but leaving it would keep tens of MB of dead cache on disk.
        _safe_unlink(paths["frames_ht"])
        return True
    except Exception as exc:
        log.warning(
            "Head/tail clip encode failed (%s); keeping the lossless tensor cache.",
            exc,
        )
        _safe_unlink(mp4_path)
        _safe_unlink(meta_path)
        return False


def _load_headtail_mp4(
    mp4_path: Path,
    meta_path: Path,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """``(head, tail, total)`` from ``frames_ht.mp4``; ``(None, None, 0)`` if unusable.

    Two things are checked before the window is trusted, because a wrong window
    is worse than no window — :func:`_splice_headtail` would otherwise blend
    foreign pixels into the seam:

    * the decoded frame count must equal ``head_n + tail_n``, so a truncated or
      re-encoded file is dropped rather than split at the wrong offset;
    * the sidecar's H/W crops the decode back to the render's real size, undoing
      the even-dimension padding the encoder applied.
    """
    if not mp4_path.is_file() or not meta_path.is_file():
        return None, None, 0
    try:
        side = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.debug("Head/tail sidecar unreadable for %s: %s", mp4_path.name, exc)
        return None, None, 0
    if not isinstance(side, dict):
        return None, None, 0
    head_n = int(side.get("head_n") or 0)
    tail_n = int(side.get("tail_n") or 0)
    if head_n <= 0:
        return None, None, 0
    frames = _decode_clip_frames(mp4_path)
    if frames is None or int(frames.shape[0]) != head_n + tail_n:
        log.debug(
            "Head/tail clip %s holds %s frames, expected %d; ignored.",
            mp4_path.name,
            "no" if frames is None else int(frames.shape[0]),
            head_n + tail_n,
        )
        return None, None, 0
    height = int(side.get("height") or 0)
    width = int(side.get("width") or 0)
    if height > 0 and width > 0:
        frames = frames[:, :height, :width, :]
    head = frames[:head_n].contiguous()
    tail = frames[head_n:].contiguous() if tail_n > 0 else frames[:0].contiguous()
    return head, tail, int(side.get("total") or 0)


def _has_headtail(paths: dict[str, Path]) -> bool:
    """Whether any seam-window form is on disk for this segment."""
    return (
        paths["frames_ht_mp4"].is_file() and paths["frames_ht_meta"].is_file()
    ) or paths["frames_ht"].is_file()


def _load_headtail_paths(
    paths: dict[str, Path],
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Seam window from whichever form is on disk — video first, tensor legacy."""
    head, tail, total = _load_headtail_mp4(paths["frames_ht_mp4"], paths["frames_ht_meta"])
    if head is not None:
        return head, tail, total
    if paths["frames_ht"].is_file():
        return _load_headtail(paths["frames_ht"])
    return None, None, 0


def _splice_headtail(
    full: torch.Tensor,
    head: torch.Tensor,
    tail: torch.Tensor | None,
    total: int,
) -> torch.Tensor:
    """Overlay the head/tail window onto the decoded ``clip.mp4`` body.

    The clip is the segment's authoritative video; ``frames_ht`` only carries the
    first/last frames the seam pipeline works on, so the join keeps the pixels
    the seam was built from instead of the encoder's version of them. Both are
    written by one :func:`save_segment_cache` call, so stitching them is only
    meaningful while their frame counts agree. When they do not, the pair
    belongs to two different renders and the clip is returned untouched rather
    than being turned into a hybrid of both.
    """
    n_full = int(full.shape[0])
    if n_full <= 0 or head is None or int(head.shape[0]) <= 0:
        return full
    if tuple(int(d) for d in head.shape[1:]) != tuple(int(d) for d in full.shape[1:]):
        # The mp4 is padded to even H/W (``_pad_even_hw``), so an odd-sized
        # segment decodes one row/column wider than its cached window.
        log.debug(
            "Segment head/tail window %s does not match clip %s; using the clip as-is.",
            tuple(head.shape[1:]), tuple(full.shape[1:]),
        )
        return full

    src = int(total or 0)
    if src <= 0:
        # Legacy payload: place the window by shape alone, exactly as the old
        # reader did, so pre-``ht_total`` caches keep behaving as they did.
        src = n_full
        n_ht = int(head.shape[0])
        hn = min(HEADTAIL_N, n_ht // 2, n_full)
        tail = head[n_ht - hn:] if hn > 0 else head[:0]
        head = head[:hn]
    if src != n_full:
        log.warning(
            "Segment head/tail window is %df but clip.mp4 holds %df — the two come "
            "from different renders, so the clip is used unmodified.",
            src, n_full,
        )
        return full
    if tail is None or int(tail.shape[0]) <= 0:
        # Short segment: the window already holds every frame.
        if int(head.shape[0]) < n_full:
            return full
        return _frames_as(_frames_to_float(head[:n_full]), full.dtype)
    hn = min(int(head.shape[0]), int(tail.shape[0]), n_full)
    if hn <= 0:
        return full
    # Route through :func:`_frames_as`: these tensors straddle the uint8/float
    # range boundary, and a bare ``.to(full.dtype)`` turned a float [0,1] window
    # into 0/1 pixels the moment ``full`` became uint8.
    full[:hn] = _frames_as(_frames_to_float(head[:hn]), full.dtype)
    full[n_full - hn:] = _frames_as(_frames_to_float(tail[int(tail.shape[0]) - hn:]), full.dtype)
    return full


def _decode_clip_frames(
    clip_path: Path, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor | None:
    """Decode a clip to RGB frames with PyAV (no OpenCV).

    ``dtype`` follows :func:`decode_video_frames`: float32 [0,1] by default,
    uint8 [0,255] for callers that only move pixels around.
    The portable build launches ComfyUI with ``python -s``, which hides the
    user-level site-packages where ``cv2`` would live, so all video I/O goes
    through the bundled PyAV. ``None`` on any failure.
    """
    from ..lib.video_io import decode_video_frames

    return decode_video_frames(clip_path, dtype=dtype)


def _load_full_segment_via_clip(
    clip_path: Path, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor | None:
    """Decode a rendered ``clip.mp4`` back into a full frame tensor.

    Used when the full ``frames.pt`` was dropped to save space; the seam
    pipeline only ever needs the head/tail (see :func:`load_segment_tail`), but
    export/merge still needs the whole segment and ``clip.mp4`` is already on
    disk. Falls back to ``None`` on any decode failure (caller keeps the run
    alive) — including an unreadable container or a missing decoder backend.

    ``dtype`` selects the pixel domain. The source is an 8-bit render, so
    ``torch.uint8`` is the pixel-exact, 4x smaller form; use it for transport
    that performs no arithmetic, and lift to float32 (the default) for anything
    that blends.
    """
    if not clip_path or not Path(clip_path).is_file():
        return None
    try:
        return _decode_clip_frames(Path(clip_path), dtype=dtype)
    except Exception as exc:
        log.warning("Segment clip decode failed for %s: %s", clip_path.name, exc)
        return None
