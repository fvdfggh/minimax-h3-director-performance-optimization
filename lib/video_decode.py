"""Decoding video into model-ready frame tensors.

The middle layer: :func:`decode_video_frames` reads a frame range out of a file,
:func:`load_video_resampled` fits that to a target long edge and frame count, and
:func:`load_reference_video_clip` is the reference-video entry point used by the
plan.

Everything here goes through :mod:`av_probe`, so path resolution and the PyAV
requirement are stated in exactly one place.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np
import torch

from .av_probe import (
    _AvVideoReader,
    _aspect_close,
    _aspect_ratio,
    _require_av,
    _resize_rgb,
    resolve_video_path,
)
from .image_prep import resolve_output_dimensions

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.video.io.decode")


def decode_video_frames(path, *, dtype=torch.float32) -> torch.Tensor | None:
    """Decode a whole video file to an RGB frame tensor.

    ``dtype`` picks the pixel domain: ``torch.float32`` yields [0,1] (the
    ComfyUI IMAGE convention, still the default for every existing caller),
    ``torch.uint8`` yields [0,255].

    The uint8 path is what long timelines should use for pure transport: the
    decoder already produces uint8, so asking for it skips both a 4x expansion
    and a per-element divide. Quantisation is not being traded away — the
    source is an 8-bit H.264 render, so [0,255] is the pixel-exact round trip
    and converting afterwards is lossless. Lift to float32 only around work
    that genuinely needs the headroom (the continuity seam pipeline).

    ``None`` when nothing could be decoded (empty / unreadable container).
    """
    if not path or not os.path.isfile(path):
        return None
    out = torch.uint8 if dtype is torch.uint8 else torch.float32
    container = _require_av().open(str(path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if out is torch.uint8:
            frames = [
                torch.from_numpy(frame.to_ndarray(format="rgb24"))
                for frame in container.decode(stream)
            ]
        else:
            frames = [
                torch.from_numpy(frame.to_ndarray(format="rgb24")).to(torch.float32).div(255.0)
                for frame in container.decode(stream)
            ]
    finally:
        container.close()
    if not frames:
        return None
    # Stack uint8 first and convert once: per-frame promotion would materialise
    # a second full-size float copy of the whole clip.
    stacked = torch.stack(frames, dim=0).contiguous()
    del frames
    if out is not torch.uint8 and stacked.dtype != out:
        stacked = stacked.to(out)
    return stacked


def load_video_resampled(
    path: str,
    frame_rate: float,
    frame_indices: Sequence[int],
    *,
    storage_width: int | None = None,
    storage_height: int | None = None,
    long_edge: int = 848,
) -> torch.Tensor:
    """Decode selected resampled frame indices from a video file (PyAV, no OpenCV)."""
    if not frame_indices:
        raise ValueError("No frames requested from video.")

    with _AvVideoReader(path) as reader:
        native_fps = float(reader.fps or 0.0)
        if native_fps <= 0:
            native_fps = float(frame_rate or 24.0)

        out_w, out_h, rotate_90_cw = _resolve_load_dimensions(
            reader.width,
            reader.height,
            storage_width=storage_width,
            storage_height=storage_height,
            long_edge=long_edge,
        )

        unique = sorted({int(i) for i in frame_indices})
        decoded: dict[int, np.ndarray] = {}
        fallback: np.ndarray | None = None

        for src_idx in unique:
            t_sec = max(0.0, src_idx / float(frame_rate or 24.0))
            native_frame = int(round(t_sec * native_fps))
            rgb = reader.read(native_frame)
            if rgb is None:
                log.warning("Failed to read frame %d (t=%.3fs) from %s", native_frame, t_sec, path)
                if fallback is not None:
                    decoded[src_idx] = fallback
                continue

            if rotate_90_cw:
                rgb = np.ascontiguousarray(np.rot90(rgb, k=-1))
            if (rgb.shape[1], rgb.shape[0]) != (out_w, out_h):
                rgb = _resize_rgb(rgb, out_w, out_h)
            rgb = rgb.astype(np.float32) / 255.0
            decoded[src_idx] = rgb
            fallback = rgb

    if not decoded:
        raise ValueError(f"No frames decoded from video: {path}")

    rows = []
    last = next(iter(decoded.values()))
    for idx in frame_indices:
        rows.append(decoded.get(int(idx), last))
        last = rows[-1]

    return torch.from_numpy(np.stack(rows, axis=0))


def _resolve_load_dimensions(
    source_w: int,
    source_h: int,
    *,
    storage_width: int | None,
    storage_height: int | None,
    long_edge: int,
) -> tuple[int, int, bool]:
    """Return (out_w, out_h, rotate_90_cw) for proportional long-edge loading."""
    if source_w <= 0 or source_h <= 0:
        out_w, out_h, _, _ = resolve_output_dimensions(
            source_w,
            source_h,
            mode="long_edge",
            long_edge=long_edge,
        )
        return out_w, out_h, False

    native_portrait = source_h > source_w
    native_landscape = source_w > source_h

    if storage_width and storage_height:
        sw, sh = int(storage_width), int(storage_height)
        storage_portrait = sh > sw
        storage_landscape = sw > sh
        native_ar = _aspect_ratio(source_w, source_h)
        storage_ar = _aspect_ratio(sw, sh)
        transposed_ar = _aspect_ratio(source_h, source_w)

        if _aspect_close(native_ar, storage_ar):
            return sw, sh, False

        # Portrait-native video must not be rotated into a landscape target (common when
        # browser metadata or node defaults supply 832脳480 for a vertical phone clip).
        if native_portrait and storage_landscape:
            log.info(
                "Video %dx%d portrait native vs storage %dx%d landscape; keeping orientation",
                source_w,
                source_h,
                sw,
                sh,
            )
            out_w, out_h, _, _ = resolve_output_dimensions(
                source_w,
                source_h,
                mode="long_edge",
                long_edge=long_edge,
            )
            return out_w, out_h, False

        # Landscape-native with portrait storage (rotation metadata in container).
        if native_landscape and storage_portrait and _aspect_close(transposed_ar, storage_ar):
            log.info(
                "Video %dx%d landscape native vs storage %dx%d portrait; applying 90掳 rotation",
                source_w,
                source_h,
                sw,
                sh,
            )
            return sw, sh, True

        if _aspect_close(transposed_ar, storage_ar):
            log.info(
                "Video %dx%d decoded transposed vs storage %dx%d; applying 90掳 rotation before scale",
                source_w,
                source_h,
                sw,
                sh,
            )
            return sw, sh, True

        log.warning(
            "storage %dx%d aspect mismatch vs native %dx%d; using proportional long_edge=%d",
            sw,
            sh,
            source_w,
            source_h,
            long_edge,
        )

    out_w, out_h, _, _ = resolve_output_dimensions(
        source_w,
        source_h,
        mode="long_edge",
        long_edge=long_edge,
    )
    return out_w, out_h, False


def load_reference_video_clip(
    ref_block: dict,
    timeline: dict,
    num_frames: int,
    *,
    start_frame: int = 0,
) -> torch.Tensor | None:
    """Load an ads2v reference video clip, resampled to ``num_frames`` at timeline FPS.

    ``start_frame`` is the logical timeline offset (0 = from beginning). Used when
    global *continuous reference* is enabled so segment N uses ref frame N onward.
    """
    if not (ref_block.get("videoFile") or ref_block.get("fileName") or "").strip():
        return None

    path = resolve_video_path(ref_block)
    frame_rate = float(timeline.get("frameRate") or 24)
    output_block = timeline.get("output") or {}
    long_edge = int(
        output_block.get("longEdge")
        or output_block.get("long_edge")
        or timeline.get("refMaxSize")
        or 848
    )
    count = max(1, int(num_frames))
    offset = max(0, int(start_frame))
    frame_indices = list(range(offset, offset + count))
    return load_video_resampled(
        path,
        frame_rate,
        frame_indices,
        storage_width=ref_block.get("storageWidth"),
        storage_height=ref_block.get("storageHeight"),
        long_edge=long_edge,
    )
