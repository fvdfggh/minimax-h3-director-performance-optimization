"""Load source videos from ComfyUI input folder (VHS-style file references)."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

import folder_paths

from .image_prep import resolve_output_dimensions

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.video_io")


def _require_av():
    """PyAV is bundled with this ComfyUI build — no OpenCV dependency.

    The portable launcher runs ``python -s``, which hides the user-level
    site-packages where ``cv2`` usually lives, so every video I/O path here goes
    through PyAV instead.
    """
    try:
        import av

        return av
    except ImportError as exc:  # pragma: no cover - packaging guard
        raise ImportError(
            "PyAV is required for MiniMax H3 Director Opt video loading. "
            "Install: pip install av"
        ) from exc


def _directories_for_type(type_name: str) -> list[str]:
    """Absolute directories to search for a media ``type``, most specific first.

    Keeps to input/output/temp — the directories ComfyUI itself resolves for
    ``/api/view?type=…`` — so a timeline can never point outside them.
    """
    key = (type_name or "input").strip().lower() or "input"
    out: list[str] = []
    try:
        primary = folder_paths.get_directory_by_type(key)
    except Exception:
        primary = None
    if primary:
        out.append(primary)
    if key != "input":
        # Fall back to input, not the reverse: a record written before ``type``
        # was honoured still has to resolve, and input is where uploads land.
        try:
            inp = folder_paths.get_input_directory()
        except Exception:
            inp = None
        if inp and inp not in out:
            out.append(inp)
    return out


def resolve_video_path(video: dict) -> str:
    """Resolve timeline video metadata to an absolute file path.

    Honours ``video["type"]`` (``input`` / ``output`` / ``temp``), which every
    reference-video record already carries and this function used to discard —
    it pinned *all* candidates under ComfyUI's input directory, so any clip
    living elsewhere was reported missing even though the caller had told us
    exactly where it was. That kept Director's own renders unreachable: they are
    written under ``output/minimax_director_opt_cache/…``.

    Resolution order, mirroring how ComfyUI's own ``/api/view`` reads files:

      1. the directory matching ``type`` (default ``input``), trying
         ``subfolder + basename``, then ``videoFile`` as-is, then ``basename``;
      2. for non-input types, the same three shapes under ``input``.

    I/O backends add extra directories; ``get_directory_by_type`` already knows
    which ones exist, so this stays correct for those too.
    """
    video_file = (video.get("videoFile") or video.get("fileName") or "").strip()
    if not video_file:
        raise ValueError("No video file in MiniMax H3 Director Opt timeline.")

    subfolder = (video.get("subfolder") or "").strip().replace("\\", "/")
    # Basename only, with any traversal attempt stripped before it reaches join.
    clean_name = os.path.basename(str(video_file).replace("\\", "/"))
    if not clean_name or clean_name in (".", ".."):
        raise ValueError(f"Invalid video file reference: {video_file!r}")

    bases = _directories_for_type(video.get("type"))
    if not bases:
        bases = [folder_paths.get_input_directory()]

    for base in bases:
        root = os.path.abspath(base)
        candidates = []
        if subfolder:
            candidates.append(os.path.join(root, subfolder.replace("/", os.sep), clean_name))
        candidates.append(os.path.join(root, str(video_file).replace("/", os.sep)))
        candidates.append(os.path.join(root, clean_name))
        for path in candidates:
            resolved = os.path.abspath(path)
            # Never let a crafted subfolder/file escape the type root.
            if os.path.commonpath([resolved, root]) != root:
                continue
            if os.path.isfile(resolved):
                return resolved

    raise ValueError(
        f"Video file not found ({video.get('type') or 'input'}): {video_file}"
    )


def ffprobe_bin() -> str | None:
    """Resolve ffprobe: PATH first, then imageio-ffmpeg sibling binary if present."""
    import shutil

    probe = shutil.which("ffprobe")
    if probe:
        return probe
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        import os

        ff = get_ffmpeg_exe()
        stem = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        candidate = os.path.join(os.path.dirname(ff), stem)
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass
    return None


# Back-compat alias used by older call sites / hot-reload.
_ffprobe_bin = ffprobe_bin


def _parse_rate(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f > 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _ffprobe_stream_info(path: str) -> dict | None:
    import json
    import subprocess

    probe = _ffprobe_bin()
    if not probe:
        return None
    try:
        res = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration,nb_frames,r_frame_rate,avg_frame_rate",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            check=True,
        )
        payload = json.loads(res.stdout.decode("utf-8", "replace"))
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as exc:
        log.debug("ffprobe stream info failed for %s: %s", path, exc)
        return None
    streams = payload.get("streams") or []
    return streams[0] if streams else None


def peek_video_size(path: str) -> tuple[int, int]:
    """Width/height from container metadata. Does not decode frames or count them."""
    if not path or not os.path.isfile(path):
        return 0, 0
    try:
        stream = _ffprobe_stream_info(path)
        if stream:
            w = int(stream.get("width") or 0)
            h = int(stream.get("height") or 0)
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass
    try:
        meta = av_video_meta(path)
        return int(meta.get("width") or 0), int(meta.get("height") or 0)
    except Exception:
        return 0, 0


def _ffprobe_count_frames(path: str) -> int | None:
    import subprocess

    probe = _ffprobe_bin()
    if not probe:
        return None
    try:
        res = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            check=True,
        )
        text = res.stdout.decode("utf-8", "replace").strip()
        if text.isdigit():
            return int(text)
    except (subprocess.CalledProcessError, OSError) as exc:
        log.debug("ffprobe count_frames failed for %s: %s", path, exc)
    return None


def _av_stream_fps(stream) -> float:
    rate = getattr(stream, "average_rate", None) or getattr(stream, "guessed_rate", None)
    try:
        return float(rate) if rate else 0.0
    except (TypeError, ValueError):
        return 0.0


class _AvVideoReader:
    """Frame-accurate reader indexed by decode order (``cv2.CAP_PROP_POS_FRAMES`` semantics).

    Callers ask for frames in ascending order, so the common case is one forward
    pass with a single decode iterator; a backward jump re-seeks the container.
    Invariant: ``_last`` holds the frame at index ``_next_index - 1``.
    """

    def __init__(self, path: str):
        _require_av()
        self.path = path
        self._open()

    def _open(self) -> None:
        import av

        self.container = av.open(self.path)
        try:
            stream = self.container.streams.video[0]
        except (IndexError, AttributeError) as exc:
            self.container.close()
            raise ValueError(f"No video stream in: {self.path}") from exc
        stream.thread_type = "AUTO"
        self.stream = stream
        self.fps = _av_stream_fps(stream)
        self.width = int(stream.codec_context.width or 0)
        self.height = int(stream.codec_context.height or 0)
        self._iter = self.container.decode(stream)
        self._next_index = 0
        self._last: np.ndarray | None = None
        self._eof = False

    def close(self) -> None:
        try:
            self.container.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def __enter__(self) -> _AvVideoReader:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _pts_index(self, frame) -> int | None:
        pts = getattr(frame, "pts", None)
        tb = self.stream.time_base
        if pts is None or not tb or self.fps <= 0:
            return None
        return int(round(float(pts) * float(tb) * self.fps))

    def seek(self, index: int) -> None:
        """Position the iterator so the next produced frame is ``index``."""
        tb = float(self.stream.time_base)
        offset = int(round(index / self.fps / tb)) if (self.fps > 0 and tb > 0) else int(index)
        try:
            self.container.seek(max(0, offset), stream=self.stream, backward=True, any_frame=False)
        except Exception as exc:
            log.debug("PyAV seek to %d failed for %s (%s); rewinding", index, self.path, exc)
            try:
                self.container.seek(0, stream=self.stream, backward=True, any_frame=True)
            except Exception:  # pragma: no cover - defensive
                pass
        self._iter = self.container.decode(self.stream)
        self._last = None
        self._eof = False
        frame = next(self._iter, None)
        if frame is None:
            self._eof = True
            self._next_index = index
            return
        k = self._pts_index(frame)
        if k is None:
            log.debug("PyAV: stream has no PTS for %s; frame indices may drift.", self.path)
            k = 0
        while k < index:
            frame = next(self._iter, None)
            if frame is None:
                self._eof = True
                self._next_index = index
                return
            k += 1
        self._last = frame.to_ndarray(format="rgb24")
        self._next_index = index + 1

    def read(self, index: int) -> np.ndarray | None:
        """uint8 RGB frame ``index``; repeats the last decoded frame past EOF."""
        index = max(0, int(index))
        if index < self._next_index - 1:
            self.seek(index)
        while self._next_index <= index:
            frame = next(self._iter, None)
            if frame is None:
                self._eof = True
                break
            self._last = frame.to_ndarray(format="rgb24")
            self._next_index += 1
        return self._last


def _resize_rgb(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize ``HWC`` uint8 RGB without OpenCV (BOX ≈ ``cv2.INTER_AREA``)."""
    from PIL import Image

    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    shrink = out_w < src_w or out_h < src_h
    resample = Image.Resampling.BOX if shrink else Image.Resampling.LANCZOS
    img = Image.fromarray(np.ascontiguousarray(arr))
    return np.asarray(img.resize((int(out_w), int(out_h)), resample=resample))


def av_video_meta(path: str) -> dict:
    """Container metadata via PyAV. Raises when the file has no decodable video."""
    if not path or not os.path.isfile(path):
        raise ValueError(f"Video file not found: {path}")
    with _AvVideoReader(path) as reader:
        stream = reader.stream
        fps = reader.fps
        frame_count = int(getattr(stream, "frames", 0) or 0)
        duration = 0.0
        if getattr(stream, "duration", None):
            duration = float(stream.duration) * float(stream.time_base)
        if duration <= 0 and getattr(reader.container, "duration", None):
            duration = float(reader.container.duration) / 1_000_000.0
        if frame_count <= 0 and duration > 0 and fps > 0:
            frame_count = int(round(duration * fps))
        if fps <= 0 and duration > 0 and frame_count > 0:
            fps = frame_count / duration
        return {
            "width": reader.width,
            "height": reader.height,
            "duration": duration,
            "native_fps": fps,
            "frame_count": max(0, frame_count),
        }


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


def probe_video_file(path: str) -> dict:
    """Probe container metadata and an accurate frame count for Director UI."""
    if not path or not os.path.isfile(path):
        raise ValueError(f"Video file not found: {path}")

    method = "estimated"
    stream = _ffprobe_stream_info(path)
    container_meta = None

    width = height = 0
    duration = 0.0
    native_fps = 0.0
    frame_count = 0

    if stream:
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        duration = float(stream.get("duration") or 0.0)
        native_fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
        nb_frames = str(stream.get("nb_frames") or "").strip()
        if nb_frames.isdigit() and int(nb_frames) > 0:
            frame_count = int(nb_frames)
            method = "ffprobe_nb_frames"

    if frame_count <= 0:
        counted = _ffprobe_count_frames(path)
        if counted is not None and counted > 0:
            frame_count = counted
            method = "ffprobe_count_frames"

    if frame_count <= 0 or width <= 0 or height <= 0 or native_fps <= 0:
        try:
            container_meta = av_video_meta(path)
        except Exception as exc:
            log.debug("PyAV probe failed for %s: %s", path, exc)
            container_meta = None
        if container_meta:
            width = width or int(container_meta["width"])
            height = height or int(container_meta["height"])
            native_fps = native_fps or float(container_meta["native_fps"])
            if frame_count <= 0 and int(container_meta["frame_count"]) > 0:
                frame_count = int(container_meta["frame_count"])
                method = "pyav"

    if duration <= 0 and frame_count > 0 and native_fps > 0:
        duration = frame_count / native_fps
    elif duration <= 0 and container_meta:
        duration = float(container_meta.get("duration") or 0.0)

    if frame_count <= 0 and duration > 0 and native_fps > 0:
        frame_count = max(1, int(round(duration * native_fps)))
        if method == "estimated":
            method = "duration_estimate"

    if frame_count <= 0:
        raise ValueError(f"Could not determine frame count for video: {path}")

    if native_fps <= 0:
        native_fps = 24.0

    return {
        "width": width,
        "height": height,
        "duration": duration,
        "native_fps": native_fps,
        "frame_count": frame_count,
        "probe_method": method,
    }


def probe_video_clip(video: dict) -> dict:
    """Probe a timeline clip dict (videoFile / subfolder / type)."""
    return probe_video_file(resolve_video_path(video))


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


def _aspect_ratio(w: int, h: int) -> float:
    return w / h if h > 0 else 0.0


def _aspect_close(a: float, b: float, *, tol: float = 0.04) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / max(a, b) <= tol


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


def parse_frame_map_entry(entry: Any, default_clip: int = 0) -> tuple[int, int]:
    """Parse a frameMap entry to (clip_index, source_frame_index)."""
    if isinstance(entry, dict):
        clip = int(entry.get("clip", entry.get("videoClip", default_clip)))
        frame = int(entry.get("frame", 0))
        return clip, frame
    return default_clip, int(entry)


def video_clips_from_timeline(timeline: dict) -> list[dict]:
    """Return ordered video clip metadata; falls back to legacy single ``video`` block."""
    clips = timeline.get("videoClips") or timeline.get("video_clips")
    if clips:
        return list(clips)
    video = timeline.get("video") or {}
    if (video.get("videoFile") or video.get("fileName") or "").strip():
        return [video]
    return []


def deleted_source_ranges(timeline: dict) -> list[tuple[int, int]]:
    """Source-frame spans removed from the logical timeline (sparse single-clip edits)."""
    video = timeline.get("video") or {}
    raw = video.get("deletedSourceRanges") or video.get("deleted_source_ranges") or []
    ranges: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = int(item[0]), int(item[1])
            if end > start:
                ranges.append((start, end))
    return sorted(ranges)


def logical_frame_map(timeline: dict) -> list[Any]:
    """Explicit per-logical-frame map only; empty means sparse identity mapping."""
    video = timeline.get("video") or {}
    frame_map = video.get("frameMap")
    if frame_map:
        return list(frame_map)
    return []


def logical_frame_count(timeline: dict) -> int:
    frame_map = logical_frame_map(timeline)
    if frame_map:
        return len(frame_map)

    total = int(timeline.get("totalFrames") or 0)
    if total > 0:
        return total

    video = timeline.get("video") or {}
    source_count = int(video.get("sourceFrameCount") or 0)
    if source_count > 0:
        removed = sum(end - start for start, end in deleted_source_ranges(timeline))
        return max(0, source_count - removed)

    clips = video_clips_from_timeline(timeline)
    if len(clips) > 1:
        return sum(int(c.get("sourceFrameCount") or 0) for c in clips)

    return 0


def resolve_logical_frame_entry(timeline: dict, logical_index: int) -> tuple[int, int]:
    """Map a logical timeline index to (clip_index, source_frame_index)."""
    video = timeline.get("video") or {}
    frame_map = video.get("frameMap") or []
    if logical_index < len(frame_map):
        return parse_frame_map_entry(frame_map[logical_index])

    src = logical_index
    for start, end in deleted_source_ranges(timeline):
        if src >= start:
            src += end - start
        else:
            break

    clips = video_clips_from_timeline(timeline)
    if len(clips) <= 1:
        return 0, src

    offset = 0
    for clip_idx, clip in enumerate(clips):
        count = int(clip.get("sourceFrameCount") or 0)
        if logical_index < offset + count:
            return clip_idx, logical_index - offset
        offset += count

    last = clips[-1]
    last_count = max(1, int(last.get("sourceFrameCount") or 1))
    return len(clips) - 1, last_count - 1


def frame_indices_from_timeline(timeline: dict) -> list[int]:
    """Legacy helper: source-frame indices for single-clip timelines."""
    total = logical_frame_count(timeline)
    entries = [resolve_logical_frame_entry(timeline, i) for i in range(total)]
    if entries and all(c == 0 for c, _ in entries):
        return [f for _, f in entries]
    return list(range(total))


def _decode_timeline_entries(
    timeline: dict,
    entries: list[tuple[int, int]],
    *,
    frame_rate: float,
    default_long_edge: int,
) -> torch.Tensor:
    clips = video_clips_from_timeline(timeline)
    if not clips:
        raise ValueError("No video clips in MiniMax H3 Director Opt timeline.")

    by_clip: dict[int, set[int]] = defaultdict(set)
    for clip_idx, frame_idx in entries:
        if clip_idx < 0 or clip_idx >= len(clips):
            clip_idx = 0
        by_clip[clip_idx].add(frame_idx)

    frame_tensors: dict[tuple[int, int], torch.Tensor] = {}
    for clip_idx, frame_set in sorted(by_clip.items()):
        clip = clips[clip_idx]
        path = resolve_video_path(clip)
        sorted_idx = sorted(frame_set)
        tensor = load_video_resampled(
            path,
            frame_rate,
            sorted_idx,
            storage_width=clip.get("storageWidth"),
            storage_height=clip.get("storageHeight"),
            long_edge=int(clip.get("longEdge") or default_long_edge),
        )
        for row, fi in enumerate(sorted_idx):
            frame_tensors[(clip_idx, fi)] = tensor[row]

    rows: list[torch.Tensor] = []
    fallback: torch.Tensor | None = None
    for clip_idx, frame_idx in entries:
        if clip_idx < 0 or clip_idx >= len(clips):
            clip_idx = 0
        key = (clip_idx, frame_idx)
        tensor = frame_tensors.get(key, fallback)
        if tensor is None:
            raise ValueError(f"Missing decoded frame for clip {clip_idx} frame {frame_idx}")
        fallback = tensor
        rows.append(tensor)

    return torch.stack(rows, dim=0)


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


def load_timeline_segment(timeline: dict, start: int, end: int) -> torch.Tensor:
    """Decode only logical frames in [start, end) 鈥?supports arbitrarily long timelines."""
    total = logical_frame_count(timeline)
    start = max(0, min(int(start), total))
    end = max(start, min(int(end), total))
    if start >= end:
        raise ValueError(f"No frames in timeline range [{start}, {end})")

    video = timeline.get("video") or {}
    frames_b64 = video.get("frames") or []
    if frames_b64:
        chunks: list[torch.Tensor] = []
        for frame_b64 in frames_b64[start:end]:
            chunks.append(_decode_image_b64_inline(frame_b64))
        if not chunks:
            raise ValueError("Uploaded video has no decodable frames in range.")
        return torch.cat(chunks, dim=0)

    frame_rate = float(timeline.get("frameRate") or 24)
    output_block = timeline.get("output") or {}
    default_long_edge = int(
        output_block.get("longEdge")
        or output_block.get("long_edge")
        or timeline.get("refMaxSize")
        or 848
    )

    entries = [resolve_logical_frame_entry(timeline, i) for i in range(start, end)]
    return _decode_timeline_entries(
        timeline,
        entries,
        frame_rate=frame_rate,
        default_long_edge=default_long_edge,
    )


def _decode_image_b64_inline(b64_str: str) -> torch.Tensor:
    import base64
    import io

    from PIL import Image

    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def load_multi_clip_timeline(
    timeline: dict,
    frame_map: list[Any],
    *,
    frame_rate: float,
    default_long_edge: int,
) -> torch.Tensor:
    """Decode a logical timeline that may reference multiple source videos."""
    entries = [parse_frame_map_entry(e) for e in frame_map]
    return _decode_timeline_entries(
        timeline,
        entries,
        frame_rate=frame_rate,
        default_long_edge=default_long_edge,
    )
