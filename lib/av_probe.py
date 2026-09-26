"""Probing and reading video files through PyAV (no OpenCV).

PyAV ships with this ComfyUI build, ``cv2`` usually does not: the portable launcher
runs ``python -s``, which hides the user-level site-packages where OpenCV lives.
Every video path in the plugin therefore goes through PyAV.

This is the bottom layer: resolve a file reference to an absolute path, ask
ffprobe / PyAV for stream metadata, count frames, peek at a frame's size, and read
frames out as float RGB arrays. Decoding into model-ready tensors is
:mod:`video_decode`; turning timeline frame maps into reads is
:mod:`timeline_frames`.
"""

from __future__ import annotations

import logging
import os

import numpy as np

import folder_paths

from .ffmpeg import ffprobe_bin

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.video.io.probe")


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

    probe = ffprobe_bin()
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

    probe = ffprobe_bin()
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


def _sane_fps(fps: float, frame_count: int, duration: float) -> tuple[float, str]:
    """Filter frame rates that containers report but that cannot be real.

    Two everyday lies: a variable-frame-rate file reports ``avg_frame_rate = 0/0``
    (so the code fell through to ``r_frame_rate``), and on some containers
    ``r_frame_rate`` is the **time base** — ``1000/1``. Handing either to the UI ends
    up clamped to the 240 ceiling, which looks like「帧率莫名其妙变成 240」.

    ``frame_count / duration`` is what actually plays, so it wins whenever the
    container's claim disagrees with it by more than 20 %. Returns
    ``(fps, source)`` with ``fps == 0.0`` when nothing trustworthy is available.
    """
    derived = (float(frame_count) / float(duration)) if (frame_count > 0 and duration > 0) else 0.0

    def plausible(value: float) -> bool:
        return 1.0 <= value <= 240.0

    if plausible(fps) and not (plausible(derived) and abs(fps - derived) > derived * 0.2):
        return float(fps), "container"
    if plausible(derived):
        log.info(
            "Video frame rate looks wrong (container %.3f vs %.3f from %d frames / %.3fs);"
            " using the derived rate.", fps, derived, frame_count, duration,
        )
        return float(derived), "duration"
    if plausible(fps):
        return float(fps), "container"
    log.warning(
        "Video frame rate unusable (container %.3f, derived %.3f); leaving it unset.",
        fps, derived,
    )
    return 0.0, "unusable"


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

    # 容器标称帧率可能不可信（VFR 的 0/0、r_frame_rate 实为时间基 1000/1）：
    # 交给 _sane_fps 用「帧数 / 时长」交叉验证，宁可报 0 让调用方兜底 24。
    fps_source = "default"
    if native_fps > 0 or (frame_count > 0 and duration > 0):
        native_fps, fps_source = _sane_fps(native_fps, frame_count, duration)
    if native_fps <= 0:
        native_fps = 24.0

    return {
        "width": width,
        "height": height,
        "duration": duration,
        "native_fps": native_fps,
        # 帧率取自容器 / 由帧数与时长推出 / 兜底 24 —— 排错时一眼看出是谁给的值。
        "fps_source": fps_source,
        "frame_count": frame_count,
        "probe_method": method,
    }


def probe_video_clip(video: dict) -> dict:
    """Probe a timeline clip dict (videoFile / subfolder / type)."""
    return probe_video_file(resolve_video_path(video))


def _aspect_ratio(w: int, h: int) -> float:
    return w / h if h > 0 else 0.0


def _aspect_close(a: float, b: float, *, tol: float = 0.04) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / max(a, b) <= tol
