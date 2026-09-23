"""Locate the ffmpeg / ffprobe binaries used by every media helper.

Single source of truth: the resolution rules (imageio-ffmpeg's bundled build vs.
PATH) used to be duplicated in ``video_export`` and ``audio_io`` and had already
drifted apart, so binary lookup lives here instead.
"""

from __future__ import annotations

import os
import shutil


def ffmpeg_bin() -> str | None:
    """Bundled imageio-ffmpeg binary first, then ``ffmpeg`` on PATH."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def ffprobe_bin() -> str | None:
    """Resolve ffprobe: PATH first, then imageio-ffmpeg sibling binary if present."""
    probe = shutil.which("ffprobe")
    if probe:
        return probe
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ff = get_ffmpeg_exe()
        stem = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        candidate = os.path.join(os.path.dirname(ff), stem)
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass
    return None
