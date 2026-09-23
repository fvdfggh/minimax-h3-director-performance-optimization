"""File-name, path and media-extension helpers shared across the plugin.

The media extension sets, the Windows-illegal-character rules and the "safe
suffix" pattern used to be declared separately in the HTTP routes, the pack code
and the cache layout — three copies that could drift apart. Uploads, packing and
the on-disk cache now agree on one definition.
"""

from __future__ import annotations

import os
import re

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".mts", ".ts"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS

#: ``.ext`` with 1-8 alphanumerics — rejects junk suffixes on upload/import.
SAFE_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")
#: Windows-illegal path characters, and Windows reserved device names.
WIN_ILLEGAL_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
WIN_RESERVED_RE = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)


def posix_relpath(path: str | os.PathLike[str], start: str | os.PathLike[str]) -> str:
    """``os.path.relpath`` rendered with forward slashes (the JSON/wire form).

    Raises ``ValueError`` for paths on different drives, as ``relpath`` does.
    """
    return os.path.relpath(str(path), str(start)).replace("\\", "/")
