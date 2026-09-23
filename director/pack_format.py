"""On-disk pack format: ids, prefixes, name patterns, caps and key lists.

This is the single source for the zip contract shared with upstream
``ComfyUI_MiniMaxH3_Director``: a pack exported by either plugin must import in the
other, so everything here is an **external interface** — change a value only
together with a migration.

Same on-disk layout as upstream:

    shared_params/   Asset group folders  PictureN / VideoN / AudioN
    asset_groups/
    source_video/
    extra/

Zip paths are ASCII and match the English UI.
"""

from __future__ import annotations

import re


PACK_FORMAT = "minimax-h3-director-pack"
#: Opt flavour. Exports keep the upstream id so the original plugin can open
#: them too; both ids are accepted on import.
PACK_FORMAT_OPT = "minimax-h3-director-opt-pack"
PACK_FORMATS = (PACK_FORMAT, PACK_FORMAT_OPT)
PACK_VERSION = 1
PACK_PREFIXES = ("shared_params/", "asset_groups/", "source_video/", "extra/")
ASCII_PATH_RE = re.compile(r"^[A-Za-z0-9_./]+$")
#: Opt removes the upstream slot caps: PictureN / VideoN / AudioN accept any
#: count. Upstream only reads 1-9 / 1-3 when it has to rebuild a timeline from
#: folder scanning, but it always prefers timeline.json — which Opt writes in
#: full — so extra slots still survive a round-trip through the original plugin.
PICTURE_FILE_RE = re.compile(r"^Picture(\d+)(\.[A-Za-z0-9]{1,8})$", re.I)
VIDEO_FILE_RE = re.compile(r"^Video(\d+)(\.[A-Za-z0-9]{1,8})$", re.I)
AUDIO_FILE_RE = re.compile(r"^Audio(\d+)(\.[A-Za-z0-9]{1,8})$", re.I)
START_FILE_RE = re.compile(r"^start(\.[A-Za-z0-9]{1,8})$", re.I)
END_FILE_RE = re.compile(r"^end(\.[A-Za-z0-9]{1,8})$", re.I)

MAX_UNCOMPRESSED = 16 * 1024 * 1024 * 1024
MAX_ZIP_ENTRIES = 8000
MAX_SINGLE_FILE = 8 * 1024 * 1024 * 1024
PACK_EXPORT_TTL_SEC = 60 * 60
ZIP_STREAM_CHUNK = 1024 * 1024
# POST JSON already works; keep small packs on that path instead of FileResponse.
INLINE_JSON_MAX = 48 * 1024 * 1024

IMAGE_KEYS = ("imageFile", "image_file")
AUDIO_KEYS = ("audioFile", "audio_file")
VIDEO_KEYS = ("videoFile", "video_file")
PREVIEW_KEYS = ("previewImageFile", "preview_image_file")
PAIRED_AUDIO_KEYS = ("pairedAudioFile", "paired_audio_file")
