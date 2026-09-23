"""Rewriting a Director timeline's media references while packing.

A timeline's media fields may hold an inline base64 payload, an ``input``-relative
path or an absolute path. Packing has to turn all three into one stable relative
path inside the zip, and import has to turn them back — that translation is the
``_rewrite_*`` family here, one helper per card shape (image / audio / video list,
single image ref, video media), plus the group/card walkers that drive them.

Only the reference *strings* change: prompts, timings and task options are copied
through untouched, so a pack round-trip never alters what a segment asks for.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Any

from ..lib.pathutil import AUDIO_EXTS, IMAGE_EXTS, VIDEO_EXTS
from ..lib.task_prompts import (
    TASK_PROMPT_BY_KEY,
    resolve_task_key,
    task_type_option_label,
)
from .pack_format import (
    AUDIO_KEYS,
    IMAGE_KEYS,
    PACK_PREFIXES,
    PAIRED_AUDIO_KEYS,
    PREVIEW_KEYS,
    VIDEO_KEYS,
)
from .pack_io import _posix, _safe_ext, resolve_media_path


def _item_rel(item: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        val = item.get(key)
        if val:
            return str(val).replace("\\", "/").strip()
    return ""


def _copy_file(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    shutil.copy2(src, dest)
    return dest


def _set_media_fields(item: dict, pack_rel: str, primary_key: str) -> None:
    name = Path(pack_rel).name
    parent = _posix(Path(pack_rel).parent)
    item[primary_key] = pack_rel
    if primary_key == "imageFile":
        item.pop("image_file", None)
        item["fileName"] = name
    elif primary_key == "audioFile":
        item.pop("audio_file", None)
        item["fileName"] = name
    elif primary_key == "videoFile":
        item.pop("video_file", None)
        item["fileName"] = name
    item["type"] = "input"
    item["subfolder"] = "" if parent in (".", "") else parent
    item["imageB64"] = ""
    item.pop("previewB64", None)


def _rewrite_one(
    item: dict | None,
    keys: tuple[str, ...],
    dest_rel: str,
    staging: Path,
    missing: list[str],
    *,
    dry_run: bool,
    sizes: list[int],
) -> str | None:
    if not isinstance(item, dict):
        return None
    rel = _item_rel(item, keys)
    if not rel:
        return None
    if rel.replace("\\", "/").startswith(PACK_PREFIXES):
        return rel.replace("\\", "/")
    src = resolve_media_path(rel, subfolder=str(item.get("subfolder") or ""), type_name=str(item.get("type") or "input"))
    if src is None:
        missing.append(rel)
        for key in keys:
            if key in item:
                item[key] = ""
        return None
    try:
        sizes.append(int(src.stat().st_size))
    except OSError:
        sizes.append(0)
    if not dry_run:
        _copy_file(src, staging / dest_rel.replace("/", os.sep))
    _set_media_fields(item, dest_rel, keys[0])
    return dest_rel


def _slot_index(item: dict, fallback: int) -> int:
    raw = item.get("index", item.get("slot", fallback))
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = fallback
    return n


def _rewrite_image_list(refs: list, folder: str, staging: Path, missing: list[str], dry_run: bool, sizes: list[int]) -> None:
    if not isinstance(refs, list):
        return
    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue
        idx = _slot_index(ref, i)
        if idx < 0:
            continue
        src = resolve_media_path(
            _item_rel(ref, IMAGE_KEYS),
            subfolder=str(ref.get("subfolder") or ""),
            type_name=str(ref.get("type") or "input"),
        )
        ext = _safe_ext(src or _item_rel(ref, IMAGE_KEYS), ".png")
        if ext not in IMAGE_EXTS:
            ext = ".png"
        dest = f"{folder}/Picture{idx + 1}{ext}"
        _rewrite_one(ref, IMAGE_KEYS, dest, staging, missing, dry_run=dry_run, sizes=sizes)


def _rewrite_audio_list(refs: list, folder: str, staging: Path, missing: list[str], dry_run: bool, sizes: list[int]) -> None:
    if not isinstance(refs, list):
        return
    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue
        idx = _slot_index(ref, i)
        if idx < 0:
            continue
        src = resolve_media_path(
            _item_rel(ref, AUDIO_KEYS),
            subfolder=str(ref.get("subfolder") or ""),
            type_name=str(ref.get("type") or "input"),
        )
        ext = _safe_ext(src or _item_rel(ref, AUDIO_KEYS), ".wav")
        if ext not in AUDIO_EXTS:
            ext = ".wav"
        dest = f"{folder}/Audio{idx + 1}{ext}"
        _rewrite_one(ref, AUDIO_KEYS, dest, staging, missing, dry_run=dry_run, sizes=sizes)


def _rewrite_video_list(refs: list, folder: str, staging: Path, missing: list[str], dry_run: bool, sizes: list[int]) -> None:
    if not isinstance(refs, list):
        return
    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue
        idx = _slot_index(ref, i)
        if idx < 0:
            continue
        src = resolve_media_path(
            _item_rel(ref, VIDEO_KEYS),
            subfolder=str(ref.get("subfolder") or ""),
            type_name=str(ref.get("type") or "input"),
        )
        ext = _safe_ext(src or _item_rel(ref, VIDEO_KEYS), ".mp4")
        if ext not in VIDEO_EXTS:
            ext = ".mp4"
        dest = f"{folder}/Video{idx + 1}{ext}"
        _rewrite_one(ref, VIDEO_KEYS, dest, staging, missing, dry_run=dry_run, sizes=sizes)
        preview_src = resolve_media_path(
            _item_rel(ref, PREVIEW_KEYS),
            subfolder=str(ref.get("subfolder") or ""),
            type_name=str(ref.get("type") or "input"),
        )
        if preview_src or _item_rel(ref, PREVIEW_KEYS):
            pext = _safe_ext(preview_src or _item_rel(ref, PREVIEW_KEYS), ".jpg")
            if pext not in IMAGE_EXTS:
                pext = ".jpg"
            _rewrite_one(
                ref, PREVIEW_KEYS, f"{folder}/Video{idx + 1}_preview{pext}",
                staging, missing, dry_run=dry_run, sizes=sizes,
            )
        if _item_rel(ref, PAIRED_AUDIO_KEYS):
            aext = _safe_ext(_item_rel(ref, PAIRED_AUDIO_KEYS), ".flac")
            if aext not in AUDIO_EXTS:
                aext = ".flac"
            _rewrite_one(
                ref, PAIRED_AUDIO_KEYS, f"{folder}/Video{idx + 1}_audio{aext}",
                staging, missing, dry_run=dry_run, sizes=sizes,
            )


def _rewrite_image_ref(item: dict | None, dest_stem: str, folder: str, staging: Path, missing: list[str], dry_run: bool, sizes: list[int]) -> None:
    if not isinstance(item, dict):
        return
    rel = _item_rel(item, IMAGE_KEYS)
    if not rel:
        return
    src = resolve_media_path(rel, subfolder=str(item.get("subfolder") or ""), type_name=str(item.get("type") or "input"))
    ext = _safe_ext(src or rel, ".jpg")
    if ext not in IMAGE_EXTS:
        ext = ".jpg"
    _rewrite_one(item, IMAGE_KEYS, f"{folder}/{dest_stem}{ext}", staging, missing, dry_run=dry_run, sizes=sizes)


def _rewrite_video_media(item: dict | None, dest_stem: str, folder: str, staging: Path, missing: list[str], dry_run: bool, sizes: list[int]) -> None:
    if not isinstance(item, dict):
        return
    rel = _item_rel(item, VIDEO_KEYS) or str(item.get("fileName") or "").replace("\\", "/").strip()
    if not rel:
        return
    src = resolve_media_path(rel, subfolder=str(item.get("subfolder") or ""), type_name=str(item.get("type") or "input"))
    ext = _safe_ext(src or rel, ".mp4")
    if ext not in VIDEO_EXTS:
        ext = ".mp4"
    _rewrite_one(item, VIDEO_KEYS, f"{folder}/{dest_stem}{ext}", staging, missing, dry_run=dry_run, sizes=sizes)


def _group_json(seg: dict) -> dict:
    out = {
        "id": seg.get("id") or "",
        "prompt": seg.get("prompt") or "",
        "negativePrompt": seg.get("negativePrompt") or "",
        "durationSec": seg.get("durationSec"),
        "frameCount": seg.get("frameCount") if seg.get("frameCount") is not None else seg.get("length"),
        "length": seg.get("length") if seg.get("length") is not None else seg.get("frameCount"),
        "start": seg.get("start"),
        "taskType": seg.get("taskType") or "",
        "refs": seg.get("refs") or [],
        "refAudios": seg.get("refAudios") or seg.get("ref_audios") or [],
        "refVideos": seg.get("refVideos") or seg.get("ref_videos") or [],
        "continuityFromPrev": seg.get("continuityFromPrev", seg.get("continuity_from_prev")),
        "refImageSize": seg.get("refImageSize") or seg.get("ref_image_size"),
    }
    if isinstance(seg.get("genImage"), dict):
        out["genImage"] = {
            "imageFile": seg["genImage"].get("imageFile") or "",
            "fileName": seg["genImage"].get("fileName") or "",
        }
    if seg.get("imageFile"):
        out["imageFile"] = seg.get("imageFile")
    for key in ("startImage", "endImage"):
        val = seg.get(key)
        if isinstance(val, dict) and (val.get("imageFile") or val.get("image_file")):
            out[key] = {
                "imageFile": val.get("imageFile") or val.get("image_file") or "",
                "width": val.get("width") or 0,
                "height": val.get("height") or 0,
            }
        else:
            out[key] = None
    return out


def _explode_card(card: dict, folder: str, staging: Path, missing: list[str], dry_run: bool, sizes: list[int]) -> None:
    _rewrite_image_list(card.get("refs") or [], folder, staging, missing, dry_run, sizes)
    _rewrite_audio_list(card.get("refAudios") or card.get("ref_audios") or [], folder, staging, missing, dry_run, sizes)
    _rewrite_video_list(card.get("refVideos") or card.get("ref_videos") or [], folder, staging, missing, dry_run, sizes)
    if isinstance(card.get("genImage"), dict):
        _rewrite_image_ref(card["genImage"], "start", folder, staging, missing, dry_run, sizes)
        if card["genImage"].get("imageFile"):
            card["imageFile"] = card["genImage"]["imageFile"]
    elif card.get("imageFile"):
        dummy = {"imageFile": card.get("imageFile"), "subfolder": card.get("subfolder") or "", "type": card.get("type") or "input"}
        _rewrite_image_ref(dummy, "start", folder, staging, missing, dry_run, sizes)
        card["imageFile"] = dummy.get("imageFile") or card.get("imageFile")
    _rewrite_image_ref(card.get("startImage"), "start", folder, staging, missing, dry_run, sizes)
    _rewrite_image_ref(card.get("endImage"), "end", folder, staging, missing, dry_run, sizes)


def _ascii_extra_name(src: Path, used: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", src.stem).strip("_")[:40] or "media"
    digest = hashlib.sha1(str(src).encode("utf-8", "replace")).hexdigest()[:8]
    if not re.fullmatch(r"[A-Za-z0-9_]+", stem):
        stem = "media"
    ext = _safe_ext(src, ".bin")
    name = f"{stem}_{digest}{ext}"
    n = 1
    while name in used:
        name = f"{stem}_{digest}_{n}{ext}"
        n += 1
    used.add(name)
    return f"extra/{name}"


def _walk_rewrite_leftovers(
    obj: Any,
    staging: Path,
    missing: list[str],
    dry_run: bool,
    sizes: list[int],
    extra_used: set[str],
) -> None:
    if isinstance(obj, list):
        for item in obj:
            _walk_rewrite_leftovers(item, staging, missing, dry_run, sizes, extra_used)
        return
    if not isinstance(obj, dict):
        return
    for keys, fallback_ext, allowed in (
        (IMAGE_KEYS, ".png", IMAGE_EXTS),
        (AUDIO_KEYS, ".wav", AUDIO_EXTS),
        (VIDEO_KEYS, ".mp4", VIDEO_EXTS),
        (PREVIEW_KEYS, ".jpg", IMAGE_EXTS),
        (PAIRED_AUDIO_KEYS, ".flac", AUDIO_EXTS),
    ):
        rel = _item_rel(obj, keys)
        if not rel or rel.replace("\\", "/").startswith(PACK_PREFIXES):
            continue
        src = resolve_media_path(rel, subfolder=str(obj.get("subfolder") or ""), type_name=str(obj.get("type") or "input"))
        ext = _safe_ext(src or rel, fallback_ext)
        if ext not in allowed:
            ext = fallback_ext
        dest = _ascii_extra_name(src or Path(rel), extra_used)
        # keep intended extension
        dest = str(Path(dest).with_suffix(ext)).replace("\\", "/")
        extra_used.add(Path(dest).name)
        _rewrite_one(obj, keys, dest, staging, missing, dry_run=dry_run, sizes=sizes)
    for key, val in obj.items():
        if key in IMAGE_KEYS + AUDIO_KEYS + VIDEO_KEYS + PREVIEW_KEYS + PAIRED_AUDIO_KEYS:
            continue
        if isinstance(val, (dict, list)):
            _walk_rewrite_leftovers(val, staging, missing, dry_run, sizes, extra_used)


def _task_combo(value: str) -> str:
    key = resolve_task_key(value or "t2v")
    spec = TASK_PROMPT_BY_KEY.get(key)
    if spec is None or key == "default":
        spec = TASK_PROMPT_BY_KEY["t2v"]
        key = "t2v"
    return task_type_option_label(spec)
