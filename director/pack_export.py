"""Building a pack and serving it to the browser.

``build_export_pack`` writes the zip (script + assets + the rewritten timeline),
then the two handlers either report the resulting file or stream it out.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import re
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from ..lib.task_prompts import resolve_task_key
from .pack_format import INLINE_JSON_MAX, PACK_FORMAT, PACK_VERSION
from .pack_io import (
    _is_ascii_pack_path,
    _pack_export_root,
    _posix,
    _purge_pack_exports,
    _send_zip_file,
    _unlink_quiet,
    _write_json,
)
from .pack_rewrite import (
    _explode_card,
    _group_json,
    _rewrite_audio_list,
    _rewrite_image_list,
    _rewrite_image_ref,
    _rewrite_video_list,
    _rewrite_video_media,
    _walk_rewrite_leftovers,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director-Opt.director.pack.export")


def build_export_pack(timeline: dict, widgets: dict | None = None, *, dry_run: bool = False) -> dict[str, Any]:
    if not isinstance(timeline, dict):
        raise ValueError("timeline must be an object")
    data = copy.deepcopy(timeline)
    widgets = widgets if isinstance(widgets, dict) else {}
    missing: list[str] = []
    sizes: list[int] = []
    staging: Path | None = None
    if not dry_run:
        staging = Path(tempfile.mkdtemp(prefix="mmx_pack_"))

    global_block = data.setdefault("global", {})
    if not isinstance(global_block, dict):
        global_block = {}
        data["global"] = global_block
    shared_folder = "shared_params"
    _rewrite_image_list(global_block.get("refs") or [], shared_folder, staging or Path("."), missing, dry_run, sizes)
    _rewrite_audio_list(global_block.get("refAudios") or global_block.get("ref_audios") or [], shared_folder, staging or Path("."), missing, dry_run, sizes)
    _rewrite_video_list(global_block.get("refVideos") or global_block.get("ref_videos") or [], shared_folder, staging or Path("."), missing, dry_run, sizes)
    if isinstance(global_block.get("referenceVideo"), dict):
        _rewrite_video_media(global_block["referenceVideo"], "reference_video", shared_folder, staging or Path("."), missing, dry_run, sizes)
    if isinstance(global_block.get("genImage"), dict):
        _rewrite_image_ref(global_block["genImage"], "start", shared_folder, staging or Path("."), missing, dry_run, sizes)

    task_key = resolve_task_key(
        str(widgets.get("task_type") or widgets.get("taskType") or global_block.get("taskType") or "t2v")
    )
    cards: list[dict] = []
    if task_key == "fl2v" and isinstance(data.get("shots"), list) and data["shots"]:
        cards = [c for c in data["shots"] if isinstance(c, dict)]
    elif isinstance(data.get("segments"), list) and data["segments"]:
        cards = [c for c in data["segments"] if isinstance(c, dict)]

    for i, card in enumerate(cards):
        folder = f"asset_groups/{i + 1:02d}"
        _explode_card(card, folder, staging or Path("."), missing, dry_run, sizes)

    _rewrite_video_media(data.get("video") if isinstance(data.get("video"), dict) else None, "clip_1", "source_video", staging or Path("."), missing, dry_run, sizes)
    clips = data.get("videoClips")
    if isinstance(clips, list):
        for i, clip in enumerate(clips):
            stem = "clip_1" if i == 0 else f"clip_{i + 1}"
            _rewrite_video_media(clip if isinstance(clip, dict) else None, stem, "source_video", staging or Path("."), missing, dry_run, sizes)

    extra_used: set[str] = set()
    _walk_rewrite_leftovers(data, staging or Path("."), missing, dry_run, sizes, extra_used)

    total_bytes = int(sum(sizes))
    unique_missing = []
    seen_m: set[str] = set()
    for item in missing:
        if item not in seen_m:
            seen_m.add(item)
            unique_missing.append(item)

    pack_meta = {
        "format": PACK_FORMAT,
        "formatVersion": PACK_VERSION,
        "taskType": task_key,
        "widgets": {
            k: widgets[k]
            for k in ("steps", "sampler", "scheduler", "cfg", "shift_video", "shift_audio", "seed", "task_type")
            if k in widgets
        },
        "output": data.get("output") if isinstance(data.get("output"), dict) else {},
    }
    shared_json = {
        "commonEnabled": bool(global_block.get("commonEnabled") or global_block.get("common_enabled")),
        "commonCollapsed": bool(global_block.get("commonCollapsed") or global_block.get("common_collapsed")),
        "prompt": global_block.get("prompt") or "",
        "refs": global_block.get("refs") or [],
        "refAudios": global_block.get("refAudios") or global_block.get("ref_audios") or [],
        "refVideos": global_block.get("refVideos") or global_block.get("ref_videos") or [],
    }

    result: dict[str, Any] = {
        "missing": unique_missing,
        "totalBytes": total_bytes,
        "fileCount": len(sizes),
        "taskType": task_key,
    }
    if dry_run:
        return result

    assert staging is not None
    try:
        _write_json(staging / "pack.json", pack_meta)
        _write_json(staging / "shared_params" / "shared_params.json", shared_json)
        (staging / "shared_params").mkdir(exist_ok=True)
        (staging / "asset_groups").mkdir(exist_ok=True)
        for i, card in enumerate(cards):
            folder = staging / "asset_groups" / f"{i + 1:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            _write_json(folder / "group.json", _group_json(card))
        _write_json(staging / "timeline.json", data)

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        download_name = f"MiniMaxH3DirectorOpt-{task_key}-{stamp}.mmxpack.zip"
        stored_name = f"{uuid.uuid4().hex}.mmxpack.zip"
        _purge_pack_exports(keep=stored_name)
        zip_path = _pack_export_root() / stored_name
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in staging.rglob("*"):
                if not path.is_file():
                    continue
                rel = _posix(path.relative_to(staging))
                if not _is_ascii_pack_path(rel):
                    raise ValueError(f"Non-ASCII pack path: {rel}")
                zf.write(path, rel)
        zip_bytes = int(zip_path.stat().st_size)
        if zip_bytes <= 0:
            _unlink_quiet(zip_path)
            raise ValueError("Export produced an empty pack zip.")
        result["filename"] = stored_name
        result["downloadName"] = download_name
        result["bytes"] = zip_bytes
        return result
    finally:
        shutil.rmtree(staging, ignore_errors=True)


async def minimax_export_pack(request):
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")
    timeline = body.get("timeline")
    if not isinstance(timeline, dict):
        return web.Response(status=400, text="Missing timeline object.")
    widgets = body.get("widgets") if isinstance(body.get("widgets"), dict) else {}
    dry_run = bool(body.get("dryRun") or body.get("dry_run"))
    try:
        result = build_export_pack(timeline, widgets, dry_run=dry_run)
    except Exception as exc:
        log.warning("Director pack export failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    if dry_run:
        return web.json_response(result)
    filename = str(result.get("filename") or "")
    if not re.fullmatch(r"[A-Za-z0-9]+\.mmxpack\.zip", filename):
        return web.Response(status=500, text="Export produced an invalid pack filename.")
    path = _pack_export_root() / filename
    download_name = str(result.get("downloadName") or filename)
    download_name = re.sub(r"[^A-Za-z0-9._-]+", "_", download_name) or filename
    extra = {
        "X-Pack-Download-Name": download_name,
        "X-Pack-Missing": json.dumps(result.get("missing") or [], ensure_ascii=True),
    }
    try:
        size = int(path.stat().st_size)
    except OSError:
        return web.Response(status=404, text="Pack not found.")
    if size <= 0:
        return web.Response(status=500, text="Export produced an empty pack zip.")
    # Small packs: JSON + base64. fetchApi POST JSON is reliable; GET FileResponse
    # on Windows is HTTP 200 with an empty body (0-byte .zip).
    if size <= INLINE_JSON_MAX:
        data = path.read_bytes()
        _unlink_quiet(path)
        result["zipB64"] = base64.b64encode(data).decode("ascii")
        result["downloadName"] = download_name
        result["bytes"] = size
        return web.json_response(result)
    return await _send_zip_file(request, path, download_name, extra, unlink_after=True)


async def minimax_download_pack(request):
    filename = str(request.query.get("filename") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9]+\.mmxpack\.zip", filename):
        return web.Response(status=400, text="Invalid pack filename.")
    path = _pack_export_root() / filename
    if not path.is_file():
        return web.Response(status=404, text="Pack not found.")
    download_name = str(request.query.get("download") or filename)
    download_name = re.sub(r"[^A-Za-z0-9._-]+", "_", download_name) or filename
    return await _send_zip_file(request, path, download_name)
