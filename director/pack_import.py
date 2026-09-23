"""Reading a pack back in: zip validation, extraction, timeline rebuild.

Import is deliberately forgiving in one direction only: it accepts both pack
format ids (upstream and Opt) and rebuilds a timeline from the folder scan when
``timeline.json`` is missing, because upstream writes 1-9 / 1-3 slots. What it
never does is trust the archive — :func:`_validate_zip_entry` rejects absolute
paths, ``..`` escapes and oversized entries before anything is written.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from aiohttp import web

from ..lib.task_prompts import resolve_task_key
from .pack_format import (
    AUDIO_FILE_RE,
    AUDIO_KEYS,
    END_FILE_RE,
    IMAGE_KEYS,
    MAX_SINGLE_FILE,
    MAX_UNCOMPRESSED,
    MAX_ZIP_ENTRIES,
    PACK_FORMATS,
    PACK_PREFIXES,
    PACK_VERSION,
    PAIRED_AUDIO_KEYS,
    PICTURE_FILE_RE,
    PREVIEW_KEYS,
    START_FILE_RE,
    VIDEO_FILE_RE,
    VIDEO_KEYS,
)
from .pack_io import (
    _input_dir,
    _is_ascii_pack_path,
    _is_under_dir,
    _posix,
    _read_json,
    _unlink_quiet,
    resolve_media_path,
)
from .pack_rewrite import _item_rel, _slot_index, _task_combo

log = logging.getLogger("ComfyUI-MiniMaxH3-Director-Opt.director.pack.import")


def _validate_zip_entry(name: str, info: zipfile.ZipInfo, uncompressed_total: int) -> str:
    rel = name.replace("\\", "/").strip()
    if not rel or rel.endswith("/"):
        return ""
    if rel.startswith("__MACOSX/") or rel.startswith("."):
        return ""
    parts = [p for p in rel.split("/") if p]
    if any(p == ".." for p in parts):
        raise ValueError(f"Unsafe path in pack: {name}")
    if any(p.startswith(".") for p in parts):
        return ""
    if not _is_ascii_pack_path(rel):
        raise ValueError(f"Pack paths must be ASCII: {name}")
    if info.file_size > MAX_SINGLE_FILE:
        raise ValueError("A file in the pack exceeds the size limit.")
    if uncompressed_total + info.file_size > MAX_UNCOMPRESSED:
        raise ValueError("Pack uncompressed size exceeds the limit.")
    return rel


def extract_pack_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise ValueError("Pack has too many files.")
        total = 0
        kept: list[tuple[str, zipfile.ZipInfo]] = []
        for info in infos:
            rel = _validate_zip_entry(info.filename, info, total)
            if not rel:
                continue
            total += max(0, int(info.file_size or 0))
            kept.append((rel, info))
        if not kept:
            raise ValueError("Pack zip is empty.")
        for rel, info in kept:
            target = dest / rel.replace("/", os.sep)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _scan_slot_files(folder: Path) -> dict[str, list[dict]]:
    refs: list[dict] = []
    audios: list[dict] = []
    videos: list[dict] = []
    start = None
    end = None
    if not folder.is_dir():
        return {"refs": refs, "refAudios": audios, "refVideos": videos, "startImage": start, "endImage": end}
    pack_folder = _posix(folder.name if folder.parent.name != "asset_groups" else f"asset_groups/{folder.name}")
    if folder.name == "shared_params":
        pack_folder = "shared_params"
    elif folder.parent.name == "asset_groups":
        pack_folder = f"asset_groups/{folder.name}"
    for path in folder.iterdir():
        if not path.is_file():
            continue
        name = path.name
        m = PICTURE_FILE_RE.fullmatch(name)
        if m:
            idx = int(m.group(1)) - 1
            rel = f"{pack_folder}/{name}"
            refs.append({"index": idx, "imageFile": rel, "fileName": name, "type": "input", "subfolder": pack_folder})
            continue
        m = AUDIO_FILE_RE.fullmatch(name)
        if m:
            idx = int(m.group(1)) - 1
            rel = f"{pack_folder}/{name}"
            audios.append({"index": idx, "audioFile": rel, "fileName": name, "type": "input", "subfolder": pack_folder})
            continue
        m = VIDEO_FILE_RE.fullmatch(name)
        if m:
            idx = int(m.group(1)) - 1
            rel = f"{pack_folder}/{name}"
            videos.append({"index": idx, "videoFile": rel, "fileName": name, "type": "input", "subfolder": pack_folder})
            continue
        if START_FILE_RE.fullmatch(name):
            rel = f"{pack_folder}/{name}"
            start = {"imageFile": rel, "fileName": name, "type": "input", "subfolder": pack_folder}
        elif END_FILE_RE.fullmatch(name):
            rel = f"{pack_folder}/{name}"
            end = {"imageFile": rel, "fileName": name, "type": "input", "subfolder": pack_folder}
    refs.sort(key=lambda r: int(r["index"]))
    audios.sort(key=lambda r: int(r["index"]))
    videos.sort(key=lambda r: int(r["index"]))
    return {"refs": refs, "refAudios": audios, "refVideos": videos, "startImage": start, "endImage": end}


def _merge_refs(json_refs: list | None, scanned: list) -> list:
    by_idx: dict[int, dict] = {}
    for item in scanned:
        by_idx[int(item["index"])] = dict(item)
    for item in json_refs or []:
        if not isinstance(item, dict):
            continue
        idx = _slot_index(item, -1)
        if idx < 0:
            continue
        merged = {**by_idx.get(idx, {}), **item}
        if not _item_rel(merged, IMAGE_KEYS + AUDIO_KEYS + VIDEO_KEYS):
            if idx in by_idx:
                merged = {**item, **by_idx[idx]}
        by_idx[idx] = merged
    return [by_idx[k] for k in sorted(by_idx)]


def _assemble_timeline(extracted: Path, pack_meta: dict) -> dict:
    shared_path = extracted / "shared_params" / "shared_params.json"
    shared = _read_json(shared_path) if shared_path.is_file() else {}
    scanned_shared = _scan_slot_files(extracted / "shared_params")
    global_block = {
        "taskType": _task_combo(str(pack_meta.get("taskType") or "t2v")),
        "prompt": shared.get("prompt") or "",
        "commonEnabled": bool(shared.get("commonEnabled")),
        "commonCollapsed": bool(shared.get("commonCollapsed")),
        "refs": _merge_refs(shared.get("refs"), scanned_shared["refs"]),
        "refAudios": _merge_refs(shared.get("refAudios") or shared.get("ref_audios"), scanned_shared["refAudios"]),
        "refVideos": _merge_refs(shared.get("refVideos") or shared.get("ref_videos"), scanned_shared["refVideos"]),
        "referenceVideo": {},
        "continuousReference": False,
    }
    groups_root = extracted / "asset_groups"
    group_dirs = sorted(
        [p for p in groups_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    ) if groups_root.is_dir() else []
    segments: list[dict] = []
    shots: list[dict] = []
    cursor = 0
    for i, gdir in enumerate(group_dirs):
        gj = gdir / "group.json"
        raw = _read_json(gj) if gj.is_file() else {}
        scanned = _scan_slot_files(gdir)
        fc = int(raw.get("frameCount") or raw.get("length") or 124)
        dur = raw.get("durationSec")
        start_img = raw.get("startImage") if isinstance(raw.get("startImage"), dict) else scanned["startImage"]
        end_img = raw.get("endImage") if isinstance(raw.get("endImage"), dict) else scanned["endImage"]
        gen = raw.get("genImage") if isinstance(raw.get("genImage"), dict) else None
        if scanned["startImage"] and not (gen and gen.get("imageFile")) and not (start_img and start_img.get("imageFile")):
            gen = {"imageFile": scanned["startImage"]["imageFile"], "fileName": scanned["startImage"]["fileName"]}
        seg = {
            "id": raw.get("id") or f"g{i}",
            "start": raw.get("start") if raw.get("start") is not None else cursor,
            "length": fc,
            "frameCount": fc,
            "durationSec": dur,
            "prompt": raw.get("prompt") or "",
            "negativePrompt": raw.get("negativePrompt") or "",
            "taskType": raw.get("taskType") or "",
            "refs": _merge_refs(raw.get("refs"), scanned["refs"]),
            "refAudios": _merge_refs(raw.get("refAudios") or raw.get("ref_audios"), scanned["refAudios"]),
            "refVideos": _merge_refs(raw.get("refVideos") or raw.get("ref_videos"), scanned["refVideos"]),
            "continuityFromPrev": raw.get("continuityFromPrev", raw.get("continuity_from_prev")),
            "refImageSize": raw.get("refImageSize") or raw.get("ref_image_size"),
            "genImage": gen or {"imageFile": ""},
            "imageFile": (gen or {}).get("imageFile") or raw.get("imageFile") or "",
            "startImage": start_img,
            "endImage": end_img,
        }
        segments.append(seg)
        shots.append({
            "id": seg["id"],
            "durationSec": dur,
            "prompt": seg["prompt"],
            "negativePrompt": seg["negativePrompt"],
            "continuityFromPrev": seg["continuityFromPrev"],
            "startImage": start_img,
            "endImage": end_img,
        })
        cursor += fc
    task_key = resolve_task_key(str(pack_meta.get("taskType") or global_block["taskType"]))
    output = pack_meta.get("output") if isinstance(pack_meta.get("output"), dict) else {}
    mode = "fl2v" if task_key == "fl2v" else ("video" if task_key in ("v2v", "rv2v") else "prompt_batch")
    timeline = {
        "version": 5,
        "timelineMode": mode,
        "editMode": "segment" if mode != "video" else "global",
        "frameRate": output.get("frameRate") or 24,
        "totalFrames": cursor or 124,
        "global": global_block,
        "output": output or {
            "mode": "fixed",
            "width": 864,
            "height": 480,
            "exportMode": "all",
            "audioMode": "generate",
            "refImageSize": "match",
            "continuityEnabled": False,
            "continuityOverlapFrames": 22,
            "continuityMode": "guide",
            "continuityRedraw": 0.10,
        },
        "segments": segments or [{
            "id": "g0",
            "start": 0,
            "length": 124,
            "frameCount": 124,
            "prompt": "",
            "refs": [],
            "refAudios": [],
            "refVideos": [],
            "genImage": {"imageFile": ""},
        }],
        "video": {"fileName": "", "videoFile": "", "subfolder": "", "type": "input", "frames": [], "frameMap": []},
        "videoClips": [],
        "runSelectEnabled": False,
        "runSelection": [],
    }
    if task_key == "fl2v":
        timeline["shots"] = shots
        timeline["keyframes"] = []
    src_dir = extracted / "source_video"
    if src_dir.is_dir():
        clips = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
        video_clips = []
        for i, path in enumerate(clips):
            rel = f"source_video/{path.name}"
            rec = {"videoFile": rel, "fileName": path.name, "type": "input", "subfolder": "source_video", "frames": []}
            video_clips.append(rec)
            if i == 0:
                timeline["video"] = dict(rec)
        if video_clips:
            timeline["videoClips"] = video_clips
    return timeline


def _prefix_pack_paths(obj: Any, prefix: str) -> None:
    if isinstance(obj, list):
        for item in obj:
            _prefix_pack_paths(item, prefix)
        return
    if not isinstance(obj, dict):
        return
    for key in IMAGE_KEYS + AUDIO_KEYS + VIDEO_KEYS + PREVIEW_KEYS + PAIRED_AUDIO_KEYS:
        val = obj.get(key)
        if not val:
            continue
        rel = str(val).replace("\\", "/").strip().lstrip("/")
        if rel.startswith("minimax_director_packs/"):
            continue
        if rel.startswith(PACK_PREFIXES):
            new_rel = f"{prefix}/{rel}"
            obj[key] = new_rel
            obj["type"] = "input"
            parent = str(Path(new_rel).parent).replace("\\", "/")
            obj["subfolder"] = "" if parent in (".", "") else parent
            obj["fileName"] = Path(new_rel).name
    for key, val in obj.items():
        if key in IMAGE_KEYS + AUDIO_KEYS + VIDEO_KEYS + PREVIEW_KEYS + PAIRED_AUDIO_KEYS:
            continue
        if isinstance(val, (dict, list)):
            _prefix_pack_paths(val, prefix)


def _copy_tree_media(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = _posix(path.relative_to(src))
        if rel.endswith(".json"):
            continue
        if not _is_ascii_pack_path(rel):
            continue
        ext = path.suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"
        if ext not in MEDIA_EXTS:
            continue
        target = dest / rel.replace("/", os.sep)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _collect_missing_media(obj: Any, dest: Path, rel_prefix: str, missing: list[str]) -> None:
    if isinstance(obj, list):
        for item in obj:
            _collect_missing_media(item, dest, rel_prefix, missing)
        return
    if not isinstance(obj, dict):
        return
    prefix = rel_prefix.strip("/") + "/"
    for keys in (IMAGE_KEYS, AUDIO_KEYS, VIDEO_KEYS, PREVIEW_KEYS, PAIRED_AUDIO_KEYS):
        rel = _item_rel(obj, keys)
        if not rel:
            continue
        rel = rel.replace("\\", "/").strip().lstrip("/")
        if rel.startswith(prefix):
            inner = rel[len(prefix):]
        elif rel.startswith(PACK_PREFIXES):
            inner = rel
        else:
            continue
        path = dest / inner.replace("/", os.sep)
        if path.is_file():
            continue
        missing.append(rel)
        for key in keys:
            if key in obj:
                obj[key] = ""
    for key, val in obj.items():
        if key in IMAGE_KEYS + AUDIO_KEYS + VIDEO_KEYS + PREVIEW_KEYS + PAIRED_AUDIO_KEYS:
            continue
        if isinstance(val, (dict, list)):
            _collect_missing_media(val, dest, rel_prefix, missing)


def import_extracted_pack(extracted: Path) -> dict[str, Any]:
    pack_path = extracted / "pack.json"
    pack_meta: dict[str, Any] = {}
    if pack_path.is_file():
        pack_meta = _read_json(pack_path)
        if not isinstance(pack_meta, dict):
            pack_meta = {}
    fmt = str(pack_meta.get("format") or "")
    if pack_path.is_file() and fmt and fmt not in PACK_FORMATS:
        raise ValueError(f"Unsupported pack format: {fmt}")
    version = int(pack_meta.get("formatVersion") or 1)
    if version > PACK_VERSION:
        raise ValueError(f"Pack formatVersion {version} is newer than this plugin.")

    timeline_path = extracted / "timeline.json"
    if timeline_path.is_file():
        timeline = _read_json(timeline_path)
        if not isinstance(timeline, dict):
            raise ValueError("timeline.json is invalid.")
    else:
        timeline = _assemble_timeline(extracted, pack_meta)

    pack_id = uuid.uuid4().hex[:12]
    rel_prefix = f"minimax_director_packs/{pack_id}"
    dest = _input_dir() / "minimax_director_packs" / pack_id
    _copy_tree_media(extracted, dest)
    _prefix_pack_paths(timeline, rel_prefix)
    missing: list[str] = []
    _collect_missing_media(timeline, dest, rel_prefix, missing)
    unique_missing: list[str] = []
    seen_m: set[str] = set()
    for item in missing:
        if item not in seen_m:
            seen_m.add(item)
            unique_missing.append(item)

    widgets = pack_meta.get("widgets") if isinstance(pack_meta.get("widgets"), dict) else {}
    task_type = widgets.get("task_type") or _task_combo(str(pack_meta.get("taskType") or (timeline.get("global") or {}).get("taskType") or "t2v"))
    widgets = {**widgets, "task_type": task_type}
    if isinstance(timeline.get("global"), dict):
        timeline["global"]["taskType"] = task_type

    return {
        "timeline": timeline,
        "widgets": widgets,
        "packId": pack_id,
        "missing": unique_missing,
    }


def _resolve_uploaded_zip(name: str, subfolder: str = "", type_name: str = "input") -> Path:
    path = resolve_media_path(name, subfolder=subfolder, type_name=type_name)
    if path is None or not path.is_file():
        raise ValueError("Uploaded pack zip was not found.")
    if path.suffix.lower() not in {".zip"}:
        raise ValueError("Pack must be a .zip file.")
    return path


async def minimax_import_pack(request):
    extracted: Path | None = None
    upload_dir: Path | None = None
    input_zip: Path | None = None
    try:
        ctype = request.content_type or ""
        if "multipart" in ctype:
            post = await request.post()
            upload = post.get("pack")
            if upload is None or not hasattr(upload, "file"):
                return web.Response(status=400, text="Missing pack file.")
            upload_dir = Path(tempfile.mkdtemp(prefix="mmx_pack_up_"))
            zip_path = upload_dir / "pack.zip"
            with open(zip_path, "wb") as out:
                shutil.copyfileobj(upload.file, out)
        else:
            body = await request.json()
            zip_path = _resolve_uploaded_zip(
                str(body.get("filename") or body.get("name") or ""),
                subfolder=str(body.get("subfolder") or ""),
                type_name=str(body.get("type") or "input"),
            )
            if zip_path.suffix.lower() == ".zip" and _is_under_dir(zip_path, _input_dir()):
                input_zip = zip_path
        extracted = Path(tempfile.mkdtemp(prefix="mmx_pack_ex_"))
        extract_pack_zip(zip_path, extracted)
        result = import_extracted_pack(extracted)
        return web.json_response(result)
    except Exception as exc:
        log.warning("Director pack import failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    finally:
        if extracted is not None:
            shutil.rmtree(extracted, ignore_errors=True)
        if upload_dir is not None:
            shutil.rmtree(upload_dir, ignore_errors=True)
        if input_zip is not None:
            _unlink_quiet(input_zip)
