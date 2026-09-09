"""HTTP routes for MiniMax H3 Director Opt (chunked video upload)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import folder_paths
from aiohttp import web
from server import PromptServer

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director")

CHUNK_ROOT = os.path.join(folder_paths.get_temp_directory(), "minimax_upload_chunks")
REF_AUDIO_CHUNK_ROOT = os.path.join(folder_paths.get_temp_directory(), "minimax_ref_audio_chunks")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".mts", ".ts"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
_WIN_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WIN_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)
_SAFE_EXT = re.compile(r"\.[A-Za-z0-9]{1,8}$")
_ROUTES_REGISTERED = False


def _safe_basename(name: str) -> str:
    """Keep CJK names; only strip path pieces and Windows-illegal characters."""
    base = os.path.basename(str(name or "upload.bin").replace("\\", "/"))
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if not _SAFE_EXT.fullmatch(ext):
        ext = ".bin"
    if ext == ".jpeg":
        ext = ".jpg"
    stem = _WIN_ILLEGAL.sub("_", stem).rstrip(" .")[:80]
    if not stem or _WIN_RESERVED.match(stem):
        stem = "upload"
    return f"{stem}{ext}"


def _get_media_exts(kind: str) -> set[str]:
    kind = str(kind or "").strip().lower()
    if kind == "image":
        return IMAGE_EXTS
    if kind == "video":
        return VIDEO_EXTS
    if kind == "audio":
        return AUDIO_EXTS
    if kind == "reference_audio":
        return AUDIO_EXTS | VIDEO_EXTS
    raise ValueError("kind must be image, video, audio or reference_audio")


def _peek_image_size(path: str) -> tuple[int, int]:
    """Read width/height from the image header without decoding pixels."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            return int(w or 0), int(h or 0)
    except Exception:
        return 0, 0


#: Cap on how many Director rendered clips are listed. The cache tree grows one
#: file group per segment per node, so an unbounded walk would stall the picker
#: on a large project; newest-first means the cap costs relevance, not recency.
MAX_DIRECTOR_CLIPS = 300


def _list_director_clips(exclude_rel: set[str] | None = None) -> list[dict]:
    """Rendered Director clips (``*_clip.mp4``) living under ``output/``.

    Video picker entries for the "use something I already generated as a
    reference" case. They are returned with ``type="output"``, which is exactly
    what :func:`resolve_video_path` and ``/api/view`` need now that both honour
    it; before that they were unlistable *and* unresolvable.

    Grouped by the parent directory (``<workflow slug>/node_<id>``) rather than
    reported per file: mediainfo probes are the expensive part, so each group is
    probed once for dimensions and its members inherit them. Files are skipped
    when a directory can no longer be read, keeping one bad node from emptying
    the whole list.
    """
    try:
        from .cache_layout import CACHE_ROOT, CLIP_SUFFIX, output_root

        out_root = Path(output_root())
        cache_root = out_root / CACHE_ROOT
    except Exception as exc:  # pragma: no cover - import guard
        log.warning("MiniMax H3 Director Opt cache root unavailable: %s", exc)
        return []
    if not cache_root.is_dir():
        return []

    groups: dict[str, dict] = {}
    for dirpath, _dirs, files in os.walk(cache_root):
        rel_dir = ""
        try:
            rel_dir = os.path.relpath(dirpath, cache_root).replace("\\", "/")
        except ValueError:
            continue
        if rel_dir == ".":
            rel_dir = ""
        members: list[str] = []
        for name in files:
            if not name.endswith(CLIP_SUFFIX):
                continue
            abs_path = os.path.join(dirpath, name)
            try:
                mtime = float(os.stat(abs_path).st_mtime)
            except OSError:
                continue
            rel_path = f"{CACHE_ROOT}/{rel_dir}/{name}" if rel_dir else f"{CACHE_ROOT}/{name}"
            members.append((abs_path, name, rel_path, mtime))
        if not members:
            continue
        members.sort(key=lambda item: (-item[3], item[1]))
        groups[rel_dir] = {"members": members, "newest": max(m[3] for m in members)}

    # Newest group first: long projects bury today's renders under months of runs.
    ordered = sorted(groups.items(), key=lambda kv: (-kv[1]["newest"], kv[0]))

    items: list[dict] = []
    seen: set[str] = set(exclude_rel or set())
    peek_video = None
    try:
        from ..lib.video_io import peek_video_size as peek_video
    except Exception:  # pragma: no cover - probe failures are non-fatal
        peek_video = None

    for _rel_dir, info in ordered:
        if len(items) >= MAX_DIRECTOR_CLIPS:
            break
        width, height = 0, 0
        # Probe the newest member once: same node ⇒ same dimensions.
        first_abs = info["members"][0][0]
        if peek_video is not None:
            try:
                width, height = peek_video(first_abs)
            except Exception:
                width, height = 0, 0
        for abs_path, name, rel_path, mtime in info["members"]:
            if len(items) >= MAX_DIRECTOR_CLIPS:
                break
            if rel_path in seen:
                # Guard against an input-dir file whose relative path happens to
                # match: the picker keys rows by relPath, so a collision would
                # silently bind one row to the other's ``type``.
                continue
            seen.add(rel_path)
            items.append(
                {
                    "name": name,
                    "fileName": name,
                    "relPath": rel_path,
                    "subfolder": os.path.dirname(rel_path).replace("\\", "/"),
                    "type": "output",
                    "modified": mtime,
                    "width": width,
                    "height": height,
                    "mediaKind": "video",
                    "source": "director-cache",
                }
            )
    return items


def _list_input_media(kind: str, include_cache: bool = False) -> list[dict]:
    input_dir = folder_paths.get_input_directory()
    exts = _get_media_exts(kind)
    peek_video = None
    if kind == "video":
        from ..lib.video_io import peek_video_size as peek_video
    items: list[dict] = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in exts:
                continue
            abs_path = os.path.join(root, name)
            try:
                stat = os.stat(abs_path)
            except OSError:
                continue
            try:
                rel_path = os.path.relpath(abs_path, input_dir).replace("\\", "/")
            except ValueError:
                continue
            if rel_path.startswith(".."):
                continue
            subfolder = os.path.dirname(rel_path).replace("\\", "/")
            if subfolder == ".":
                subfolder = ""
            width, height = (0, 0)
            if ext in IMAGE_EXTS:
                width, height = _peek_image_size(abs_path)
            elif peek_video is not None and ext in VIDEO_EXTS:
                try:
                    width, height = peek_video(abs_path)
                except Exception:
                    width, height = 0, 0
            items.append(
                {
                    "name": name,
                    "fileName": name,
                    "relPath": rel_path,
                    "subfolder": subfolder,
                    "type": "input",
                    "modified": float(stat.st_mtime),
                    "width": width,
                    "height": height,
                    "mediaKind": "video" if ext in VIDEO_EXTS else (
                        "audio" if ext in AUDIO_EXTS else "image"
                    ),
                }
            )
    items.sort(key=lambda item: (-item["modified"], item["relPath"]))
    if include_cache and kind == "video":
        # Appended rather than merged inline: uploads and renders solve different
        # problems, and keeping them contiguous makes each block scannable.
        items.extend(_list_director_clips(exclude_rel={item["relPath"] for item in items}))
    return items


async def minimax_upload_video_chunk(request):
    try:
        post = await request.post()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid upload: {exc}")

    upload_id = str(post.get("upload_id") or "").strip()
    filename = _safe_basename(post.get("filename"))
    chunk_field = post.get("chunk")
    if not upload_id or chunk_field is None:
        return web.Response(status=400, text="Missing upload_id or chunk.")

    if ".." in upload_id or "/" in upload_id or "\\" in upload_id:
        return web.Response(status=400, text="Invalid upload_id.")

    try:
        chunk_index = int(post.get("chunk_index", 0))
        total_chunks = int(post.get("total_chunks", 1))
    except (TypeError, ValueError):
        return web.Response(status=400, text="Invalid chunk index.")

    if total_chunks < 1 or chunk_index < 0 or chunk_index >= total_chunks:
        return web.Response(status=400, text="Chunk index out of range.")

    session_dir = os.path.join(CHUNK_ROOT, upload_id)
    os.makedirs(session_dir, exist_ok=True)
    part_path = os.path.join(session_dir, f"{chunk_index:06d}.part")

    with open(part_path, "wb") as out:
        while True:
            block = chunk_field.file.read(1024 * 1024)
            if not block:
                break
            out.write(block)

    if chunk_index + 1 < total_chunks:
        return web.json_response({"status": "ok", "chunk_index": chunk_index})

    input_dir = folder_paths.get_input_directory()
    out_path = os.path.join(input_dir, filename)
    if os.path.exists(out_path):
        stem, ext = os.path.splitext(filename)
        for n in range(1, 1000):
            candidate = f"{stem}_{n}{ext}"
            candidate_path = os.path.join(input_dir, candidate)
            if not os.path.exists(candidate_path):
                out_path = candidate_path
                filename = candidate
                break

    with open(out_path, "wb") as out:
        for i in range(total_chunks):
            part = os.path.join(session_dir, f"{i:06d}.part")
            if not os.path.isfile(part):
                shutil.rmtree(session_dir, ignore_errors=True)
                return web.Response(status=400, text=f"Missing chunk {i}.")
            with open(part, "rb") as src:
                shutil.copyfileobj(src, out)

    shutil.rmtree(session_dir, ignore_errors=True)
    log.info("MiniMax H3 Director Opt uploaded video to input/: %s", filename)
    return web.json_response({"name": filename, "subfolder": "", "type": "input"})


def _reference_audio_result(path: str, *, reused: bool, source_kind: str) -> dict:
    name = os.path.basename(path)
    return {
        "name": name,
        "fileName": name,
        "relPath": name,
        "subfolder": "",
        "type": "input",
        "reused": bool(reused),
        "sourceKind": source_kind,
    }


def _files_identical(first: str, second: str) -> bool:
    """Match ComfyUI upload dedupe without assigning content-derived filenames."""
    try:
        if os.path.getsize(first) != os.path.getsize(second):
            return False
        with open(first, "rb") as left, open(second, "rb") as right:
            while True:
                left_block = left.read(4 * 1024 * 1024)
                right_block = right.read(4 * 1024 * 1024)
                if left_block != right_block:
                    return False
                if not left_block:
                    return True
    except OSError:
        return False


def _place_in_input_like_comfy_upload(temp_path: str, filename: str) -> tuple[str, bool]:
    """Use ComfyUI's non-overwrite rule: reuse identical, otherwise append ` (n)`."""
    input_dir = folder_paths.get_input_directory()
    filename = _safe_basename(filename)
    stem, ext = os.path.splitext(filename)
    candidate_name = filename
    index = 1
    while True:
        candidate_path = os.path.join(input_dir, candidate_name)
        if not os.path.exists(candidate_path):
            os.replace(temp_path, candidate_path)
            return candidate_path, False
        if _files_identical(candidate_path, temp_path):
            os.remove(temp_path)
            return candidate_path, True
        candidate_name = f"{stem} ({index}){ext}"
        index += 1


def _prepare_reference_audio(source_path: str, display_name: str) -> dict:
    """Extract a video's first audio stream and place it directly in input/."""
    if not os.path.isfile(source_path) or os.path.getsize(source_path) <= 0:
        raise ValueError("Reference audio source is empty or missing.")
    ext = os.path.splitext(display_name or source_path)[1].lower()
    if ext not in VIDEO_EXTS:
        raise ValueError("Selected source is not a supported video.")

    safe_name = _safe_basename(display_name or os.path.basename(source_path))
    safe_stem = os.path.splitext(safe_name)[0] or "reference_audio"
    output_name = f"{safe_stem}.flac"
    output_dir = folder_paths.get_input_directory()

    from ..lib.audio_io import _ffmpeg_bin

    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is unavailable; cannot extract audio from video.")
    tmp_path = os.path.join(output_dir, f".minimax_ref_audio_{uuid.uuid4().hex}.flac")
    args = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        source_path,
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "flac",
        "-compression_level",
        "5",
        "-y",
        tmp_path,
    ]
    try:
        result = subprocess.run(args, capture_output=True, check=False)
        if result.returncode != 0 or not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) <= 0:
            error = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or "The selected video has no decodable audio stream.")
        output_path, reused = _place_in_input_like_comfy_upload(tmp_path, output_name)
    finally:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
    return _reference_audio_result(output_path, reused=reused, source_kind="video")


async def minimax_extract_reference_audio(request):
    """Extract an existing input video's audio immediately into input/."""
    try:
        body = await request.json()
        video_file = str(body.get("videoFile") or body.get("relPath") or "").strip()
        if not video_file:
            return web.Response(status=400, text="Missing videoFile.")
        from ..lib.video_io import resolve_video_path

        clip = {
            "videoFile": video_file,
            "fileName": str(body.get("fileName") or os.path.basename(video_file)),
            "subfolder": str(body.get("subfolder") or ""),
            "type": str(body.get("type") or "input"),
        }
        source_path = resolve_video_path(clip)
        if os.path.splitext(source_path)[1].lower() not in VIDEO_EXTS:
            return web.Response(status=400, text="Selected source is not a supported video.")
        result = await asyncio.to_thread(
            _prepare_reference_audio,
            source_path,
            clip["fileName"] or os.path.basename(source_path),
        )
        return web.json_response(result)
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt reference audio extraction failed: %s", exc)
        return web.Response(status=400, text=str(exc))


async def minimax_prepare_reference_audio_chunk(request):
    """Receive large local audio/video; store audio or extract video audio into input/."""
    session_dir = ""
    try:
        post = await request.post()
        upload_id = str(post.get("upload_id") or "").strip()
        filename = _safe_basename(post.get("filename"))
        chunk_field = post.get("chunk")
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", upload_id) or chunk_field is None:
            return web.Response(status=400, text="Invalid reference audio upload.")
        try:
            chunk_index = int(post.get("chunk_index", 0))
            total_chunks = int(post.get("total_chunks", 1))
        except (TypeError, ValueError):
            return web.Response(status=400, text="Invalid chunk index.")
        if total_chunks < 1 or chunk_index < 0 or chunk_index >= total_chunks:
            return web.Response(status=400, text="Chunk index out of range.")
        source_ext = os.path.splitext(filename)[1].lower()
        if source_ext not in AUDIO_EXTS | VIDEO_EXTS:
            return web.Response(status=400, text="Unsupported reference audio source format.")

        session_dir = os.path.join(REF_AUDIO_CHUNK_ROOT, upload_id)
        os.makedirs(session_dir, exist_ok=True)
        part_path = os.path.join(session_dir, f"{chunk_index:06d}.part")
        with open(part_path, "wb") as out:
            while True:
                block = chunk_field.file.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        if chunk_index + 1 < total_chunks:
            response = web.json_response({"status": "ok", "chunk_index": chunk_index})
            session_dir = ""
            return response

        source_path = os.path.join(session_dir, filename)
        with open(source_path, "wb") as out:
            for index in range(total_chunks):
                part = os.path.join(session_dir, f"{index:06d}.part")
                if not os.path.isfile(part):
                    raise ValueError(f"Missing chunk {index}.")
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, out)
        if source_ext in AUDIO_EXTS:
            output_path, reused = await asyncio.to_thread(
                _place_in_input_like_comfy_upload,
                source_path,
                filename,
            )
            result = _reference_audio_result(output_path, reused=reused, source_kind="audio")
        else:
            result = await asyncio.to_thread(_prepare_reference_audio, source_path, filename)
        return web.json_response(result)
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt local reference audio preparation failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    finally:
        if session_dir:
            shutil.rmtree(session_dir, ignore_errors=True)


async def minimax_probe_video(request):
    try:
        if request.can_read_body and request.content_type == "application/json":
            body = await request.json()
        else:
            body = dict(request.query)
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid request: {exc}")

    video_file = str(body.get("videoFile") or body.get("video_file") or "").strip()
    if not video_file:
        return web.Response(status=400, text="Missing videoFile.")

    from ..lib.video_io import probe_video_clip

    clip = {
        "videoFile": video_file,
        "fileName": os.path.basename(video_file),
        "subfolder": str(body.get("subfolder") or "").strip(),
        "type": str(body.get("type") or "input").strip() or "input",
    }
    try:
        info = probe_video_clip(clip)
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt video probe failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    return web.json_response(info)


async def minimax_list_input_media(request):
    try:
        kind = str(request.query.get("kind") or "").strip().lower()
        if not kind:
            return web.Response(status=400, text="Missing kind.")
        # Generated Director renders live under output/, so they only appear when
        # the caller asks for them — and only for video, which is the only kind
        # they can be.
        include_cache = str(request.query.get("includeCache") or "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        items = _list_input_media(kind, include_cache=include_cache)
    except ValueError as exc:
        return web.Response(status=400, text=str(exc))
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt list input media failed: %s", exc)
        return web.Response(status=500, text=str(exc))
    return web.json_response({"items": items})


async def minimax_detect_shots(request):
    """Detect shot boundaries with PySceneDetect; return logical cut frames."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    from ..lib.shot_detect import (
        detect_timeline_shot_cuts,
        scenedetect_available,
        scenedetect_install_hint,
    )

    if not scenedetect_available():
        return web.Response(
            status=400,
            text=(
                "PySceneDetect is not installed in ComfyUI's Python "
                f"({__import__('sys').executable}). "
                f"Run: {scenedetect_install_hint()}"
            ),
        )

    try:
        frame_rate = float(body.get("frameRate") or body.get("frame_rate") or 24)
    except (TypeError, ValueError):
        frame_rate = 24.0
    try:
        total_frames = int(body.get("totalFrames") or body.get("total_frames") or 0)
    except (TypeError, ValueError):
        return web.Response(status=400, text="Invalid totalFrames.")

    sensitivity = str(body.get("sensitivity") or "medium").strip().lower()
    try:
        min_shot_frames = int(body.get("minShotFrames") or body.get("min_shot_frames") or 12)
    except (TypeError, ValueError):
        min_shot_frames = 12

    clips_in = body.get("clips")
    clips: list[dict] = []
    if isinstance(clips_in, list) and clips_in:
        for item in clips_in:
            if not isinstance(item, dict):
                continue
            video_file = str(item.get("videoFile") or item.get("video_file") or "").strip()
            if not video_file:
                continue
            clips.append(
                {
                    "videoFile": video_file,
                    "fileName": os.path.basename(video_file),
                    "subfolder": str(item.get("subfolder") or "").strip(),
                    "type": str(item.get("type") or "input").strip() or "input",
                    "logicalStart": item.get("logicalStart", item.get("logical_start", 0)),
                    "logicalEnd": item.get("logicalEnd", item.get("logical_end", total_frames)),
                    "nativeFps": item.get("nativeFps", item.get("native_fps")),
                }
            )
    else:
        video_file = str(body.get("videoFile") or body.get("video_file") or "").strip()
        if not video_file:
            return web.Response(status=400, text="Missing clips[] or videoFile.")
        clips.append(
            {
                "videoFile": video_file,
                "fileName": os.path.basename(video_file),
                "subfolder": str(body.get("subfolder") or "").strip(),
                "type": str(body.get("type") or "input").strip() or "input",
                "logicalStart": 0,
                "logicalEnd": total_frames,
                "nativeFps": body.get("nativeFps", body.get("native_fps")),
            }
        )

    if total_frames <= 0:
        return web.Response(status=400, text="totalFrames must be > 0.")

    try:
        result = detect_timeline_shot_cuts(
            clips,
            frame_rate=frame_rate,
            total_frames=total_frames,
            sensitivity=sensitivity,
            min_shot_frames=min_shot_frames,
        )
    except ImportError as exc:
        return web.Response(status=400, text=str(exc))
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt shot detect failed: %s", exc)
        return web.Response(status=400, text=str(exc))

    return web.json_response(result)


async def minimax_clear_cache(request):
    """Clear the per-node cache for the given Director node.

    The Director now stores every cache kind (text/image/video conditioning,
    batch scratch intermediates, and durable segment frames / latents / audio /
    clips) in ONE flat directory: ``minimax_director_opt_cache/<workflow_slug>/node_<id>/``.

    Default clears only the transient data in that dir: text/image/video
    conditioning files and the per-run batch scratch intermediates
    (``seg_*_scratch_*.pt``), plus any ``*_frames_ht.pt`` left by runs from
    before the seam window became a clip — those are superseded dead weight,
    not part of the render. The durable rendered segments are kept.

    With ``clear_all=true`` it additionally wipes every durable ``seg_*`` file —
    every rendered frame / AV latent / audio / clip — forcing a full re-render on
    the next run.

    ``workflow_name`` is resolved to the same slug the cache layer uses, so the
    button always hits exactly the directory that holds this workflow's data.
    """
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    from .conditioning_cache import clear_conditioning_cache
    from . import cache_layout, segment_slots

    workflow_name = str(body.get("workflow_name") or "").strip() or None
    clear_all = bool(body.get("clear_all"))
    cleared = {"conditioning": 0, "batch": 0, "headtail": 0, "segments": 0}

    cache_dir = cache_layout.node_cache_dir(node_id, workflow_name, create=False)

    # 1) text/image/video conditioning files (always cleared)
    try:
        cleared["conditioning"] = await asyncio.to_thread(
            clear_conditioning_cache,
            node_id=node_id,
            workflow_name=workflow_name,
        )
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt clear conditioning cache failed: %s", exc)

    # 2) per-run batch scratch intermediates (always cleared via _scratch_ marker)
    if cache_dir.is_dir():
        for path in cache_layout.iter_scratch_files(cache_dir):
            try:
                path.unlink()
                cleared["batch"] += 1
            except OSError as exc:
                log.warning("MiniMax H3 Director Opt clear scratch %s failed: %s", path, exc)

    # 3) Legacy head/tail tensors (``*_frames_ht.pt``), on *every* clear.
    #    The seam window is a clip now; these are the pre-mp4 copies, tens of MB
    #    each, that no longer get rewritten because nothing re-renders their
    #    segment. Unlike the durable artefacts below they are safe to drop
    #    without forcing a re-render — see cache_layout.iter_legacy_headtail_files.
    #    Skipped under clear_all, which removes them along with everything else.
    if cache_dir.is_dir() and not clear_all:
        for path in cache_layout.iter_legacy_headtail_files(cache_dir):
            try:
                path.unlink()
                cleared["headtail"] += 1
            except OSError as exc:
                log.warning(
                    "MiniMax H3 Director Opt clear legacy head/tail %s failed: %s", path, exc
                )

    # 4) clear_all → also wipe durable segment files (both passes) so the next
    #    run must re-render. ``seg2_*`` is the「二级采样」cache family.
    if clear_all and cache_dir.is_dir():
        for glob in cache_layout.SEGMENT_GLOBS:
            for path in list(cache_dir.glob(glob)):
                if not path.is_file():
                    continue
                try:
                    path.unlink()
                    cleared["segments"] += 1
                except OSError as exc:
                    log.warning("MiniMax H3 Director Opt clear segment %s failed: %s", path, exc)
        # The slot maps name those files; drop them too so the next run rebuilds
        # the position → files mapping from scratch.
        segment_slots.clear_slots(cache_dir)
        segment_slots.clear_slots(cache_dir, variant=segment_slots.VARIANT_SECOND)

    log.info(
        "MiniMax H3 Director Opt cleared caches for node %s (workflow '%s', clear_all=%s): %s",
        node_id, workflow_name or "", clear_all, cleared,
    )
    return web.json_response({"cleared": cleared})


async def minimax_segment_export_status(request):
    """Availability of every segment for「分段导出」(what the picker greys out)."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    try:
        from .plan import build_director_plan, normalize_segment_export_source
        from .segment_cache import inspect_segment_export_status, sync_segment_slots
        from .segment_slots import VARIANT_SECOND

        workflow_name = str(body.get("workflow_name") or "").strip() or None
        # Which pass the picker is showing: the availability probe must answer
        # for that pass only, so switching to「二采」greys out segments that have
        # no ``seg2_*`` cache instead of reporting the first-pass render.
        source = normalize_segment_export_source(body.get("source") or body.get("cacheSource"))
        variant = VARIANT_SECOND if source == "2nd" else "1st"

        plan = build_director_plan(
            str(timeline_data),
            global_task_type=str(body.get("task_type") or ""),
            global_prompt=str(body.get("global_prompt") or ""),
            total_frames=int(body.get("total_frames") or 124),
            frame_rate=float(body.get("frame_rate") or 24.0),
            width=int(body.get("width") or 864),
            height=int(body.get("height") or 480),
            ref_max_size=int(body.get("ref_max_size") or 864),
        )
        # Reconcile first: the picker must not offer a render that belongs to a
        # group deleted from the middle of the timeline.
        sync_segment_slots(node_id, plan, workflow_name=workflow_name, variant=variant)
        return web.json_response(
            inspect_segment_export_status(node_id, plan, workflow_name=workflow_name, variant=variant)
        )
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt segment-export status failed: %s", exc)
        return web.json_response({"segments": [], "error": str(exc)}, status=400)


async def minimax_second_sample_status(request):
    """Availability of every segment for「二次采样」(what the picker greys out).

    A segment is second-sampleable only when its **first-pass latent** is cached
    AND the node holds a **text encoding cache** — mirroring how「分段导出」
    probes the same slot map for clip / latent, but for the second-pass source.
    """
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    try:
        from .plan import build_director_plan
        from .segment_cache import (
            inspect_second_sample_status,
            sync_second_segment_slots,
            sync_segment_slots,
        )

        workflow_name = str(body.get("workflow_name") or "").strip() or None

        plan = build_director_plan(
            str(timeline_data),
            global_task_type=str(body.get("task_type") or ""),
            global_prompt=str(body.get("global_prompt") or ""),
            total_frames=int(body.get("total_frames") or 124),
            frame_rate=float(body.get("frame_rate") or 24.0),
            width=int(body.get("width") or 864),
            height=int(body.get("height") or 480),
            ref_max_size=int(body.get("ref_max_size") or 864),
        )
        # Reconcile both passes first so the picker never offers a segment whose
        # position belongs to a group deleted from the middle of the timeline.
        sync_segment_slots(node_id, plan, workflow_name=workflow_name)
        sync_second_segment_slots(node_id, plan, workflow_name=workflow_name)
        return web.json_response(
            inspect_second_sample_status(node_id, plan, workflow_name=workflow_name)
        )
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt second-sample status failed: %s", exc)
        return web.json_response({"segments": [], "error": str(exc)}, status=400)


async def minimax_align_to_next_status(request):
    """Which segments may enable「对齐下段」(i.e. the next one holds an AV latent).

    The frontend cannot derive this itself: under「选择运行」``plan.index`` is
    the compact run order, so "the next segment" is not simply ``i + 1`` in the
    card list. The plan is rebuilt here (same payload as the run) so the
    backend's index semantics are the single source of truth.
    """
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    try:
        from .plan import build_director_plan
        from .segment_cache import has_next_segment_av_latent, sync_segment_slots
        from .segment_continuity import is_continuity_active

        workflow_name = str(body.get("workflow_name") or "").strip() or None

        plan = build_director_plan(
            str(timeline_data),
            global_task_type=str(body.get("task_type") or ""),
            global_prompt=str(body.get("global_prompt") or ""),
            total_frames=int(body.get("total_frames") or 124),
            frame_rate=float(body.get("frame_rate") or 24.0),
            width=int(body.get("width") or 864),
            height=int(body.get("height") or 480),
            ref_max_size=int(body.get("ref_max_size") or 864),
        )
        segments = list(getattr(plan, "segments", None) or [])
        if not segments:
            # Selection-run is ON but nothing is ticked. Align-to-next availability
            # only depends on the next segment's cached AV latent, not on the
            # current run selection, so report status for *all* segments instead of
            # erroring out (which would grey every「对齐下段」control).
            _tl = json.loads(timeline_data) if timeline_data else {}
            for _k in ("runSelection", "run_selection", "runSelectEnabled", "run_select_enabled"):
                _tl.pop(_k, None)
            plan = build_director_plan(
                json.dumps(_tl),
                global_task_type=str(body.get("task_type") or ""),
                global_prompt=str(body.get("global_prompt") or ""),
                total_frames=int(body.get("total_frames") or 124),
                frame_rate=float(body.get("frame_rate") or 24.0),
                width=int(body.get("width") or 864),
                height=int(body.get("height") or 480),
                ref_max_size=int(body.get("ref_max_size") or 864),
            )
            segments = list(getattr(plan, "segments", None) or [])
        # The next segment's latent must be the neighbour's own render, so the
        # slot map is reconciled before anything is probed.
        sync_segment_slots(node_id, plan, workflow_name=workflow_name)
        rows = []
        for seg in segments:
            rows.append(
                {
                    "index": int(seg.index),
                    # Master「段间引导」must be on for the pin to mean anything.
                    "continuity": bool(is_continuity_active(plan, seg)),
                    "canAlignToNext": bool(has_next_segment_av_latent(node_id, seg.index, workflow_name=workflow_name)),
                }
            )
        return web.json_response({"node_id": node_id, "segments": rows})
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt align-to-next status failed: %s", exc)
        return web.json_response({"segments": [], "error": str(exc)}, status=400)


async def minimax_remove_segment_slot(request):
    """Drop the cached artefacts of one timeline position (UI delete).

    The UI calls this the moment a group is removed so the deleted group's
    render cannot be inherited by the group that slides into its place.
    """
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")
    try:
        index = int(body.get("index"))
    except (TypeError, ValueError):
        return web.Response(status=400, text="Invalid segment index.")

    try:
        from .segment_cache import remove_segment_slot

        removed = await asyncio.to_thread(
            remove_segment_slot,
            node_id,
            index,
            workflow_name=str(body.get("workflow_name") or "").strip() or None,
        )
        return web.json_response({"removed": bool(removed)})
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt segment cache drop failed: %s", exc)
        return web.json_response({"removed": False, "error": str(exc)}, status=400)


async def minimax_segment_export(request):
    """Run a「分段导出」request against the cached segments.

    Body mirrors the timeline block: ``enabled``, ``mode``
    (``piecewise`` | ``continuous``), ``indices``. Each checked segment must have
    an exportable source — its clip cache, its frame cache, or (only when the
    caller supplies models) its latent — otherwise it is skipped and reported.
    """
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    mode = str(body.get("mode") or "").lower()
    raw_indices = body.get("indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        return web.Response(status=400, text="No segment indices selected.")

    try:
        from .plan import build_director_plan, normalize_segment_export_mode
        from .plan import normalize_segment_export_source
        from .segment_cache import run_segment_export, sync_segment_slots
        from .segment_slots import VARIANT_SECOND

        mode = normalize_segment_export_mode(mode)
        source = normalize_segment_export_source(body.get("source") or body.get("cacheSource"))
        variant = VARIANT_SECOND if source == "2nd" else "1st"
        try:
            indices = [int(i) for i in raw_indices]
        except (TypeError, ValueError):
            return web.Response(status=400, text="Invalid segment indices.")

        plan = build_director_plan(
            str(timeline_data),
            global_task_type=str(body.get("task_type") or ""),
            global_prompt=str(body.get("global_prompt") or ""),
            total_frames=int(body.get("total_frames") or 124),
            frame_rate=float(body.get("frame_rate") or 24.0),
            width=int(body.get("width") or 864),
            height=int(body.get("height") or 480),
            ref_max_size=int(body.get("ref_max_size") or 864),
        )
        workflow_name = str(body.get("workflow_name") or "").strip() or None
        # Reconcile before exporting: never copy out a file group that belongs
        # to a group already deleted from the timeline.
        sync_segment_slots(node_id, plan, workflow_name=workflow_name, variant=variant)
        # Disk-only export: clip cache / raw frame cache. Latent-only segments
        # cannot be decoded here (no VAE in an HTTP request), so they are skipped
        # with a hint — the full latent decode happens during node execution when
        # the VAE is loaded.
        result = await asyncio.to_thread(
            run_segment_export,
            node_id,
            plan,
            indices,
            mode=mode,
            workflow_name=workflow_name,
            variant=variant,
        )
        return web.json_response(result)
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt segment-export failed: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


async def minimax_segment_clip(request):
    """Stream a cached segment clip so the UI「二采」tab can play it in place.

    Read-only: resolves the same slot map the export status uses, so it always
    follows the segment currently living at ``index`` (and its superseded group
    when the newest stem is still empty). Never writes anything.
    """
    node_id = str(request.query.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")
    try:
        index = int(request.query.get("index") or 0)
    except Exception:
        return web.Response(status=400, text="Invalid segment index.")
    if index < 0:
        return web.Response(status=400, text="Invalid segment index.")

    variant = str(request.query.get("variant") or "").strip().lower()
    workflow_name = str(request.query.get("workflow_name") or "").strip() or None

    try:
        from .segment_cache import clip_cache_path
        from .segment_slots import VARIANT_FIRST, VARIANT_SECOND
    except Exception as exc:  # pragma: no cover - import guard
        log.warning("MiniMax H3 Director Opt segment-clip import failed: %s", exc)
        return web.Response(status=500, text="Segment cache unavailable.")

    variant_key = VARIANT_SECOND if variant in ("2nd", "second", "2") else VARIANT_FIRST
    try:
        path = clip_cache_path(
            node_id, index, workflow_name=workflow_name,
            allow_prev=True, variant=variant_key,
        )
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt segment-clip failed: %s", exc)
        return web.Response(status=404, text="Segment clip not cached.")
    if path is None:
        return web.Response(status=404, text="Segment clip not cached.")
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return web.Response(status=404, text="Segment clip not cached.")
    except OSError:
        return web.Response(status=404, text="Segment clip not cached.")

    return web.FileResponse(
        str(path),
        headers={"Cache-Control": "no-store"},
    )


def _register_route(routes, method: str, path: str, handler) -> None:
    if hasattr(routes, "add_route"):
        # aiohttp UrlDispatcher: registers exactly this method (no implicit HEAD).
        routes.add_route(method, path, handler)
    elif method == "POST" and hasattr(routes, "post"):
        routes.post(path)(handler)
    elif method == "GET" and hasattr(routes, "get"):
        routes.get(path)(handler)
    else:
        raise AttributeError("Unsupported ComfyUI route table API")


def register_routes() -> bool:
    """Register MiniMax H3 Director Opt HTTP routes on the ComfyUI PromptServer."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True

    server = PromptServer.instance
    if server is None:
        log.warning("MiniMax H3 Director Opt: PromptServer not ready, HTTP routes not registered")
        return False

    routes = server.routes
    _register_route(routes, "POST", "/minimax/director_opt/upload_chunk", minimax_upload_video_chunk)
    _register_route(
        routes,
        "POST",
        "/minimax/director_opt/extract_reference_audio",
        minimax_extract_reference_audio,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director_opt/prepare_reference_audio_chunk",
        minimax_prepare_reference_audio_chunk,
    )
    _register_route(routes, "POST", "/minimax/director_opt/probe_video", minimax_probe_video)
    _register_route(routes, "GET", "/minimax/director_opt/probe_video", minimax_probe_video)
    _register_route(routes, "GET", "/minimax/director_opt/list_input_media", minimax_list_input_media)
    _register_route(routes, "POST", "/minimax/director_opt/clear_cache", minimax_clear_cache)
    _register_route(routes, "POST", "/minimax/director_opt/detect_shots", minimax_detect_shots)
    _register_route(
        routes,
        "POST",
        "/minimax/director_opt/segment_export_status",
        minimax_segment_export_status,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director_opt/second_sample_status",
        minimax_second_sample_status,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director_opt/align_to_next_status",
        minimax_align_to_next_status,
    )
    # HEAD 无需注册：RouteTableDef.get() 走 UrlDispatcher.add_get()，默认
    # allow_head=True 会自动挂上 HEAD；再显式注册一次会直接 RuntimeError
    # （"Added route will never be executed, method HEAD is already registered"）。
    _register_route(routes, "GET", "/minimax/director_opt/segment_clip", minimax_segment_clip)
    _register_route(
        routes,
        "POST",
        "/minimax/director_opt/segment_export",
        minimax_segment_export,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director_opt/remove_segment_slot",
        minimax_remove_segment_slot,
    )
    from .pack import minimax_download_pack, minimax_export_pack, minimax_import_pack

    _register_route(routes, "POST", "/minimax/director_opt/export_pack", minimax_export_pack)
    _register_route(routes, "GET", "/minimax/director_opt/download_pack", minimax_download_pack)
    _register_route(routes, "POST", "/minimax/director_opt/import_pack", minimax_import_pack)
    _ROUTES_REGISTERED = True
    log.info("MiniMax H3 Director Opt HTTP routes registered")
    return True
