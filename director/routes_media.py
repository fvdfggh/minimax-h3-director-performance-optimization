"""Media discovery routes: probe, list input media, detect shots.

Everything the front end calls *before* a run to find out what it has: the
director clips already on disk, the media available in ComfyUI's input directory,
a single video's geometry, and the shot boundaries used to seed groups.

Relative imports *inside* the handlers resolve against :mod:`director` exactly as
they did in :mod:`http_routes`, because this module sits in the same package.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import folder_paths
from aiohttp import web

from ..lib.constants import ROUTE_PREFIX
from ..lib.pathutil import AUDIO_EXTS, IMAGE_EXTS, VIDEO_EXTS, posix_relpath

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.routes.media")


def register(routes, register_route) -> None:
    """Attach this group's routes; called by :func:`http_routes.register_routes`."""
    register_route(routes, "POST", f"{ROUTE_PREFIX}/probe_video", minimax_probe_video)
    register_route(routes, "GET", f"{ROUTE_PREFIX}/probe_video", minimax_probe_video)
    register_route(routes, "GET", f"{ROUTE_PREFIX}/list_input_media", minimax_list_input_media)
    register_route(routes, "POST", f"{ROUTE_PREFIX}/detect_shots", minimax_detect_shots)


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
            rel_dir = posix_relpath(dirpath, cache_root)
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
                rel_path = posix_relpath(abs_path, input_dir)
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
