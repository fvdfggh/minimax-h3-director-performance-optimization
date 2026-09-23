"""Upload and reference-audio routes.

The two concerns share one file because they share the plumbing: a file arrives in
chunks (``/upload_chunk``, ``/prepare_reference_audio_chunk``), is placed in
ComfyUI's input directory, and is then probed or turned into a reference-audio
waveform (``/extract_reference_audio``).

Relative imports *inside* the handlers resolve against :mod:`director` exactly as
they did in :mod:`http_routes`, because this module sits in the same package.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import uuid

import folder_paths
from aiohttp import web

from ..lib.constants import ROUTE_PREFIX
from ..lib.pathutil import AUDIO_EXTS, VIDEO_EXTS
from .routes_common import _safe_basename

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.routes.upload")


def register(routes, register_route) -> None:
    """Attach this group's routes; called by :func:`http_routes.register_routes`."""
    register_route(routes, "POST", f"{ROUTE_PREFIX}/upload_chunk", minimax_upload_video_chunk)
    register_route(
        routes,
        "POST",
        f"{ROUTE_PREFIX}/extract_reference_audio",
        minimax_extract_reference_audio,
    )
    register_route(
        routes,
        "POST",
        f"{ROUTE_PREFIX}/prepare_reference_audio_chunk",
        minimax_prepare_reference_audio_chunk,
    )


CHUNK_ROOT = os.path.join(folder_paths.get_temp_directory(), "minimax_upload_chunks")


REF_AUDIO_CHUNK_ROOT = os.path.join(folder_paths.get_temp_directory(), "minimax_ref_audio_chunks")


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

    from ..lib.ffmpeg import ffmpeg_bin

    ffmpeg = ffmpeg_bin()
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
