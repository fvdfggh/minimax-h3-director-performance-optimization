"""Pack IO plumbing: export scratch dirs, zip streaming, media path resolution.

Shared by :mod:`pack_export` and :mod:`pack_import`. Nothing here knows about the
timeline structure — that is :mod:`pack_rewrite` — and nothing here writes a pack:
these are the primitives both directions need.

The streamed-zip helpers exist because a pack can be far larger than memory:
``_send_zip_file`` copies the archive out in :data:`pack_format.ZIP_STREAM_CHUNK`
slices, and ``_zip_headers`` keeps non-ASCII names working on every browser.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import folder_paths
from aiohttp import web

from ..lib.pathutil import MEDIA_EXTS, SAFE_EXT_RE
from .pack_format import ASCII_PATH_RE, PACK_EXPORT_TTL_SEC, ZIP_STREAM_CHUNK


def _pack_export_root() -> Path:
    root = Path(folder_paths.get_temp_directory()) / "minimax_director_opt_pack_export"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _purge_pack_exports(*, keep: str | None = None) -> None:
    """Remove export zips older than TTL. Never delete a file that may still be downloading."""
    root = _pack_export_root()
    now = time.time()
    for path in root.glob("*.mmxpack.zip"):
        if keep and path.name == keep:
            continue
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if age >= PACK_EXPORT_TTL_SEC:
            _unlink_quiet(path)


def _is_under_dir(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _zip_headers(download_name: str, size: int, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/zip",
        "Content-Disposition": f'attachment; filename="{download_name}"',
        "Content-Length": str(int(size)),
        "Cache-Control": "no-store",
    }
    if extra:
        headers.update(extra)
    return headers


async def _send_zip_file(
    request,
    path: Path,
    download_name: str,
    extra_headers: dict[str, str] | None = None,
    *,
    unlink_after: bool = False,
):
    """Stream a zip without aiohttp FileResponse.

    FileResponse/sendfile on Windows often yields HTTP 200 with an empty body,
    which browsers save as a 0-byte .zip that unzip tools reject.
    """
    try:
        size = int(path.stat().st_size)
    except OSError:
        return web.Response(status=404, text="Pack not found.")
    if size <= 0:
        return web.Response(status=404, text="Pack not found.")
    headers = _zip_headers(download_name, size, extra_headers)
    resp = web.StreamResponse(status=200, headers=headers)
    await resp.prepare(request)
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(ZIP_STREAM_CHUNK)
                if not chunk:
                    break
                await resp.write(chunk)
        await resp.write_eof()
        return resp
    finally:
        if unlink_after:
            _unlink_quiet(path)


def _input_dir() -> Path:
    return Path(folder_paths.get_input_directory())


def _is_ascii_pack_path(rel: str) -> bool:
    s = str(rel or "").replace("\\", "/").strip()
    if not s or s.startswith("/") or ".." in s.split("/"):
        return False
    return bool(ASCII_PATH_RE.fullmatch(s))


def _safe_ext(path: Path | str, fallback: str = ".bin") -> str:
    ext = Path(str(path)).suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    if not SAFE_EXT_RE.fullmatch(ext) or ext not in MEDIA_EXTS | {".json"}:
        return fallback
    return ext


def _posix(rel: Path | str) -> str:
    return str(rel).replace("\\", "/").lstrip("/")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _media_bases() -> list[tuple[str, Path]]:
    bases: list[tuple[str, Path]] = [("input", _input_dir())]
    try:
        bases.append(("output", Path(folder_paths.get_output_directory())))
    except Exception:
        pass
    try:
        bases.append(("temp", Path(folder_paths.get_temp_directory())))
    except Exception:
        pass
    return bases


def resolve_media_path(
    rel: str,
    *,
    subfolder: str = "",
    type_name: str = "input",
) -> Path | None:
    raw = str(rel or "").replace("\\", "/").strip()
    if not raw:
        return None
    sub = str(subfolder or "").replace("\\", "/").strip().strip("/")
    if sub and "/" not in raw and not raw.startswith(sub + "/"):
        raw = f"{sub}/{raw}"
    type_name = str(type_name or "input").strip() or "input"
    candidates: list[Path] = []
    for kind, base in _media_bases():
        if kind == type_name or not type_name:
            candidates.append(base / raw.replace("/", os.sep))
    for _kind, base in _media_bases():
        p = base / raw.replace("/", os.sep)
        if p not in candidates:
            candidates.append(p)
        candidates.append(base / Path(raw).name)
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None
