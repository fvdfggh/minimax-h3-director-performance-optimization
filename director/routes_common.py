"""Shared pieces for the Director Opt HTTP route modules.

``_register_route`` is the one place that knows how to attach a handler to
ComfyUI's route table (aiohttp ``UrlDispatcher`` when available, otherwise the
``post`` / ``get`` decorators). ``_safe_basename`` and ``_plan_from_request`` are
used by more than one route group, so they live here rather than in any one group.

Route groups, and the paths each owns (all relative to
:data:`lib.constants.ROUTE_PREFIX`):

* :mod:`routes_upload`   — ``/upload_chunk``, ``/extract_reference_audio``,
  ``/prepare_reference_audio_chunk``
* :mod:`routes_media`    — ``/probe_video``, ``/list_input_media``,
  ``/detect_shots``
* :mod:`routes_segments` — ``/clear_cache``, ``/segment_export_status``,
  ``/second_sample_status``, ``/align_to_next_status``, ``/remove_segment_slot``,
  ``/segment_export``, ``/segment_clip``
* :mod:`http_routes`     — the pack routes, plus ``register_routes`` which calls
  each group's ``register``

Each group owns its own registration next to its handlers, so a handler can no
longer be written without a matching route (the failure mode that left the
prompt-enhance routes unreachable).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..lib.constants import (
    DEFAULT_FRAME_RATE,
    DEFAULT_HEIGHT,
    DEFAULT_REF_MAX_SIZE,
    DEFAULT_TOTAL_FRAMES,
    DEFAULT_WIDTH,
)
from ..lib.pathutil import (
    SAFE_EXT_RE as _SAFE_EXT,
    WIN_ILLEGAL_RE as _WIN_ILLEGAL,
    WIN_RESERVED_RE as _WIN_RESERVED,
)

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from .plan_types import DirectorPlan


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


def _plan_from_request(body: dict, timeline_data: str) -> "DirectorPlan":
    """Build the ``DirectorPlan`` a picker request describes.

    Every picker endpoint (segment export / second sample / align-to-next)
    rebuilds the same plan from the same request fields. Centralising it means a
    new plan input only needs to be added here, next to the node's own defaults —
    the five copies used to drift apart.
    """
    from .plan import build_director_plan

    return build_director_plan(
        str(timeline_data),
        global_task_type=str(body.get("task_type") or ""),
        global_prompt=str(body.get("global_prompt") or ""),
        total_frames=int(body.get("total_frames") or DEFAULT_TOTAL_FRAMES),
        frame_rate=float(body.get("frame_rate") or DEFAULT_FRAME_RATE),
        width=int(body.get("width") or DEFAULT_WIDTH),
        height=int(body.get("height") or DEFAULT_HEIGHT),
        ref_max_size=int(body.get("ref_max_size") or DEFAULT_REF_MAX_SIZE),
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
