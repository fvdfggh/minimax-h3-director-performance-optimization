"""HTTP route registration for MiniMax H3 Director Opt.

This module owns only the *wiring*: the pack routes (their handlers live with the
pack logic in :mod:`pack`) and :func:`register_routes`, which hands the route
table to each group's ``register``:

* :mod:`routes_upload`   — chunked upload + reference audio
* :mod:`routes_media`    — probe / list input media / detect shots
* :mod:`routes_segments` — segment cache status, export, clip streaming

Each group keeps its handlers and its registration together, which is what makes
"handler written but never registered" impossible — the failure mode that left the
prompt-enhance routes unreachable before this cleanup.

Called once from ``__init__.py``; idempotent.
"""

from __future__ import annotations

import logging

from server import PromptServer

from ..lib.constants import ROUTE_PREFIX
from . import routes_media, routes_segments, routes_upload
from .routes_common import _register_route

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director")

_ROUTES_REGISTERED = False


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
    routes_upload.register(routes, _register_route)
    routes_media.register(routes, _register_route)
    routes_segments.register(routes, _register_route)

    # Pack routes: handlers live next to the pack logic.
    from .pack import minimax_download_pack, minimax_export_pack, minimax_import_pack

    _register_route(routes, "POST", f"{ROUTE_PREFIX}/export_pack", minimax_export_pack)
    _register_route(routes, "GET", f"{ROUTE_PREFIX}/download_pack", minimax_download_pack)
    _register_route(routes, "POST", f"{ROUTE_PREFIX}/import_pack", minimax_import_pack)
    _ROUTES_REGISTERED = True
    log.info("MiniMax H3 Director Opt HTTP routes registered")
    return True
