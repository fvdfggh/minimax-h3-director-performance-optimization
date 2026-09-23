"""Cross-layer literals for the plugin: HTTP contract, frontend events, defaults.

Everything here is an **external contract** — already-saved workflows, the
browser frontend (``web/js/*.js``, which repeats the route prefix by necessity)
and on-disk caches depend on the exact values. They are defined once so that a
change is a single edit instead of a search-and-replace across a dozen files.

Layering note: this module lives in ``lib`` (the leaf layer) because ``lib``,
``director`` and ``nodes`` all need the canvas defaults; nothing in ``lib`` may
import from ``director``.
"""

from __future__ import annotations

# --- HTTP contract -----------------------------------------------------------
#: Prefix of every route registered by :func:`director.http_routes.register_routes`.
ROUTE_PREFIX = "/minimax/director_opt"

# --- PromptServer websocket events (consumed by web/js/minimax_timeline.js) ---
EVENT_PROGRESS = "minimax_director_opt_progress"
EVENT_PREVIEW = "minimax_director_opt_preview"

# --- Default canvas / timeline (0.4 MP 16:9, ~5 s at 24 fps) ------------------
DEFAULT_FRAME_RATE = 24.0
DEFAULT_WIDTH = 864
DEFAULT_HEIGHT = 480
DEFAULT_REF_MAX_SIZE = 864
DEFAULT_TOTAL_FRAMES = 124

#: Fallback long edge used when a timeline carries no size at all. Distinct from
#: ``DEFAULT_REF_MAX_SIZE``: 848 is the largest long edge already on the H3 patch
#: grid, so it is used as the "last resort" canvas rather than the node default.
FALLBACK_LONG_EDGE = 848
