"""Conditioning cache facade.

CLIP-encoded tensors are cached so the second pass (「二次采样」) can re-use the
first pass's encoding instead of re-running the Qwen prefill. The cache is split by
concern; this module re-exports what the rest of the plugin imports:

* :mod:`conditioning_keys`   — key / hash addressing and the cache directory
* :mod:`conditioning_store`  — saving and loading the encoded tensors
* :mod:`conditioning_params` — the per-segment params map the second pass needs
* :mod:`conditioning_admin`  — clearing, pruning and stats

Two design points worth keeping in view:

* **One cache root per workflow**, shared with the segment caches and batch
  scratch, so a workflow's whole state lives in a single folder;
* the cache file name carries **no segment index on purpose** — two segments with
  identical text inputs must resolve to the same file so the second one is served
  from disk instead of being encoded again. Text is the only encoding kind cached
  here; reference image / video encoding lives in ``_vit/`` (see
  :mod:`vision_cache`).
"""

from .conditioning_admin import (
    check_clear_edge,
    clear_all_conditioning_cache,
    clear_conditioning_cache,
    get_all_cache_stats,
    get_cache_stats,
    mark_clear_state,
    prune_unused_conditioning_cache,
)
from .conditioning_keys import (
    retime_conditioning_for_frames,
    slugify_workflow_name,
    text_cache_key,
)
from .conditioning_params import load_segment_second_params, save_segment_second_params
from .conditioning_store import (
    load_conditioning_by_key,
    load_conditioning_cache,
    save_conditioning_cache,
)

__all__ = [
    # store
    "save_conditioning_cache",
    "load_conditioning_cache",
    "load_conditioning_by_key",
    # keys
    "text_cache_key",
    "slugify_workflow_name",
    "retime_conditioning_for_frames",
    # params map
    "save_segment_second_params",
    "load_segment_second_params",
    # admin
    "clear_conditioning_cache",
    "clear_all_conditioning_cache",
    "check_clear_edge",
    "mark_clear_state",
    "get_cache_stats",
    "get_all_cache_stats",
    "prune_unused_conditioning_cache",
]
