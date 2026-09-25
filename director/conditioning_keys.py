"""Keys, hashes and the cache directory for the conditioning cache.

Every cached encoding is addressed by a key derived from what was actually
encoded — the (node-modified) prompt, the visual inputs, the workflow name —
never by timeline position. A position-keyed file hands a segment its neighbour's
conditioning after a reorder *and* still looks valid, because the file it names
does exist; failing silently is exactly what this addressing avoids.

``_get_cache_dir`` lives here too: everything else resolves its directory through
it, so the "one cache root per workflow" rule is stated once.

``retime_conditioning_for_frames`` is the only transform the second pass applies to
a loaded encoding, so it sits next to the keys it has to stay consistent with.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch

from . import cache_layout

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.conditioning.keys")


def slugify_workflow_name(name: str | None) -> str:
    """Turn a workflow name into a single safe path component.

    Thin re-export of the shared implementation in :mod:`cache_layout`, kept
    because several modules import it from here.
    """
    return cache_layout.slugify_workflow_name(name)


def _get_cache_dir(node_id: str | None = None, workflow_name: str | None = None) -> Path:
    """Get or create the cache directory shared by every artefact kind.

    Layout is ``<cache root>/[<workflow slug>/]node_<id>/``. The workflow layer
    keeps caches from differently named workflows from colliding on the same
    ``node_<id>`` — node ids are per-graph and routinely reused across files.
    """
    return cache_layout.node_cache_dir(node_id, workflow_name)


def _hash_ref_images(ref_images: Any) -> str:
    """Generate a hash for reference images to include in cache key.
    
    This ensures cache invalidation when reference images change.
    """
    if ref_images is None:
        return "none"
    
    try:
        # Handle different ref_images formats
        if isinstance(ref_images, dict):
            # Dict format: {"ref_image_0": tensor, ...}
            hash_parts = []
            for key in sorted(ref_images.keys()):
                tensor = ref_images[key]
                if isinstance(tensor, torch.Tensor):
                    # Hash the tensor data
                    tensor_bytes = tensor.cpu().numpy().tobytes()
                    hash_parts.append(f"{key}:{hashlib.md5(tensor_bytes).hexdigest()[:8]}")
                else:
                    hash_parts.append(f"{key}:{hash(str(tensor))}")
            return "|".join(hash_parts) if hash_parts else "empty_dict"
        
        elif isinstance(ref_images, (list, tuple)):
            # List/tuple format
            hash_parts = []
            for i, item in enumerate(ref_images):
                if isinstance(item, torch.Tensor):
                    tensor_bytes = item.cpu().numpy().tobytes()
                    hash_parts.append(f"img_{i}:{hashlib.md5(tensor_bytes).hexdigest()[:8]}")
                elif isinstance(item, dict):
                    # Nested dict
                    for key in sorted(item.keys()):
                        tensor = item[key]
                        if isinstance(tensor, torch.Tensor):
                            tensor_bytes = tensor.cpu().numpy().tobytes()
                            hash_parts.append(f"{key}:{hashlib.md5(tensor_bytes).hexdigest()[:8]}")
            return "|".join(hash_parts) if hash_parts else "empty_list"
        
        else:
            return f"other:{hash(str(ref_images))}"
    
    except Exception as exc:
        log.warning("Failed to hash ref_images: %s", exc)
        return f"error:{hash(str(ref_images))}"


def text_cache_key(
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_key: str,
    ref_image_size: str = "match",
    ref_images: Any = None,
    *,
    ref_videos: Any = None,
    first_frame: Any = None,
    last_frame: Any = None,
) -> str:
    """Public cache key for a text encoding, shared by save/load/dedupe.

    Exposed so the batch path can group segments that would produce byte-identical
    text encodings and encode each distinct key exactly once.

    ``length`` deliberately takes no part in the key (see :func:`_prompt_hash`).
    """
    return _prompt_hash(
        prompt, width, height, length, task_key, ref_image_size, ref_images,
        ref_videos=ref_videos, first_frame=first_frame, last_frame=last_frame,
    )


def _canvas_bound(
    ref_images: Any = None,
    ref_videos: Any = None,
    first_frame: Any = None,
    last_frame: Any = None,
) -> bool:
    """True when this encoding cannot be reused at another resolution.

    Only the i2v / fl2v **first / last frames** bind it: they are the generated
    video's own opening and closing frame, so they are stretched and centre-cropped
    onto the canvas, and Qwen sees them at exactly that size.

    Nothing else does. Reference images and reference videos both reach Qwen at
    native resolution, and the latents they produce live in
    :mod:`ref_latent_cache` under an address the *reader* re-derives from its own
    canvas — so the same file serves 768p and 1080p alike, each resolving to its
    own latents. Pure-prompt batches (t2v, or reference audio only) are
    canvas-free as well.

    ``ref_images`` / ``ref_videos`` are accepted but deliberately unused: they are
    what used to bind the encoding to the canvas, and keeping the parameters makes
    the change (and the call sites still passing them) readable.
    """
    return first_frame is not None or last_frame is not None


def _prompt_hash(
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_key: str,
    ref_image_size: str = "match",
    ref_images: Any = None,
    *,
    ref_videos: Any = None,
    first_frame: Any = None,
    last_frame: Any = None,
) -> str:
    """Hash key for a text encoding.

    Including ref_images hash ensures cache invalidation when reference images change.

    Two invariants the rest of this module relies on:

    * **``length`` is NOT part of the key.** The sample length only reaches the
      *extras* that ride along in the payload (``minimax_frame_count`` and the
      keyframe anchors), never the token stream: changing a segment's duration
      therefore reuses the encoding instead of paying for another Qwen prefill.
      :func:`retime_conditioning_for_frames` re-maps those extras on load.
    * **The canvas only counts when the encoding is canvas-bound** — see
      :func:`_canvas_bound`. Only i2v / fl2v keyframes are: they *are* frames of
      the output, so they are fitted to the canvas before Qwen sees them.
      Reference images and videos are not — Qwen gets them native, and their
      latents are resolved per-canvas at load time — so a resolution change keeps
      reusing the encoding.

    Keeping both correct is what lets one file serve every duration of a prompt
    instead of one file per (prompt, resolution, duration) triple.
    """
    ref_hash = _hash_ref_images(ref_images)
    if _canvas_bound(ref_images, ref_videos, first_frame, last_frame):
        canvas = f"{int(width)}|{int(height)}"
    else:
        canvas = "canvas-free"
    content = f"{prompt}|{canvas}|{task_key}|{ref_image_size}|{ref_hash}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def retime_conditioning_for_frames(positive: Any, old_frames: int | None, new_frames: int | None) -> Any:
    """Move a cached encoding's length-dependent extras onto another length.

    Only two things in the payload know about the sample length:
    ``minimax_frame_count`` and the **last** keyframe anchor
    (``resolved_frame_index == old_frames - 1``). Anything anchored at 0 (the
    customary first frame) is already correct. Without this, reusing an encoding
    across durations would leave the DiT anchoring the tail keyframe to a frame
    count the sample no longer has.
    """
    try:
        old_n = int(old_frames or 0)
        new_n = int(new_frames or 0)
    except (TypeError, ValueError):
        return positive
    if old_n <= 0 or new_n <= 0 or old_n == new_n or not positive:
        return positive

    def _move(obj):
        if isinstance(obj, (list, tuple)) and len(obj) >= 2 and isinstance(obj[1], dict):
            cfg = dict(obj[1])
            kfs = cfg.get("minimax_keyframes")
            if isinstance(kfs, (list, tuple)) and kfs:
                moved = []
                for kf in kfs:
                    if isinstance(kf, dict):
                        kf = dict(kf)
                        try:
                            rfi = int(kf.get("resolved_frame_index", -1))
                        except (TypeError, ValueError):
                            rfi = -1
                        if rfi == old_n - 1:
                            kf["resolved_frame_index"] = new_n - 1
                    moved.append(kf)
                cfg["minimax_keyframes"] = moved
            if "minimax_frame_count" in cfg:
                cfg["minimax_frame_count"] = new_n
            return [obj[0], cfg]
        return obj

    rows = [_move(row) for row in positive]
    log.info(
        "Conditioning cache re-timed for %d → %d frames (keyframe anchors moved).",
        old_n, new_n,
    )
    return rows
