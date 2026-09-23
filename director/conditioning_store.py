"""Saving and loading encoded conditioning tensors.

The payload is an AV latent dict; the file name is the key from
:mod:`conditioning_keys`. Every read tolerates a missing or unreadable file by
returning ``None`` — a cache miss must never break a run, only cost a re-encode.

``av_frame_count`` / ``_empty_av_latent`` describe the payload's shape, which is
why they live with the readers rather than with the keys.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from .cache_layout import TEXT_PREFIX as _TEXT_PREFIX
from .conditioning_keys import (
    _get_cache_dir,
    _has_visual_inputs,
    _prompt_hash,
    retime_conditioning_for_frames,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.conditioning.store")


def av_frame_count(width: int, height: int, length: int) -> int | None:
    """Aligned frame count a canvas of ``(width, height, length)`` samples to.

    Same value ``prepare_segment_materials`` bakes into ``minimax_frame_count``.
    """
    try:
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent as _official
    except Exception as exc:  # pragma: no cover - ComfyUI always ships this
        log.warning("Cannot resolve frame count (%s).", exc)
        return None
    try:
        latent, frame_count = _official(int(width), int(height), int(length))
        del latent
        return int(frame_count)
    except Exception as exc:
        log.warning("Frame count lookup failed for %sx%sx%s (%s).", width, height, length, exc)
        return None


def _empty_av_latent(width: int, height: int, length: int) -> Any:
    """Rebuild the all-zero AV canvas instead of persisting it.

    ``comfy_extras.nodes_minimax_h3._empty_av_latent`` is a pure function of
    ``(width, height, length)`` and always returns ``torch.zeros`` — all three
    inputs are already stored in the cache ``metadata``, so writing the zeros to
    disk buys nothing and only costs a few MB per file plus save/load time.

    Returns ``None`` when the official helper is unavailable, so the caller can
    treat it as a miss and re-encode rather than sampling with no latent.
    """
    try:
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent as _official
    except Exception as exc:  # pragma: no cover - ComfyUI always ships this
        log.warning("Cannot rebuild empty latent (%s).", exc)
        return None
    try:
        latent, _frame_count = _official(int(width), int(height), int(length))
        return latent
    except Exception as exc:
        log.warning("Empty latent rebuild failed for %sx%sx%s (%s).", width, height, length, exc)
        return None


def save_conditioning_cache(
    node_id: str | None,
    segment_index: int,
    positive: Any,
    negative: Any,
    latent: Any,
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_key: str,
    ref_image_size: str = "match",
    ref_images: Any = None,
    workflow_name: str | None = None,
    *,
    ref_videos: Any = None,
    first_frame: Any = None,
    last_frame: Any = None,
    frame_count: int | None = None,
) -> Path | None:
    """Save conditioning tensors to disk cache.

    ``latent`` is accepted for call-site compatibility but is **not** written:
    the initial AV canvas is always all zeros and is a pure function of
    ``(width, height, length)``, all of which land in ``metadata``, so
    :func:`load_conditioning_cache` rebuilds it. Dropping it saves roughly
    6 MB per file and the corresponding save/load time.

    ``frame_count`` is the aligned frame count the encoding was assembled for;
    it is recorded only so a later run at another duration can re-map the
    length-dependent extras (see :func:`retime_conditioning_for_frames`).

    Returns the cache file path, or None if saving failed.
    """
    try:
        cache_dir = _get_cache_dir(node_id, workflow_name)
        prompt_key = _prompt_hash(
            prompt, width, height, length, task_key, ref_image_size, ref_images,
            ref_videos=ref_videos, first_frame=first_frame, last_frame=last_frame,
        )
        cache_file = cache_dir / f"{_TEXT_PREFIX}_{prompt_key}.pt"
        
        # Prepare conditioning for serialization
        # ComfyUI conditioning is a list of [tensor, dict] pairs
        def prepare_for_save(cond):
            if cond is None:
                return None
            if isinstance(cond, (list, tuple)):
                prepared = []
                for item in cond:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        tensor, config = item[0], item[1]
                        # Ensure tensor is on CPU for storage
                        if hasattr(tensor, 'cpu'):
                            tensor = tensor.cpu()
                        prepared.append([tensor, config])
                    else:
                        prepared.append(item)
                return prepared
            return cond
        
        del latent  # zero canvas — rebuilt from metadata on load, never stored
        cache_data = {
            "positive": prepare_for_save(positive),
            "negative": prepare_for_save(negative),
            # Kept as a key (value None) so readers of both old and new files can
            # use ``cache_data["latent"]`` without a KeyError.
            "latent": None,
            "metadata": {
                "segment_index": segment_index,
                "prompt_hash": prompt_key,
                "width": width,
                "height": height,
                "length": length,
                "frame_count": int(frame_count) if frame_count else None,
                # False ⇒ nothing in this payload was derived from the canvas,
                # so the same file may serve any resolution.
                "canvas_bound": _has_visual_inputs(ref_images, ref_videos, first_frame, last_frame),
                "task_key": task_key,
                "ref_image_size": ref_image_size,
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            }
        }
        
        torch.save(cache_data, cache_file)
        log.info("Conditioning cached: seg #%d → %s", segment_index + 1, cache_file.name)
        return cache_file
        
    except Exception as exc:
        log.warning("Failed to cache conditioning for seg #%d: %s", segment_index + 1, exc)
        return None


def load_conditioning_cache(
    node_id: str | None,
    segment_index: int,
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_key: str,
    ref_image_size: str = "match",
    ref_images: Any = None,
    workflow_name: str | None = None,
    *,
    ref_videos: Any = None,
    first_frame: Any = None,
    last_frame: Any = None,
) -> dict | None:
    """Load cached conditioning tensors from disk.

    ``segment_index`` is accepted for call-site compatibility but takes no part in
    the lookup: files are named by text hash alone, so any segment whose text
    inputs hash the same resolves to the same file.

    Returns dict with 'positive', 'negative', 'latent' keys, or None if cache miss.
    Tensors are moved to CUDA (GPU) for use by the model.

    ``latent`` is the initial AV canvas. Files written before it was dropped from
    the payload still carry it and are used as-is; newer files store ``None`` and
    it is rebuilt from the metadata here, so no existing cache needs clearing.

    A **different sample length is not a miss**: the key ignores ``length``, and
    whatever the payload remembers about it is re-timed to the caller's length
    before it leaves this function.
    """
    try:
        cache_dir = _get_cache_dir(node_id, workflow_name)
        prompt_key = _prompt_hash(
            prompt, width, height, length, task_key, ref_image_size, ref_images,
            ref_videos=ref_videos, first_frame=first_frame, last_frame=last_frame,
        )
        cache_file = cache_dir / f"{_TEXT_PREFIX}_{prompt_key}.pt"

        if not cache_file.exists():
            log.debug("No cache hit for seg #%d (key=%s)", segment_index + 1, prompt_key)
            return None
        
        cache_data = torch.load(cache_file, map_location="cpu", weights_only=False)
        
        # Validate cache metadata. The canvas only has to match when the encoding
        # actually looked at pixels (reference media / keyframe images); a canvas
        # -free file is valid at any resolution. The length never has to match.
        meta = cache_data.get("metadata", {})
        canvas_bound = bool(meta.get("canvas_bound", True))
        if canvas_bound and (
            meta.get("width") != width or
            meta.get("height") != height
        ):
            log.info("Cache invalidated for seg #%d (dimensions changed)", segment_index + 1)
            return None
        if meta.get("task_key") != task_key:
            log.info("Cache invalidated for seg #%d (task key changed)", segment_index + 1)
            return None

        # Move tensors to CUDA (GPU) for model use
        def move_to_device(cond):
            """Recursively move all tensors in conditioning to CUDA."""
            if cond is None:
                return None
            if isinstance(cond, torch.Tensor):
                return cond.cuda()
            if isinstance(cond, (list, tuple)):
                return type(cond)(move_to_device(item) for item in cond)
            if isinstance(cond, dict):
                return {k: move_to_device(v) for k, v in cond.items()}
            return cond

        positive = move_to_device(cache_data["positive"])
        negative = move_to_device(cache_data["negative"])
        latent = cache_data.get("latent")
        if latent is not None and meta.get("length") and int(meta.get("length")) != int(length):
            # Legacy file that still carries its own zero canvas — built for another
            # length, so it must not be reused. New-style files store None here.
            latent = None
        # Same text, another duration: move the anchors that encode the old length.
        # Only payloads with keyframes carry them, and resolving the frame count
        # costs an extra canvas allocation, so check before paying for it.
        if any(
            isinstance(row, (list, tuple)) and len(row) >= 2 and isinstance(row[1], dict)
            and "minimax_frame_count" in row[1]
            for row in (positive or [])
        ):
            stored_frames = meta.get("frame_count")
            if not stored_frames and meta.get("length"):
                stored_frames = av_frame_count(
                    int(meta.get("width") or width),
                    int(meta.get("height") or height),
                    int(meta.get("length")),
                )
            positive = retime_conditioning_for_frames(
                positive, stored_frames, av_frame_count(int(width), int(height), int(length)),
            )
        if latent is None:
            # New-style file: the zero canvas was never written — rebuild it.
            latent = _empty_av_latent(width, height, length)
            if latent is None:
                log.warning(
                    "Conditioning cache unusable for seg #%d (no latent, rebuild failed); "
                    "falling back to a fresh encode.", segment_index + 1,
                )
                return None
        latent = move_to_device(latent)

        log.info("Conditioning cache HIT: seg #%d from %s (moved to CUDA)", segment_index + 1, cache_file.name)
        return {
            "positive": positive,
            "negative": negative,
            "latent": latent,
        }
        
    except Exception as exc:
        log.warning("Failed to load conditioning cache for seg #%d: %s", segment_index + 1, exc)
        return None


def load_conditioning_by_key(
    node_id: str | None,
    workflow_name: str | None,
    text_key: str,
) -> dict | None:
    """Load a cached text encoding straight from its known hash key.

    The second pass (「二次采样」) cannot recompute ``text_key`` — it needs the
    reference pixels that only exist while encoding runs — so it reads the key
    the first pass recorded (``load_segment_second_params``) and loads by it.
    Tensors are moved to CUDA, mirroring :func:`load_conditioning_cache`.
    """
    if not node_id or not str(text_key or "").strip():
        return None
    try:
        cache_dir = _get_cache_dir(node_id, workflow_name)
        cache_file = cache_dir / f"{_TEXT_PREFIX}_{str(text_key).strip()}.pt"
        if not cache_file.is_file():
            log.debug("No conditioning cache hit for key=%s", text_key)
            return None

        cache_data = torch.load(cache_file, map_location="cpu", weights_only=False)
        if not isinstance(cache_data, dict) or "positive" not in cache_data:
            return None

        def move_to_device(cond):
            if cond is None:
                return None
            if isinstance(cond, torch.Tensor):
                return cond.cuda()
            if isinstance(cond, (list, tuple)):
                out = []
                for item in cond:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        tensor = item[0]
                        if hasattr(tensor, "cuda"):
                            tensor = tensor.cuda()
                        out.append([tensor, item[1]])
                    else:
                        moved = move_to_device(item)
                        out.append(moved)
                return out
            if isinstance(cond, dict):
                return {k: move_to_device(v) for k, v in cond.items()}
            return cond

        return {
            "positive": move_to_device(cache_data.get("positive")),
            "negative": move_to_device(cache_data.get("negative")),
            "latent": None,
            "metadata": cache_data.get("metadata", {}),
        }
    except Exception as exc:
        log.warning("Failed to load conditioning by key %s: %s", text_key, exc)
        return None
